from pathlib import Path
import pickle
from datetime import date, timedelta
import os
import time

import numpy as np
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from app.api.saih_opendata import fetch_saih_history
    from app.core.config import SITES
except ImportError:
    from api.saih_opendata import fetch_saih_history
    from core.config import SITES


BASE_DIR = Path(__file__).resolve().parent
print(f"BASE_DIR: {BASE_DIR}")

VENTANA = 14
HORIZONTE = 7
LIVE_SAIH_LOOKBACK_DAYS = int(os.getenv("PREDICTION_LIVE_SAIH_LOOKBACK_DAYS", "21"))
LIVE_SAIH_CACHE_SECONDS = int(os.getenv("PREDICTION_LIVE_SAIH_CACHE_SECONDS", "1800"))
USE_LIVE_SAIH_WINDOW = os.getenv("PREDICTION_USE_LIVE_SAIH", "1").lower() in {"1", "true", "yes"}
SITES_BY_ID = {site["id"]: site for site in SITES}
_ARTIFACT_CACHE = {}
_DATASET_CACHE = {}
_LIVE_WINDOW_CACHE = {}


def _none_if_invalid(value):
    try:
        if value is None or pd.isna(value) or not np.isfinite(float(value)):
            return None
        return float(value)
    except Exception:
        return None


def _load_prediction_artifacts(site_id: str):
    cached = _ARTIFACT_CACHE.get(site_id)
    if cached is not None:
        return cached

    carpeta = BASE_DIR / "modelos_municipios" / site_id

    modelo_path = carpeta / "modelo.keras"
    scaler_x_path = carpeta / "scaler_X.pkl"
    scaler_nivel_path = carpeta / "scaler_nivel.pkl"
    scaler_caudal_path = carpeta / "scaler_caudal.pkl"
    features_path = carpeta / "features.pkl"

    if not modelo_path.exists():
        print(f"Modelo no encontrado para {site_id}")
        return None

    from tensorflow.keras.models import load_model

    artifacts = {
        "modelo": load_model(modelo_path, compile=False),
        "scaler_X": pickle.load(open(scaler_x_path, "rb")),
        "scaler_nivel": pickle.load(open(scaler_nivel_path, "rb")),
        "scaler_caudal": pickle.load(open(scaler_caudal_path, "rb")),
        "features": pickle.load(open(features_path, "rb")),
    }
    _ARTIFACT_CACHE[site_id] = artifacts
    return artifacts


def _load_site_dataset(site_id: str):
    cached = _DATASET_CACHE.get(site_id)
    if cached is not None:
        return cached.copy()

    dataset_path = BASE_DIR / "datasets_modelo_municipios" / f"{site_id}.csv"

    if not dataset_path.exists():
        print(f"Dataset no encontrado para {site_id}")
        return None

    df = pd.read_csv(dataset_path).dropna().reset_index(drop=True)
    _DATASET_CACHE[site_id] = df
    return df.copy()


def _add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["nivel_m"] = pd.to_numeric(df.get("nivel_m"), errors="coerce")
    df["caudal_m3s"] = pd.to_numeric(df.get("caudal_m3s"), errors="coerce")
    df["lluvia_mm"] = (
        pd.to_numeric(df["lluvia_mm"], errors="coerce").fillna(0)
        if "lluvia_mm" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    df["desbordamiento"] = (
        pd.to_numeric(df["desbordamiento"], errors="coerce").fillna(0)
        if "desbordamiento" in df.columns
        else pd.Series(0, index=df.index)
    )
    df["caudal_log"] = np.log1p(df["caudal_m3s"].clip(lower=0))
    df["nivel_lag1"] = df["nivel_m"].shift(1)
    df["caudal_lag1"] = df["caudal_log"].shift(1)
    df["lluvia_3d"] = df["lluvia_mm"].rolling(3, min_periods=1).sum()
    df["lluvia_7d"] = df["lluvia_mm"].rolling(7, min_periods=1).sum()
    df["nivel_diff"] = df["nivel_m"].diff()
    df["caudal_diff"] = df["caudal_m3s"].diff()
    df["nivel_media_3"] = df["nivel_m"].rolling(3, min_periods=1).mean()
    return df


def _records_to_daily_dataset(site_id: str, records: list[dict]) -> pd.DataFrame:
    site = SITES_BY_ID.get(site_id) or {}
    saih = site.get("saih") or {}
    signal_to_column = {
        saih.get("nivel"): "nivel_m",
        saih.get("caudal"): "caudal_m3s",
    }
    rows_by_ts = {}

    for record in records:
        column = signal_to_column.get(record.get("senal"))
        timestamp = record.get("fecha")

        if not column or not timestamp:
            continue

        row = rows_by_ts.setdefault(timestamp, {"fecha": timestamp})
        row[column] = record.get("valor")

    df = pd.DataFrame(rows_by_ts.values())
    if df.empty:
        return df

    df["fecha_dt"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha_dt"]).sort_values("fecha_dt")

    for column in ["nivel_m", "caudal_m3s"]:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    daily = (
        df.set_index("fecha_dt")[["nivel_m", "caudal_m3s"]]
        .resample("D")
        .mean()
        .dropna(how="any")
        .reset_index()
    )

    if daily.empty:
        return daily

    daily["fecha"] = daily["fecha_dt"].dt.strftime("%Y-%m-%d")
    daily["lluvia_mm"] = 0.0
    daily["desbordamiento"] = 0
    return daily[["fecha", "nivel_m", "caudal_m3s", "lluvia_mm", "desbordamiento"]]


def _load_live_prediction_window(site_id: str, base_df: pd.DataFrame, use_live_saih: bool | None = None) -> pd.DataFrame:
    use_live = USE_LIVE_SAIH_WINDOW if use_live_saih is None else bool(use_live_saih)

    if not use_live:
        return base_df

    cached = _LIVE_WINDOW_CACHE.get(site_id)
    now = time.time()
    if cached and (now - cached["epoch"]) < LIVE_SAIH_CACHE_SECONDS:
        return cached["df"].copy()

    site = SITES_BY_ID.get(site_id)
    if not site:
        return base_df

    saih = site.get("saih") or {}
    tags = [tag for tag in [saih.get("nivel"), saih.get("caudal")] if tag]
    if len(tags) < 2:
        return base_df

    end_date = date.today()
    start_date = end_date - timedelta(days=max(LIVE_SAIH_LOOKBACK_DAYS, VENTANA + 7))

    try:
        records = fetch_saih_history(tags, start_date, end_date)
        live_df = _records_to_daily_dataset(site_id, records)
    except Exception as exc:
        print(f"SAIH no disponible para ventana IA {site_id}: {exc}")
        return base_df

    if live_df.empty or len(live_df) < VENTANA:
        return base_df

    base_min = base_df[["fecha", "nivel_m", "caudal_m3s", "lluvia_mm", "desbordamiento"]].copy()
    combined = pd.concat([base_min.tail(VENTANA + 7), live_df], ignore_index=True)
    combined["fecha_dt"] = pd.to_datetime(combined["fecha"], errors="coerce")
    combined = (
        combined.dropna(subset=["fecha_dt"])
        .sort_values("fecha_dt")
        .drop_duplicates(subset=["fecha"], keep="last")
        .reset_index(drop=True)
    )
    combined = _add_model_features(combined)
    combined = combined.dropna().reset_index(drop=True)

    if len(combined) < VENTANA:
        return base_df

    print(f"IA {site_id}: usando ventana SAIH reciente hasta {combined['fecha'].iloc[-1]}")
    _LIVE_WINDOW_CACHE[site_id] = {"epoch": now, "df": combined}
    return combined.copy()


def _predict_from_window(artifacts, window_df: pd.DataFrame):
    x_df = window_df.copy()

    for col in artifacts["features"]:
        if col not in x_df.columns:
            x_df[col] = 0

    x_df = x_df[artifacts["features"]]
    data_x = x_df.values[-VENTANA:]
    x = artifacts["scaler_X"].transform(data_x)
    x = x.reshape(1, VENTANA, x.shape[1])

    pred_nivel_scaled, pred_caudal_scaled = artifacts["modelo"].predict(x, verbose=0)

    pred_nivel = artifacts["scaler_nivel"].inverse_transform(pred_nivel_scaled)[0]
    pred_caudal_log = artifacts["scaler_caudal"].inverse_transform(pred_caudal_scaled)[0]
    pred_caudal = np.expm1(pred_caudal_log)

    return pred_nivel, pred_caudal


def predecir_semana_municipio(site_id: str, use_live_saih: bool | None = None):
    try:
        artifacts = _load_prediction_artifacts(site_id)

        if artifacts is None:
            return []

        df = _load_site_dataset(site_id)

        if df is None:
            return []

        df = _add_model_features(df)
        df = _load_live_prediction_window(site_id, df, use_live_saih=use_live_saih)

        if len(df) < VENTANA:
            print(f"Muy pocos datos en {site_id}")
            return []

        pred_nivel, pred_caudal = _predict_from_window(artifacts, df)

        return [
            {"nivel": float(n), "caudal": float(c)}
            for n, c in zip(pred_nivel, pred_caudal)
        ]

    except Exception as e:
        print(f"Error IA en {site_id}: {e}")
        return []


def evaluar_fiabilidad_municipio(site_id: str):
    try:
        artifacts = _load_prediction_artifacts(site_id)

        if artifacts is None:
            return {"points": [], "metrics": {}, "error": "model_not_found"}

        df = _load_site_dataset(site_id)

        if df is None:
            return {"points": [], "metrics": {}, "error": "dataset_not_found"}

        required = {"nivel_m", "caudal_m3s"}
        if not required.issubset(df.columns):
            return {"points": [], "metrics": {}, "error": "dataset_missing_columns"}

        if len(df) < VENTANA + HORIZONTE:
            return {"points": [], "metrics": {}, "error": "not_enough_data"}

        window_df = df.iloc[-(VENTANA + HORIZONTE):-HORIZONTE].copy()
        actual_df = df.iloc[-HORIZONTE:].copy().reset_index(drop=True)
        pred_nivel, pred_caudal = _predict_from_window(artifacts, window_df)

        points = []
        for i, row in actual_df.iterrows():
            nivel_real = _none_if_invalid(row.get("nivel_m"))
            caudal_real = _none_if_invalid(row.get("caudal_m3s"))
            nivel_pred = _none_if_invalid(pred_nivel[i] if i < len(pred_nivel) else None)
            caudal_pred = _none_if_invalid(pred_caudal[i] if i < len(pred_caudal) else None)

            points.append({
                "date": str(row.get("fecha") or f"Dia {i + 1}"),
                "nivel_real": nivel_real,
                "nivel_pred": nivel_pred,
                "caudal_real": caudal_real,
                "caudal_pred": caudal_pred,
            })

        nivel_errors = [
            abs(p["nivel_real"] - p["nivel_pred"])
            for p in points
            if p["nivel_real"] is not None and p["nivel_pred"] is not None
        ]
        caudal_errors = [
            abs(p["caudal_real"] - p["caudal_pred"])
            for p in points
            if p["caudal_real"] is not None and p["caudal_pred"] is not None
        ]

        metrics = {
            "nivel_mae": round(float(np.mean(nivel_errors)), 3) if nivel_errors else None,
            "caudal_mae": round(float(np.mean(caudal_errors)), 3) if caudal_errors else None,
            "samples": len(points),
        }

        return {
            "points": points,
            "metrics": metrics,
            "error": None,
        }

    except Exception as e:
        print(f"Error evaluando fiabilidad IA en {site_id}: {e}")
        return {"points": [], "metrics": {}, "error": str(e)}

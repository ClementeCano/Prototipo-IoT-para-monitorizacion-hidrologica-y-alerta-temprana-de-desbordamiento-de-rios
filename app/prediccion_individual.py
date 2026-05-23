from pathlib import Path
import pickle

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
print(f"BASE_DIR: {BASE_DIR}")

VENTANA = 14
HORIZONTE = 7


def _none_if_invalid(value):
    try:
        if value is None or pd.isna(value) or not np.isfinite(float(value)):
            return None
        return float(value)
    except Exception:
        return None


def _load_prediction_artifacts(site_id: str):
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

    return {
        "modelo": load_model(modelo_path, compile=False),
        "scaler_X": pickle.load(open(scaler_x_path, "rb")),
        "scaler_nivel": pickle.load(open(scaler_nivel_path, "rb")),
        "scaler_caudal": pickle.load(open(scaler_caudal_path, "rb")),
        "features": pickle.load(open(features_path, "rb")),
    }


def _load_site_dataset(site_id: str):
    dataset_path = BASE_DIR / "datasets_modelo_municipios" / f"{site_id}.csv"

    if not dataset_path.exists():
        print(f"Dataset no encontrado para {site_id}")
        return None

    return pd.read_csv(dataset_path).dropna().reset_index(drop=True)


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


def predecir_semana_municipio(site_id: str):
    try:
        artifacts = _load_prediction_artifacts(site_id)

        if artifacts is None:
            return []

        df = _load_site_dataset(site_id)

        if df is None:
            return []

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

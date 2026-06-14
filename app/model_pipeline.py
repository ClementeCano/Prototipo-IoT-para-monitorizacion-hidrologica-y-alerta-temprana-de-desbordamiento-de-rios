from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd


BASE_COLUMNS = ["fecha", "nivel_m", "caudal_m3s", "lluvia_mm", "desbordamiento"]
EXCLUDED_FEATURE_COLUMNS = {"fecha", "fecha_dt", "nivel_m", "caudal_m3s", "desbordamiento"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return one clean daily row per date using only raw model columns."""
    if df is None or df.empty:
        return pd.DataFrame(columns=BASE_COLUMNS)

    base = df.copy()
    if "fecha" not in base.columns:
        return pd.DataFrame(columns=BASE_COLUMNS)

    base["fecha_dt"] = pd.to_datetime(base["fecha"], errors="coerce")
    base = base.dropna(subset=["fecha_dt"]).sort_values("fecha_dt")

    for column in ["nivel_m", "caudal_m3s", "lluvia_mm", "desbordamiento"]:
        if column not in base.columns:
            base[column] = 0
        base[column] = pd.to_numeric(base[column], errors="coerce")

    base["lluvia_mm"] = base["lluvia_mm"].fillna(0.0)
    base["desbordamiento"] = base["desbordamiento"].fillna(0).astype(int)
    base = base.dropna(subset=["nivel_m", "caudal_m3s"])

    base["fecha"] = base["fecha_dt"].dt.strftime("%Y-%m-%d")
    base = (
        base[BASE_COLUMNS]
        .drop_duplicates(subset=["fecha"], keep="last")
        .sort_values("fecha")
        .reset_index(drop=True)
    )
    return base


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the same feature set for dataset generation, training and inference."""
    out = normalize_base_dataframe(df)
    if out.empty:
        return out

    out["fecha_dt"] = pd.to_datetime(out["fecha"], errors="coerce")
    out["caudal_m3s"] = out["caudal_m3s"].clip(lower=0)
    out["caudal_log"] = np.log1p(out["caudal_m3s"])

    out["nivel_lag1"] = out["nivel_m"].shift(1)
    out["caudal_lag1"] = out["caudal_log"].shift(1)
    out["nivel_lag2"] = out["nivel_m"].shift(2)
    out["caudal_lag2"] = out["caudal_log"].shift(2)

    out["lluvia_3d"] = out["lluvia_mm"].rolling(3, min_periods=1).sum()
    out["lluvia_7d"] = out["lluvia_mm"].rolling(7, min_periods=1).sum()
    out["lluvia_media_7d"] = out["lluvia_mm"].rolling(7, min_periods=1).mean()

    out["nivel_diff"] = out["nivel_m"].diff()
    out["caudal_diff"] = out["caudal_m3s"].diff()
    out["nivel_media_3"] = out["nivel_m"].rolling(3, min_periods=1).mean()
    out["caudal_media_3"] = out["caudal_m3s"].rolling(3, min_periods=1).mean()
    out["nivel_media_7"] = out["nivel_m"].rolling(7, min_periods=1).mean()
    out["caudal_media_7"] = out["caudal_m3s"].rolling(7, min_periods=1).mean()

    day_of_year = out["fecha_dt"].dt.dayofyear.fillna(1).astype(float)
    out["dia_sin"] = np.sin(2 * np.pi * day_of_year / 366.0)
    out["dia_cos"] = np.cos(2 * np.pi * day_of_year / 366.0)

    out = out.drop(columns=["fecha_dt"]).dropna().reset_index(drop=True)
    return out


def model_feature_columns(df: pd.DataFrame, extra_exclude: Iterable[str] | None = None) -> list[str]:
    excluded = set(EXCLUDED_FEATURE_COLUMNS)
    if extra_exclude:
        excluded.update(extra_exclude)
    return [column for column in df.columns if column not in excluded]


def inverse_scaled_column(scaler, values: np.ndarray) -> np.ndarray:
    shape = values.shape
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(shape)


def inverse_caudal_log(scaler, values: np.ndarray) -> np.ndarray:
    return np.expm1(inverse_scaled_column(scaler, values)).clip(min=0)


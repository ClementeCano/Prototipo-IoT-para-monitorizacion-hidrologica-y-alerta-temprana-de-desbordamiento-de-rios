from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "salidas" / "dataset_base_modelo.csv"
OUTPUT_PATH = BASE_DIR / "salidas" / "dataset_modelo_final.csv"

# Lags hidrológicos
LAGS_NIVEL = [1, 2, 3]
LAGS_CAUDAL = [1, 2, 3]
LAGS_LLUVIA = [1, 2, 3, 4, 5, 6, 7]

# Acumulados
ACUM_LLUVIA = [3, 5, 7, 14]


def crear_features_por_municipio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("fecha").copy()

    # Transformación del caudal
    df["caudal_log"] = np.log1p(df["caudal_m3s"])

    # Lags de nivel
    for lag in LAGS_NIVEL:
        df[f"nivel_lag{lag}"] = df["nivel_m"].shift(lag)

    # Lags de caudal
    for lag in LAGS_CAUDAL:
        df[f"caudal_log_lag{lag}"] = df["caudal_log"].shift(lag)

    # Lags de lluvia
    for lag in LAGS_LLUVIA:
        df[f"lluvia_lag{lag}"] = df["lluvia_mm"].shift(lag)

    # Acumulados de lluvia basados en días anteriores
    for dias in ACUM_LLUVIA:
        df[f"lluvia_acum_{dias}d"] = df["lluvia_mm"].shift(1).rolling(dias).sum()

    # Variables calendario
    df["mes"] = df["fecha"].dt.month
    df["dia_ano"] = df["fecha"].dt.dayofyear

    # Sin/cos para estacionalidad
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12.0)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12.0)
    df["dia_ano_sin"] = np.sin(2 * np.pi * df["dia_ano"] / 365.25)
    df["dia_ano_cos"] = np.cos(2 * np.pi * df["dia_ano"] / 365.25)

    return df


def main():
    df = pd.read_csv(INPUT_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # Detectar columnas de municipio
    columnas_municipio = [c for c in df.columns if c.startswith("municipio_")]

    # Reconstruir nombre lógico de municipio para agrupar
    def extraer_municipio(row):
        activas = [c for c in columnas_municipio if row[c] == 1]
        return activas[0] if activas else "municipio_desconocido"

    df["grupo_municipio"] = df.apply(extraer_municipio, axis=1)

    lista = []
    for municipio, subdf in df.groupby("grupo_municipio"):
        subdf = crear_features_por_municipio(subdf)
        lista.append(subdf)

    df_final = pd.concat(lista, ignore_index=True)

    # No usar filas con lluvia del día ausente ni contexto ausente
    columnas_necesarias = [
        "nivel_m",
        "caudal_m3s",
        "caudal_log",
        "lluvia_mm",
    ]

    columnas_generadas = [
        c for c in df_final.columns
        if c.startswith("nivel_lag")
        or c.startswith("caudal_log_lag")
        or c.startswith("lluvia_lag")
        or c.startswith("lluvia_acum_")
    ]

    df_final = df_final.dropna(subset=columnas_necesarias + columnas_generadas).copy()

    # Mantener orden
    columnas_finales = (
        ["fecha", "nivel_m", "caudal_m3s", "caudal_log", "lluvia_mm", "desbordamiento",
         "nivel_interpolado", "caudal_interpolado", "lluvia_interpolada", "dato_original_completo"]
        + columnas_generadas
        + ["mes_sin", "mes_cos", "dia_ano_sin", "dia_ano_cos"]
        + columnas_municipio
    )

    columnas_finales = [c for c in columnas_finales if c in df_final.columns]
    df_final = df_final[columnas_finales].sort_values("fecha").reset_index(drop=True)

    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset final de modelo guardado en: {OUTPUT_PATH}")
    print(f"Filas finales: {len(df_final)}")


if __name__ == "__main__":
    main()
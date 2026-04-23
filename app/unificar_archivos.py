from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

CARPETA_MUNICIPIOS = BASE_DIR / "datos"
CARPETA_LLUVIA = BASE_DIR.parent / "datos_lluvia_aemet"
SALIDA_DIR = BASE_DIR / "salidas"
SALIDA_DIR.mkdir(exist_ok=True)

ARCHIVO_UNIFICADO = SALIDA_DIR / "dataset_unificado_limpio.csv"
ARCHIVO_MODELO_BASE = SALIDA_DIR / "dataset_base_modelo.csv"

MAX_GAP_INTERPOLABLE = 2  # días como máximo para interpolar nivel/caudal


# =========================================================
# UTILIDADES
# =========================================================
def normalizar_nombre_columna(col):
    if pd.isna(col):
        return ""

    col = str(col).strip().lower()

    reemplazos = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "³": "3"
    }

    for a, b in reemplazos.items():
        col = col.replace(a, b)

    return col


def mapear_columnas(df):
    columnas = {c: normalizar_nombre_columna(c) for c in df.columns}

    mapa = {}

    for original, limpio in columnas.items():

        if "fecha" in limpio:
            mapa["fecha"] = original

        elif "nivel" in limpio:
            mapa["nivel"] = original

        elif "caudal" in limpio:
            mapa["caudal"] = original

        elif "lluvia" in limpio:
            mapa["lluvia"] = original

        elif "desbord" in limpio:
            mapa["desbordamiento"] = original

    return mapa


def cargar_excel_municipio(ruta: Path) -> pd.DataFrame:
    df = pd.read_excel(ruta)

    # 🔥 Detectar columnas reales
    mapa = mapear_columnas(df)

    necesarias = ["fecha", "nivel", "caudal", "desbordamiento"]

    faltantes = [c for c in necesarias if c not in mapa]
    if faltantes:
        raise ValueError(f"{ruta.name} → faltan columnas: {faltantes}")

    # 🔥 Construir dataframe limpio
    df_limpio = pd.DataFrame({
        "fecha": df[mapa["fecha"]],
        "nivel (m)": df[mapa["nivel"]],
        "caudal (m³/s)": df[mapa["caudal"]],
        "desbordamiento": df[mapa["desbordamiento"]],
    })

    # Lluvia puede no estar en este archivo (se añade después)
    if "lluvia" in mapa:
        df_limpio["lluvia"] = df[mapa["lluvia"]]

    # =========================
    # LIMPIEZA
    # =========================
    df_limpio["fecha"] = pd.to_datetime(df_limpio["fecha"], dayfirst=True, errors="coerce")

    for col in ["nivel (m)", "caudal (m³/s)", "desbordamiento"]:
        df_limpio[col] = pd.to_numeric(df_limpio[col], errors="coerce")

    df_limpio = df_limpio.dropna(subset=["fecha"])
    df_limpio = df_limpio.sort_values("fecha").reset_index(drop=True)

    print(f"✔ {ruta.name} columnas usadas:", list(df_limpio.columns))

    return df_limpio


def cargar_lluvia(ruta: Path) -> pd.DataFrame:
    df = pd.read_excel(ruta)
    df.columns = [c.strip().lower() for c in df.columns]

    if "fecha" not in df.columns or "lluvia_mm" not in df.columns:
        raise ValueError(f"Formato de lluvia inválido en {ruta.name}")

    df = df[["fecha", "lluvia_mm"]].copy()
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
    df["lluvia_mm"] = pd.to_numeric(df["lluvia_mm"], errors="coerce")

    return df.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)


def interpolar_columna_con_mascara(serie: pd.Series, max_gap: int):
    original_nan = serie.isna()

    interpolada = serie.interpolate(
        method="linear",
        limit=max_gap,
        limit_direction="both",
        limit_area="inside",
    )

    fue_interpolado = original_nan & interpolada.notna()
    return interpolada, fue_interpolado.astype(int)


def procesar_municipio(ruta_excel: Path) -> pd.DataFrame:
    municipio = ruta_excel.stem
    ruta_lluvia = CARPETA_LLUVIA / f"{municipio}.xlsx"

    if not ruta_lluvia.exists():
        raise FileNotFoundError(f"No existe lluvia para {municipio}: {ruta_lluvia}")

    df_base = cargar_excel_municipio(ruta_excel)
    df_lluvia = cargar_lluvia(ruta_lluvia)

    df = df_base.merge(df_lluvia, on="fecha", how="left")
    df["municipio"] = municipio

    # Guardar máscara original
    nivel_original_nan = df["nivel (m)"].isna()
    caudal_original_nan = df["caudal (m³/s)"].isna()

    # Interpolación mínima solo en nivel/caudal
    df["nivel (m)"], df["nivel_interpolado"] = interpolar_columna_con_mascara(
        df["nivel (m)"], MAX_GAP_INTERPOLABLE
    )
    df["caudal (m³/s)"], df["caudal_interpolado"] = interpolar_columna_con_mascara(
        df["caudal (m³/s)"], MAX_GAP_INTERPOLABLE
    )

    # No tocar lluvia
    df["lluvia_interpolada"] = 0

    # Bandera de dato original completo
    df["dato_original_completo"] = (
        (~nivel_original_nan) &
        (~caudal_original_nan) &
        (df["lluvia_mm"].notna())
    ).astype(int)

    return df


def main():
    archivos = [
        f for f in sorted(CARPETA_MUNICIPIOS.glob("*.xlsx"))
        if f.name.lower() != "datos.xlsx"
    ]
    if not archivos:
        raise FileNotFoundError(f"No hay Excel en {CARPETA_MUNICIPIOS}")

    lista = []
    for ruta in archivos:
        print(f"Procesando {ruta.name}...")
        df = procesar_municipio(ruta)
        lista.append(df)

    df_total = pd.concat(lista, ignore_index=True)
    df_total = df_total.sort_values(["municipio", "fecha"]).reset_index(drop=True)

    df_total.to_csv(ARCHIVO_UNIFICADO, index=False)

    # Dataset base para modelo: aún sin lags
    df_modelo = df_total.copy()

    # Renombrar
    df_modelo = df_modelo.rename(
        columns={
            "nivel (m)": "nivel_m",
            "caudal (m³/s)": "caudal_m3s",
        }
    )

    # Mantener solo filas con targets presentes
    df_modelo = df_modelo.dropna(subset=["nivel_m", "caudal_m3s"]).copy()

    # Desbordamiento a entero cuando exista
    df_modelo["desbordamiento"] = pd.to_numeric(
        df_modelo["desbordamiento"], errors="coerce"
    )

    # One-hot de municipio
    df_modelo = pd.get_dummies(df_modelo, columns=["municipio"], dtype=int)

    df_modelo.to_csv(ARCHIVO_MODELO_BASE, index=False)

    print(f"\nGuardado unificado limpio: {ARCHIVO_UNIFICADO}")
    print(f"Guardado dataset base modelo: {ARCHIVO_MODELO_BASE}")


if __name__ == "__main__":
    main()
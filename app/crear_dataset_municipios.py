from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

try:
    from app.model_pipeline import add_model_features
except ImportError:
    from model_pipeline import add_model_features

# =========================
# RUTAS
# =========================
BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "datos"
OUTPUT_DIR = BASE_DIR / "datasets_modelo_municipios"
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# COLUMNAS
# =========================
COLUMNAS_USAR = {
    "FECHA": "fecha",
    "Nivel (m)": "nivel_m",
    "Caudal (m³/s)": "caudal_m3s",
    "lluvia_mm": "lluvia_mm",
    "Desbordamiento": "desbordamiento",
}


def limpiar_dataset(path_excel: Path) -> pd.DataFrame:
    df = pd.read_excel(path_excel)

    # Nos quedamos solo con las columnas necesarias
    df = df[list(COLUMNAS_USAR.keys())].copy()

    # Renombrar columnas
    df = df.rename(columns=COLUMNAS_USAR)

    # Fechas
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)

    # Numéricos
    for col in ["nivel_m", "caudal_m3s", "lluvia_mm", "desbordamiento"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ordenar
    df = df.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

    # Lluvia vacía = 0
    df["lluvia_mm"] = df["lluvia_mm"].fillna(0)

    # Eliminar filas sin nivel o caudal
    df = df.dropna(subset=["nivel_m", "caudal_m3s"]).reset_index(drop=True)

    # Desbordamiento vacío = 0
    df["desbordamiento"] = df["desbordamiento"].fillna(0).astype(int)

    return add_model_features(df)


def main():
    archivos = [
        f for f in INPUT_DIR.glob("*.xlsx")
        if f.stem.lower() != "datos"
    ]

    if not archivos:
        print(f"No hay archivos Excel en: {INPUT_DIR}")
        return

    print("Preparando datasets por municipio...")

    for archivo in archivos:
        municipio = archivo.stem

        try:
            df = limpiar_dataset(archivo)

            salida = OUTPUT_DIR / f"{municipio}.csv"
            df.to_csv(salida, index=False, encoding="utf-8-sig")

            print(f"✅ {municipio}: {len(df)} filas -> {salida}")

        except Exception as e:
            print(f"❌ Error en {municipio}: {e}")


if __name__ == "__main__":
    main()

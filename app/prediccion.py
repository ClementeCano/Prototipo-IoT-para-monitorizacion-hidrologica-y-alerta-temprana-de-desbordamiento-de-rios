import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from pathlib import Path
import traceback

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelos" / "modelo_lstm.keras"
SCALER_X_PATH = BASE_DIR / "modelos" / "scaler_X.pkl"
SCALER_NIVEL_PATH = BASE_DIR / "modelos" / "scaler_nivel.pkl"
SCALER_CAUDAL_PATH = BASE_DIR / "modelos" / "scaler_caudal.pkl"

VENTANA = 14
HORIZONTE = 7

# =========================
# CARGA
# =========================
modelo = load_model(MODEL_PATH, compile=False)


with open(SCALER_X_PATH, "rb") as f:
    scaler_X = pickle.load(f)

with open(SCALER_NIVEL_PATH, "rb") as f:
    scaler_nivel = pickle.load(f)

with open(SCALER_CAUDAL_PATH, "rb") as f:
    scaler_caudal = pickle.load(f)

# =========================
# NORMALIZAR MUNICIPIO
# =========================
def normalizar_municipio(nombre: str) -> str:
    mapping = {
        "Ascó": "Asco",
        "Castejón": "Castejon",
    }
    return mapping.get(nombre, nombre)

# =========================
# FILTRAR MUNICIPIO
# =========================
def filtrar_municipio(df: pd.DataFrame, municipio: str):
    municipio = normalizar_municipio(municipio)
    col = f"municipio_{municipio}"

    if col not in df.columns:
        print(f"⚠️ Municipio no encontrado: {municipio}")
        return None

    df_filtrado = df[df[col] == 1].copy()

    if len(df_filtrado) < VENTANA:
        print(f"⚠️ Datos insuficientes para {municipio}")
        return None

    return df_filtrado.sort_values("fecha").reset_index(drop=True)

# =========================
# FEATURE ENGINEERING (MISMO QUE TRAIN)
# =========================
def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["caudal_log"] = np.log1p(df["caudal_m3s"])

    df["nivel_lag1"] = df["nivel_m"].shift(1)
    df["caudal_lag1"] = df["caudal_log"].shift(1)

    df["lluvia_3d"] = df["lluvia_mm"].rolling(3).sum()
    df["lluvia_7d"] = df["lluvia_mm"].rolling(7).sum()

    df = df.dropna().reset_index(drop=True)

    return df

# =========================
# PREPARAR FEATURES
# =========================
def preparar_features(df: pd.DataFrame) -> list[str]:
    municipios = [c for c in df.columns if c.startswith("municipio_")]

    return [
        "nivel_m",
        "caudal_log",
        "lluvia_mm",
        "nivel_lag1",
        "caudal_lag1",
        "lluvia_3d",
        "lluvia_7d"
    ] + municipios

# =========================
# PREDICCIÓN
# =========================
import traceback

def predecir_semana(df: pd.DataFrame, site_name: str = None) -> list[list[float]]:
    print("🔍 Iniciando predicción para:", site_name)
    try:
        dataset_path = BASE_DIR / "salidas" / "dataset_modelo_final.csv"

        df = pd.read_csv(dataset_path)

        if site_name:
            df = filtrar_municipio(df, site_name)
            if df is None:
                return []

        excluir = {"fecha", "nivel_m", "caudal_m3s", "desbordamiento"}
        features = [c for c in df.columns if c not in excluir]

        #print("N_FEATURES_PRED:", len(features))
        #print("FEATURES_PRED:", features[:10], "...")

        if len(df) < VENTANA:
            print("⚠️ No hay suficientes datos")
            return []

        ultima = df[features].values[-VENTANA:]

        X = scaler_X.transform(ultima)
        X = X.reshape(1, VENTANA, len(features))

        pred_nivel_scaled, pred_caudal_scaled = modelo.predict(X, verbose=0)

        pred_nivel = scaler_nivel.inverse_transform(pred_nivel_scaled)[0]
        pred_caudal_log = scaler_caudal.inverse_transform(pred_caudal_scaled)[0]
        pred_caudal = np.expm1(pred_caudal_log)

        pred_nivel = np.clip(pred_nivel, 0, None)
        pred_caudal = np.clip(pred_caudal, 0, None)

        return [[float(n), float(c)] for n, c in zip(pred_nivel, pred_caudal)]

    except Exception as e:
        print("❌ Error en predecir_semana:")
        print(repr(e))
        traceback.print_exc()
        return []
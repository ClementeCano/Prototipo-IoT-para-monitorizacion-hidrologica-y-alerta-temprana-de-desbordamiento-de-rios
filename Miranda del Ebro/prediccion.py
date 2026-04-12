import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

MODEL_PATH = "modelo_lstm.h5"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_y.pkl"

VENTANA = 7
HORIZONTE = 7

# Cargar modelo SIN recompilar
modelo = load_model(MODEL_PATH, compile=False)

with open(SCALER_X_PATH, "rb") as f:
    scaler_X = pickle.load(f)

with open(SCALER_Y_PATH, "rb") as f:
    scaler_y = pickle.load(f)


def normalizar_municipio(nombre: str) -> str:
    mapping = {
        "Ascó": "Asco",
        "Castejón": "Castejon",
    }
    return mapping.get(nombre, nombre)


def filtrar_municipio(df: pd.DataFrame, municipio: str):
    municipio = normalizar_municipio(municipio)
    col = f"municipio_{municipio}"

    if col not in df.columns:
        print(f"⚠️ Municipio no encontrado en dataset: {municipio}")
        return None

    df_filtrado = df[df[col] == 1].copy()

    if len(df_filtrado) < VENTANA:
        print(f"⚠️ No hay suficientes datos para {municipio}")
        return None

    return df_filtrado.sort_values("fecha").reset_index(drop=True)


def preparar_features(df: pd.DataFrame) -> list[str]:
    columnas_municipio = [c for c in df.columns if c.startswith("municipio_")]
    return ["nivel_m", "caudal_m3s", "lluvia_mm"] + columnas_municipio


def predecir_semana(df: pd.DataFrame, site_name: str = None) -> list[list[float]]:
    try:
        if site_name:
            df = filtrar_municipio(df, site_name)
            if df is None:
                return []

        features = preparar_features(df)

        ultima = df[features].values[-VENTANA:]

        X = scaler_X.transform(ultima)
        X = X.reshape(1, VENTANA, len(features))

        pred_scaled = modelo.predict(X, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled)
        pred = pred.reshape(HORIZONTE, 2)

        pred = np.clip(pred, a_min=0, a_max=None)

        return pred.tolist()

    except Exception as e:
        print(f"❌ Error en predecir_semana: {e}")
        return []
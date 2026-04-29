from pathlib import Path
import numpy as np
import pickle
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
print(f"📱BASE_DIR: {BASE_DIR}")

VENTANA = 14
HORIZONTE = 7


def predecir_semana_municipio(site_id: str):
    try:
        carpeta = BASE_DIR / "modelos_municipios" / site_id

        modelo_path = carpeta / "modelo.keras"
        scaler_x_path = carpeta / "scaler_X.pkl"
        scaler_nivel_path = carpeta / "scaler_nivel.pkl"
        scaler_caudal_path = carpeta / "scaler_caudal.pkl"

        if not modelo_path.exists():
            print(f"BASE_DIR: {BASE_DIR}")
            print(f"Modelo_path: {modelo_path}")
            print(f"❌ Modelo no encontrado para {site_id}")
            return []

        # =========================
        # CARGA MODELO Y SCALERS
        # =========================
        from tensorflow.keras.models import load_model
        modelo = load_model(modelo_path, compile=False)

        with open(scaler_x_path, "rb") as f:
            scaler_X = pickle.load(f)

        with open(scaler_nivel_path, "rb") as f:
            scaler_nivel = pickle.load(f)

        with open(scaler_caudal_path, "rb") as f:
            scaler_caudal = pickle.load(f)

        # =========================
        # CARGA DATASET MUNICIPIO
        # =========================
        dataset_path = BASE_DIR / "datasets_modelo_municipios" / f"{site_id}.csv"

        if not dataset_path.exists():
            print(f"❌ Dataset no encontrado para {site_id}")
            return []

        df = pd.read_csv(dataset_path)

        # limpieza mínima
        df = df.dropna().reset_index(drop=True)

        if len(df) < VENTANA:
            print(f"⚠️ Muy pocos datos en {site_id}")
            return []

        # =========================
        # FEATURES
        # =========================
        excluir = {"fecha", "nivel_m", "caudal_m3s", "desbordamiento"}
        features = [c for c in df.columns if c not in excluir]

        data_X = df[features].values
        data_X = data_X[-VENTANA:]  # últimos datos

        # reshape
        X = scaler_X.transform(data_X)
        X = X.reshape(1, VENTANA, X.shape[1])

        # =========================
        # PREDICCIÓN
        # =========================
        pred_nivel_scaled, pred_caudal_scaled = modelo.predict(X)

        pred_nivel = scaler_nivel.inverse_transform(pred_nivel_scaled)[0]
        pred_caudal_log = scaler_caudal.inverse_transform(pred_caudal_scaled)[0]

        pred_caudal = np.expm1(pred_caudal_log)

        # =========================
        # FORMATO FINAL
        # =========================
        pred = [
            [float(n), float(c)]
            for n, c in zip(pred_nivel, pred_caudal)
        ]

        return pred

    except Exception as e:
        print(f"❌ Error IA en {site_id}: {e}")
        return []
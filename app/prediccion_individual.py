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
        features_path = carpeta / "features.pkl"

        if not modelo_path.exists():
            print(f"❌ Modelo no encontrado para {site_id}")
            return []

        # =========================
        # CARGA MODELO Y SCALERS
        # =========================
        from tensorflow.keras.models import load_model
        modelo = load_model(modelo_path, compile=False)

        scaler_X = pickle.load(open(scaler_x_path, "rb"))
        scaler_nivel = pickle.load(open(scaler_nivel_path, "rb"))
        scaler_caudal = pickle.load(open(scaler_caudal_path, "rb"))

        # 🔥 cargar features originales
        features = pickle.load(open(features_path, "rb"))

        # =========================
        # CARGA DATASET
        # =========================
        dataset_path = BASE_DIR / "datasets_modelo_municipios" / f"{site_id}.csv"

        if not dataset_path.exists():
            print(f"❌ Dataset no encontrado para {site_id}")
            return []

        df = pd.read_csv(dataset_path)
        df = df.dropna().reset_index(drop=True)

        if len(df) < VENTANA:
            print(f"⚠️ Muy pocos datos en {site_id}")
            return []

        # =========================
        # 🔥 AJUSTAR FEATURES CORRECTAMENTE
        # =========================
        X_df = df.copy()

        # añadir columnas faltantes
        for col in features:
            if col not in X_df.columns:
                X_df[col] = 0

        # ordenar columnas EXACTAMENTE igual
        X_df = X_df[features]

        # coger últimos datos
        data_X = X_df.values[-VENTANA:]

        # =========================
        # ESCALADO
        # =========================
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
            {"nivel": float(n), "caudal": float(c)}
            for n, c in zip(pred_nivel, pred_caudal)
        ]

        return pred

    except Exception as e:
        print(f"❌ Error IA en {site_id}: {e}")
        return []
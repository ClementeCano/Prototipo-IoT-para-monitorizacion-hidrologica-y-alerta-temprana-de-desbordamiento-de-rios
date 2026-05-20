import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import pickle

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score
)

# from app.umbrales import cargar_umbrales
from umbrales import cargar_umbrales


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "datasets_modelo_municipios"
MODEL_DIR = BASE_DIR / "modelos_municipios"

VENTANA = 14
HORIZONTE = 7

# 🔥 limitar evaluación para no tardar siglos
MAX_ITERS = 50

UMBRALES = cargar_umbrales()

# =========================
# CARGAR MODELO
# =========================
def cargar_modelo_municipio(municipio):

    carpeta = MODEL_DIR / municipio

    from tensorflow.keras.models import load_model

    print("👉 cargando modelo...")

    modelo = load_model(
        carpeta / "modelo.keras",
        compile=False
    )

    scaler_X = pickle.load(open(carpeta / "scaler_X.pkl", "rb"))
    scaler_nivel = pickle.load(open(carpeta / "scaler_nivel.pkl", "rb"))
    scaler_caudal = pickle.load(open(carpeta / "scaler_caudal.pkl", "rb"))
    features = pickle.load(open(carpeta / "features.pkl", "rb"))

    print("✅ modelo cargado")

    return (
        modelo,
        scaler_X,
        scaler_nivel,
        scaler_caudal,
        features
    )


# =========================
# PREDICCIÓN RÁPIDA
# =========================
def predecir_rapido(
    modelo,
    scaler_X,
    scaler_nivel,
    scaler_caudal,
    features,
    ventana_df
):

    X_df = ventana_df.copy()

    # asegurar mismas columnas
    for col in features:
        if col not in X_df.columns:
            X_df[col] = 0

    # mismo orden exacto
    X_df = X_df[features]

    X = scaler_X.transform(X_df.values)

    X = X.reshape(
        1,
        X.shape[0],
        X.shape[1]
    )

    # 🔥 sin spam consola
    pred_nivel_scaled, pred_caudal_scaled = modelo.predict(
        X,
        verbose=0
    )

    pred_nivel = scaler_nivel.inverse_transform(
        pred_nivel_scaled
    )[0]

    pred_caudal = np.expm1(
        scaler_caudal.inverse_transform(
            pred_caudal_scaled
        )[0]
    )

    return pred_nivel, pred_caudal


# =========================
# EVALUAR MUNICIPIO
# =========================
def evaluar_municipio(path_csv):

    municipio = path_csv.stem.lower()

    print(f"\n📍 {municipio}")

    df = pd.read_csv(path_csv)

    df = df.dropna().reset_index(drop=True)

    # 🔥 evaluar solo parte final
    df = df.tail(120)

    if len(df) < VENTANA + HORIZONTE:
        print("⚠️ Muy pocos datos")
        return

    # =========================
    # CARGAR MODELO
    # =========================
    try:
        (
            modelo,
            scaler_X,
            scaler_nivel,
            scaler_caudal,
            features
        ) = cargar_modelo_municipio(municipio)

    except Exception as e:
        print("⚠️ Modelo no encontrado")
        print(e)
        return

    reales_nivel = []
    pred_nivel = []

    reales_caudal = []
    pred_caudal = []

    y_true_alert = []
    y_pred_alert = []

    umbral = UMBRALES.get(municipio, {}).get("alerta")

    total_iters = min(
        len(df) - VENTANA - HORIZONTE,
        MAX_ITERS
    )

    print(f"🚀 iteraciones: {total_iters}")

    # =========================
    # LOOP TEMPORAL
    # =========================
    for i in range(total_iters):

        #print(f"iteración {i+1}/{total_iters}")

        ventana_df = df.iloc[i:i+VENTANA]

        futuro_df = df.iloc[
            i+VENTANA:
            i+VENTANA+HORIZONTE
        ]

        pred_niveles, pred_caudales = predecir_rapido(
            modelo,
            scaler_X,
            scaler_nivel,
            scaler_caudal,
            features,
            ventana_df
        )

        real_niveles = futuro_df["nivel_m"].values
        real_caudales = futuro_df["caudal_m3s"].values

        reales_nivel.extend(real_niveles)
        pred_nivel.extend(pred_niveles)

        reales_caudal.extend(real_caudales)
        pred_caudal.extend(pred_caudales)

        # =========================
        # ALERTAS
        # =========================
        if umbral:

            real_alert = int(
                np.max(real_niveles) > umbral
            )

            pred_alert = int(
                np.max(pred_niveles) > umbral
            )

            y_true_alert.append(real_alert)
            y_pred_alert.append(pred_alert)

    # =========================
    # MÉTRICAS NIVEL
    # =========================
    mae_nivel = mean_absolute_error(
        reales_nivel,
        pred_nivel
    )

    rmse_nivel = np.sqrt(
        mean_squared_error(
            reales_nivel,
            pred_nivel
        )
    )

    # =========================
    # MÉTRICAS CAUDAL
    # =========================
    mae_caudal = mean_absolute_error(
        reales_caudal,
        pred_caudal
    )

    rmse_caudal = np.sqrt(
        mean_squared_error(
            reales_caudal,
            pred_caudal
        )
    )

    print("\n📊 RESULTADOS")

    print(
        f"NIVEL  → "
        f"MAE: {mae_nivel:.3f} | "
        f"RMSE: {rmse_nivel:.3f}"
    )

    print(
        f"CAUDAL → "
        f"MAE: {mae_caudal:.3f} | "
        f"RMSE: {rmse_caudal:.3f}"
    )

    # =========================
    # MÉTRICAS ALERTA
    # =========================
    if len(y_true_alert) > 0:

        precision = precision_score(
            y_true_alert,
            y_pred_alert,
            zero_division=0
        )

        recall = recall_score(
            y_true_alert,
            y_pred_alert,
            zero_division=0
        )

        f1 = f1_score(
            y_true_alert,
            y_pred_alert,
            zero_division=0
        )

        print(
            f"ALERTA → "
            f"Precision: {precision:.3f} | "
            f"Recall: {recall:.3f} | "
            f"F1: {f1:.3f}"
        )

        if recall < 0.6:
            print("⚠️ MAL: se escapan desbordamientos")

        elif precision < 0.3:
            print("⚠️ Muchas falsas alarmas")

        else:
            print("🔥 BUEN SISTEMA DE ALERTA")


# =========================
# MAIN
# =========================
def main():

    archivos = list(DATA_DIR.glob("*.csv"))

    print("🚨 Evaluando modelo OPTIMIZADO...")

    for archivo in archivos:
        evaluar_municipio(archivo)


if __name__ == "__main__":
    main()
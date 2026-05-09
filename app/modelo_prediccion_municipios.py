import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "datasets_modelo_municipios"
MODEL_DIR = BASE_DIR / "modelos_municipios"
MODEL_DIR.mkdir(exist_ok=True)

VENTANA = 14
HORIZONTE = 7

# =========================
# CREAR VENTANAS
# =========================
def crear_ventanas(data, nivel, caudal, ventana, horizonte):
    X, y_nivel, y_caudal = [], [], []

    for i in range(len(data) - ventana - horizonte + 1):
        X.append(data[i:i+ventana])
        y_nivel.append(nivel[i+ventana:i+ventana+horizonte])
        y_caudal.append(caudal[i+ventana:i+ventana+horizonte])

    return np.array(X), np.array(y_nivel), np.array(y_caudal)

# =========================
# ENTRENAMIENTO
# =========================
def entrenar_municipio(path_csv):
    municipio = path_csv.stem.lower()
    print(f"\n📍 {municipio}")

    df = pd.read_csv(path_csv)

    if len(df) < 50:
        print("⚠️ Muy pocos datos, saltando")
        return

    # =========================
    # FEATURE ENGINEERING 🔥
    # =========================
    df["nivel_diff"] = df["nivel_m"].diff()
    df["caudal_diff"] = df["caudal_m3s"].diff()
    df["nivel_media_3"] = df["nivel_m"].rolling(3).mean()

    df = df.dropna()

    # =========================
    # FEATURES
    # =========================
    excluir = {"fecha", "nivel_m", "caudal_m3s", "desbordamiento"}
    features = [c for c in df.columns if c not in excluir]

    X_data = df[features].values
    nivel = df["nivel_m"].values
    caudal = df["caudal_log"].values

    # =========================
    # CLIPPING (outliers)
    # =========================
    nivel = np.clip(nivel, None, np.percentile(nivel, 95))
    caudal = np.clip(caudal, None, np.percentile(caudal, 95))

    # =========================
    # ESCALADO
    # =========================
    scaler_X = MinMaxScaler()
    scaler_nivel = MinMaxScaler()
    scaler_caudal = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X_data)
    nivel_scaled = scaler_nivel.fit_transform(nivel.reshape(-1, 1)).flatten()
    caudal_scaled = scaler_caudal.fit_transform(caudal.reshape(-1, 1)).flatten()    

    # =========================
    # VENTANAS
    # =========================
    X, y_nivel, y_caudal = crear_ventanas(
        X_scaled, nivel_scaled, caudal_scaled, VENTANA, HORIZONTE
    )

    if len(X) < 10:
        print("⚠️ No hay suficientes ventanas, saltando")
        return

    # =========================
    # SPLIT TEMPORAL 🔥
    # =========================
    split = int(len(X) * 0.8)

    X_train, X_val = X[:split], X[split:]
    y_nivel_train, y_nivel_val = y_nivel[:split], y_nivel[split:]
    y_caudal_train, y_caudal_val = y_caudal[:split], y_caudal[split:]

    # =========================
    # MODELO MEJORADO 🔥
    # =========================
    inputs = Input(shape=(VENTANA, X.shape[2]))

    x = LSTM(64, return_sequences=True)(inputs)
    x = LSTM(32)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)

    output_nivel = Dense(HORIZONTE, name="nivel")(x)
    output_caudal = Dense(HORIZONTE, name="caudal")(x)

    model = Model(inputs=inputs, outputs=[output_nivel, output_caudal])

    optimizer = Adam(learning_rate=0.0005)

    model.compile(
        optimizer=optimizer,
        loss={
            "nivel": "mse",
            "caudal": "mse"
        }
    )

    # =========================
    # CALLBACKS 🔥
    # =========================
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5
    )

    # =========================
    # TRAIN 🔥
    # =========================
    model.fit(
        X_train,
        {
            "nivel": y_nivel_train,
            "caudal": y_caudal_train
        },
        validation_data=(
            X_val,
            {
                "nivel": y_nivel_val,
                "caudal": y_caudal_val
            }
        ),
        epochs=200,
        batch_size=16,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )

    # =========================
    # MÉTRICAS 🔥
    # =========================
    pred_nivel, pred_caudal = model.predict(X_val)

    mae_nivel = np.mean(np.abs(pred_nivel - y_nivel_val))
    mae_caudal = np.mean(np.abs(pred_caudal - y_caudal_val))

    print(f"📊 MAE nivel: {mae_nivel:.4f}")
    print(f"📊 MAE caudal: {mae_caudal:.4f}")

    # =========================
    # GUARDAR
    # =========================
    out_dir = MODEL_DIR / municipio
    out_dir.mkdir(exist_ok=True)

    model.save(out_dir / "modelo.keras")

    with open(out_dir / "scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)

    with open(out_dir / "scaler_nivel.pkl", "wb") as f:
        pickle.dump(scaler_nivel, f)

    with open(out_dir / "scaler_caudal.pkl", "wb") as f:
        pickle.dump(scaler_caudal, f)

    with open(out_dir / "features.pkl", "wb") as f:
        pickle.dump(features, f)

    print("✅ Modelo guardado")


# =========================
# MAIN
# =========================
def main():
    archivos = list(DATA_DIR.glob("*.csv"))

    print("🚀 Entrenando modelos por municipio...")

    for archivo in archivos:
        entrenar_municipio(archivo)


if __name__ == "__main__":
    main()
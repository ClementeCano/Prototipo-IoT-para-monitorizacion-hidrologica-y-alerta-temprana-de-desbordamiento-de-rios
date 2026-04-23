from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Model

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "salidas" / "dataset_modelo_final.csv"
MODELOS_DIR = BASE_DIR / "modelos"
MODELOS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELOS_DIR / "modelo_lstm.keras"
SCALER_X_PATH = MODELOS_DIR / "scaler_X.pkl"
SCALER_NIVEL_PATH = MODELOS_DIR / "scaler_nivel.pkl"
SCALER_CAUDAL_PATH = MODELOS_DIR / "scaler_caudal.pkl"

VENTANA = 14
HORIZONTE = 7
TRAIN_RATIO = 0.85


def crear_ventanas(data_x, target_nivel, target_caudal, ventana, horizonte):
    X, y_nivel, y_caudal = [], [], []

    for i in range(len(data_x) - ventana - horizonte + 1):
        X.append(data_x[i:i + ventana])
        y_nivel.append(target_nivel[i + ventana:i + ventana + horizonte])
        y_caudal.append(target_caudal[i + ventana:i + ventana + horizonte])

    return np.array(X), np.array(y_nivel), np.array(y_caudal)


def main():
    df = pd.read_csv(DATASET_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.sort_values("fecha").reset_index(drop=True)

    excluir = {"fecha", "nivel_m", "caudal_m3s", "desbordamiento"}
    features = [c for c in df.columns if c not in excluir]

    X_raw = df[features].values
    y_nivel = df["nivel_m"].values
    y_caudal = df["caudal_log"].values  # ya viene calculado

    X, y_nivel, y_caudal = crear_ventanas(
        X_raw, y_nivel, y_caudal, VENTANA, HORIZONTE
    )

    if len(X) == 0:
        raise RuntimeError("No se pudieron crear ventanas. Revisa el dataset.")

    train_size = int(len(X) * TRAIN_RATIO)

    X_train, X_test = X[:train_size], X[train_size:]
    y_nivel_train, y_nivel_test = y_nivel[:train_size], y_nivel[train_size:]
    y_caudal_train, y_caudal_test = y_caudal[:train_size], y_caudal[train_size:]

    scaler_X = MinMaxScaler()
    scaler_nivel = MinMaxScaler()
    scaler_caudal = MinMaxScaler()

    X_train_2d = X_train.reshape(-1, X_train.shape[2])
    X_test_2d = X_test.reshape(-1, X_test.shape[2])

    X_train_scaled = scaler_X.fit_transform(X_train_2d).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test_2d).reshape(X_test.shape)

    y_nivel_train_scaled = scaler_nivel.fit_transform(y_nivel_train)
    y_nivel_test_scaled = scaler_nivel.transform(y_nivel_test)

    y_caudal_train_scaled = scaler_caudal.fit_transform(y_caudal_train)
    y_caudal_test_scaled = scaler_caudal.transform(y_caudal_test)

    inp = Input(shape=(VENTANA, X.shape[2]))
    x = LSTM(128, return_sequences=True)(inp)
    x = Dropout(0.2)(x)
    x = LSTM(64)(x)
    x = Dense(64, activation="relu")(x)

    out_nivel = Dense(HORIZONTE, name="nivel")(x)
    out_caudal = Dense(HORIZONTE, name="caudal")(x)

    modelo = Model(inputs=inp, outputs=[out_nivel, out_caudal])

    modelo.compile(
        optimizer="adam",
        loss={"nivel": "mse", "caudal": "mse"},
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
    )

    modelo.fit(
        X_train_scaled,
        {"nivel": y_nivel_train_scaled, "caudal": y_caudal_train_scaled},
        validation_data=(
            X_test_scaled,
            {"nivel": y_nivel_test_scaled, "caudal": y_caudal_test_scaled},
        ),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )

    modelo.save(MODEL_PATH)

    with open(SCALER_X_PATH, "wb") as f:
        pickle.dump(scaler_X, f)

    with open(SCALER_NIVEL_PATH, "wb") as f:
        pickle.dump(scaler_nivel, f)

    with open(SCALER_CAUDAL_PATH, "wb") as f:
        pickle.dump(scaler_caudal, f)

    print(f"Modelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
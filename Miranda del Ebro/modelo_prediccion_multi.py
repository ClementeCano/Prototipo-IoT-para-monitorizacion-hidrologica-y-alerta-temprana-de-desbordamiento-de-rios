import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# CONFIGURACIÓN
# =========================
DATASET_PATH = "dataset_modelo.csv"
MODEL_PATH = "modelo_lstm.h5"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_y.pkl"

VENTANA = 7
HORIZONTE = 7

# =========================
# CARGA DE DATOS
# =========================
df = pd.read_csv(DATASET_PATH)
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
df = df.sort_values("fecha").reset_index(drop=True)

features = ["nivel_m", "caudal_m3s", "lluvia_mm"] + [c for c in df.columns if c.startswith("municipio_")]
targets = ["nivel_m", "caudal_m3s"]

data_X = df[features].values
data_y = df[targets].values

def crear_ventanas(data, y, ventana=7, horizonte=7):
    X, y_out = [], []

    for i in range(len(data) - ventana - horizonte + 1):
        X.append(data[i:i+ventana])
        y_out.append(y[i+ventana:i+ventana+horizonte])

    return np.array(X), np.array(y_out)

X, y = crear_ventanas(data_X, data_y, ventana=VENTANA, horizonte=HORIZONTE)

# Salida: (muestras, 7, 2) -> (muestras, 14)
y = y.reshape(y.shape[0], -1)

print("X:", X.shape)
print("y:", y.shape)

# =========================
# TRAIN / TEST
# =========================
train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# =========================
# ESCALADO
# =========================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_2d = X_train.reshape(-1, X_train.shape[2])
X_test_2d = X_test.reshape(-1, X_test.shape[2])

X_train_scaled = scaler_X.fit_transform(X_train_2d).reshape(X_train.shape)
X_test_scaled = scaler_X.transform(X_test_2d).reshape(X_test.shape)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# =========================
# MODELO
# =========================
input_hist = Input(shape=(VENTANA, X.shape[2]), name="historial")
x = LSTM(64)(input_hist)
x = Dense(64, activation="relu")(x)
output = Dense(HORIZONTE * 2, name="salida")(x)

modelo = Model(inputs=input_hist, outputs=output)
modelo.compile(optimizer="adam", loss="mse")

modelo.summary()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

modelo.fit(
    X_train_scaled,
    y_train_scaled,
    validation_data=(X_test_scaled, y_test_scaled),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# =========================
# GUARDADO
# =========================
modelo.save(MODEL_PATH)

with open(SCALER_X_PATH, "wb") as f:
    pickle.dump(scaler_X, f)

with open(SCALER_Y_PATH, "wb") as f:
    pickle.dump(scaler_y, f)

print("✅ Modelo y scalers guardados correctamente")
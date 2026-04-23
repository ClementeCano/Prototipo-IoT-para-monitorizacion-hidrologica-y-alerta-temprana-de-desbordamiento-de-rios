from pathlib import Path
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

# =========================
# CONFIG
# =========================
# Carpeta base del script actual
BASE_DIR = Path(__file__).resolve().parent

# =========================
# RUTAS
# =========================
DATASET_PATH = BASE_DIR / "salidas" / "dataset_modelo_final.csv"

MODEL_PATH = BASE_DIR / "modelos" / "modelo_lstm.keras"
SCALER_X_PATH = BASE_DIR / "modelos" / "scaler_X.pkl"
SCALER_NIVEL_PATH = BASE_DIR / "modelos" / "scaler_nivel.pkl"
SCALER_CAUDAL_PATH = BASE_DIR / "modelos" / "scaler_caudal.pkl"

VENTANA = 14
HORIZONTE = 7

# =========================
# CARGA
# =========================
df = pd.read_csv(DATASET_PATH)
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
df = df.sort_values("fecha").reset_index(drop=True)

# =========================
# FEATURE ENGINEERING (igual que entrenamiento)
# =========================
# df["caudal_log"] = np.log1p(df["caudal_m3s"])

# df["nivel_lag1"] = df["nivel_m"].shift(1)
# df["caudal_lag1"] = df["caudal_log"].shift(1)

# df["lluvia_3d"] = df["lluvia_mm"].rolling(3).sum()
# df["lluvia_7d"] = df["lluvia_mm"].rolling(7).sum()

# df = df.dropna().reset_index(drop=True)

# =========================
# FEATURES
# =========================

excluir = {"fecha", "nivel_m", "caudal_m3s", "desbordamiento"}

features = [c for c in df.columns if c not in excluir]

data_X = df[features].values
nivel = df["nivel_m"].values
caudal = df["caudal_log"].values
municipios_cols = [c for c in df.columns if c.startswith("municipio_")] 

# =========================
# CREAR VENTANAS
# =========================
def crear_ventanas(data, nivel, caudal, ventana, horizonte):
    X, y_nivel, y_caudal, municipios = [], [], [], []

    for i in range(len(data) - ventana - horizonte + 1):
        X.append(data[i:i+ventana])
        y_nivel.append(nivel[i+ventana:i+ventana+horizonte])
        y_caudal.append(caudal[i+ventana:i+ventana+horizonte])

        municipios.append(data[i+ventana-1, -len(municipios_cols):])

    return (
        np.array(X),
        np.array(y_nivel),
        np.array(y_caudal),
        np.array(municipios),
    )

X, y_nivel, y_caudal, municipios = crear_ventanas(
    data_X, nivel, caudal, VENTANA, HORIZONTE
)

# =========================
# SPLIT
# =========================
train_size = int(len(X) * 0.85)

X_test = X[train_size:]
y_nivel_test = y_nivel[train_size:]
y_caudal_test = y_caudal[train_size:]
municipios_test = municipios[train_size:]

# =========================
# CARGAR MODELO
# =========================
modelo = load_model(MODEL_PATH, compile=False)

with open(SCALER_X_PATH, "rb") as f:
    scaler_X = pickle.load(f)

with open(SCALER_NIVEL_PATH, "rb") as f:
    scaler_nivel = pickle.load(f)

with open(SCALER_CAUDAL_PATH, "rb") as f:
    scaler_caudal = pickle.load(f)

# =========================
# ESCALADO
# =========================
X_test_2d = X_test.reshape(-1, X_test.shape[2])
X_test_scaled = scaler_X.transform(X_test_2d).reshape(X_test.shape)

# =========================
# PREDICCIÓN
# =========================
pred_nivel_scaled, pred_caudal_scaled = modelo.predict(X_test_scaled)

# Desescalar
pred_nivel = scaler_nivel.inverse_transform(pred_nivel_scaled)
pred_caudal_log = scaler_caudal.inverse_transform(pred_caudal_scaled)

# Volver a valores reales
pred_caudal = np.expm1(pred_caudal_log)

# Real también en escala original
y_caudal_real = np.expm1(y_caudal_test)

# =========================
# MÉTRICAS GLOBALES
# =========================
mae_nivel = mean_absolute_error(y_nivel_test, pred_nivel)
rmse_nivel = np.sqrt(mean_squared_error(y_nivel_test, pred_nivel))

mae_caudal = mean_absolute_error(y_caudal_real, pred_caudal)
rmse_caudal = np.sqrt(mean_squared_error(y_caudal_real, pred_caudal))

print("\n===== RESULTADOS GLOBALES =====")
print(f"NIVEL -> MAE: {mae_nivel:.4f} | RMSE: {rmse_nivel:.4f}")
print(f"CAUDAL -> MAE: {mae_caudal:.4f} | RMSE: {rmse_caudal:.4f}")

# =========================
# POR HORIZONTE
# =========================
print("\n===== POR HORIZONTE =====")

for dia in range(HORIZONTE):
    mae_n = mean_absolute_error(
        y_nivel_test[:, dia], pred_nivel[:, dia]
    )

    mae_c = mean_absolute_error(
        y_caudal_real[:, dia], pred_caudal[:, dia]
    )

    print(f"Día {dia+1}: Nivel MAE={mae_n:.4f} | Caudal MAE={mae_c:.4f}")

# =========================
# POR MUNICIPIO
# =========================
print("\n===== POR MUNICIPIO =====")

for i, col in enumerate(municipios_cols):
    mask = municipios_test[:, i] == 1

    if np.sum(mask) < 50:
        continue

    mae_n = mean_absolute_error(
        y_nivel_test[mask], pred_nivel[mask]
    )

    mae_c = mean_absolute_error(
        y_caudal_real[mask], pred_caudal[mask]
    )

    print(f"{col.replace('municipio_', '')}:")
    print(f"   Nivel MAE={mae_n:.4f}")
    print(f"   Caudal MAE={mae_c:.4f}")
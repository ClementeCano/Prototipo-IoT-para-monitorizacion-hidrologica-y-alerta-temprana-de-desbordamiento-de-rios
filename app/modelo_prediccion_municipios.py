from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input

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
    municipio = path_csv.stem.lower()  # 🔥 importante: minúsculas
    print(f"\n📍 {municipio}")

    df = pd.read_csv(path_csv)

    if len(df) < 50:
        print("⚠️ Muy pocos datos, saltando")
        return

    # =========================
    # FEATURES
    # =========================
    excluir = {"fecha", "nivel_m", "caudal_m3s", "desbordamiento"}
    features = [c for c in df.columns if c not in excluir]

    X_data = df[features].values
    nivel = df["nivel_m"].values
    caudal = df["caudal_log"].values

    p95_nivel = np.percentile(nivel, 95)
    nivel = np.clip(nivel, None, p95_nivel)
    p95_caudal = np.percentile(caudal, 95)
    caudal = np.clip(caudal, None, p95_caudal)

    # =========================
    # ESCALADO
    # =========================
    scaler_X = MinMaxScaler()
    scaler_nivel = MinMaxScaler()
    scaler_caudal = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X_data)
    nivel_scaled = scaler_nivel.fit_transform(nivel.reshape(-1, 1))
    caudal_scaled = scaler_caudal.fit_transform(caudal.reshape(-1, 1))

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
    # MODELO MULTI-OUTPUT 🔥
    # =========================
    inputs = Input(shape=(VENTANA, X.shape[2]))

    x = LSTM(64, return_sequences=False)(inputs)
    x = Dense(32, activation="relu")(x)

    output_nivel = Dense(HORIZONTE, name="nivel")(x)
    output_caudal = Dense(HORIZONTE, name="caudal")(x)

    model = Model(inputs=inputs, outputs=[output_nivel, output_caudal])

    model.compile(
        optimizer="adam",
        loss={
            "nivel": "mse",
            "caudal": "mse"
        }
    )

    # =========================
    # TRAIN
    # =========================
    model.fit(
        X,
        {
            "nivel": y_nivel,
            "caudal": y_caudal
        },
        epochs=20,
        batch_size=16,
        verbose=1
    )

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
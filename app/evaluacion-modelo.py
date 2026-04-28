from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "datasets_modelo_municipios"
MODEL_DIR = BASE_DIR / "modelos_municipios"
GRAFICAS_DIR = BASE_DIR / "graficas_municipios"
GRAFICAS_DIR.mkdir(exist_ok=True)

VENTANA = 14
HORIZONTE = 7
UMBRAL = 3.0  # umbral desbordamiento

# =========================
# VENTANAS
# =========================
def crear_ventanas(data, nivel, caudal, ventana, horizonte):
    X, y_nivel, y_caudal = [], [], []

    for i in range(len(data) - ventana - horizonte + 1):
        X.append(data[i:i+ventana])
        y_nivel.append(nivel[i+ventana:i+ventana+horizonte])
        y_caudal.append(caudal[i+ventana:i+ventana+horizonte])

    return np.array(X), np.array(y_nivel), np.array(y_caudal)

# =========================
# EVALUAR MUNICIPIO
# =========================
def evaluar_municipio(path_csv):
    municipio = path_csv.stem
    print(f"\n📍 {municipio}")

    df = pd.read_csv(path_csv)

    if len(df) < 50:
        print("⚠️ Muy pocos datos")
        return

    # =========================
    # FEATURES
    # =========================
    excluir = {"fecha", "nivel_m", "caudal_m3s", "desbordamiento"}
    features = [c for c in df.columns if c not in excluir]

    X_data = df[features].values
    nivel = df["nivel_m"].values
    caudal = df["caudal_log"].values

    # =========================
    # LOAD MODELO
    # =========================
    modelo_path = MODEL_DIR / municipio / "modelo.keras"
    scaler_X_path = MODEL_DIR / municipio / "scaler_X.pkl"
    scaler_nivel_path = MODEL_DIR / municipio / "scaler_nivel.pkl"
    scaler_caudal_path = MODEL_DIR / municipio / "scaler_caudal.pkl"

    if not modelo_path.exists():
        print("⚠️ Modelo no encontrado")
        return

    modelo = load_model(modelo_path, compile=False)

    scaler_X = pickle.load(open(scaler_X_path, "rb"))
    scaler_nivel = pickle.load(open(scaler_nivel_path, "rb"))
    scaler_caudal = pickle.load(open(scaler_caudal_path, "rb"))

    # =========================
    # ESCALADO
    # =========================
    X_scaled = scaler_X.transform(X_data)
    nivel_scaled = scaler_nivel.transform(nivel.reshape(-1,1))
    caudal_scaled = scaler_caudal.transform(caudal.reshape(-1,1))

    # =========================
    # VENTANAS
    # =========================
    X, y_nivel, y_caudal = crear_ventanas(
        X_scaled, nivel_scaled, caudal_scaled, VENTANA, HORIZONTE
    )

    if len(X) < 10:
        print("⚠️ Muy pocas ventanas")
        return

    # =========================
    # SPLIT
    # =========================
    split = int(len(X) * 0.85)

    X_test = X[split:]
    y_nivel_test = y_nivel[split:]
    y_caudal_test = y_caudal[split:]

    # =========================
    # PREDICCIÓN (CORRECTA)
    # =========================
    pred = modelo.predict(X_test, verbose=0)

    if not isinstance(pred, list) or len(pred) != 2:
        print("❌ El modelo no devuelve 2 salidas")
        return

    pred_nivel_scaled, pred_caudal_scaled = pred

    # =========================
    # DESESCALADO
    # =========================
    pred_nivel = scaler_nivel.inverse_transform(pred_nivel_scaled)

    pred_caudal_log = scaler_caudal.inverse_transform(pred_caudal_scaled)
    pred_caudal = np.expm1(pred_caudal_log)

    y_caudal_real = np.expm1(y_caudal_test)

    # =========================
    # RESHAPE PARA MÉTRICAS
    # =========================
    y_nivel_test = y_nivel_test.reshape(y_nivel_test.shape[0], -1)
    pred_nivel = pred_nivel.reshape(pred_nivel.shape[0], -1)

    y_caudal_real = y_caudal_real.reshape(y_caudal_real.shape[0], -1)
    pred_caudal = pred_caudal.reshape(pred_caudal.shape[0], -1)

    # =========================
    # MÉTRICAS
    # =========================
    mae_n = mean_absolute_error(y_nivel_test, pred_nivel)
    rmse_n = np.sqrt(mean_squared_error(y_nivel_test, pred_nivel))

    mae_c = mean_absolute_error(y_caudal_real, pred_caudal)
    rmse_c = np.sqrt(mean_squared_error(y_caudal_real, pred_caudal))

    print(f"NIVEL -> MAE: {mae_n:.4f} | RMSE: {rmse_n:.4f}")
    print(f"CAUDAL -> MAE: {mae_c:.4f} | RMSE: {rmse_c:.4f}")

    # =========================
    # GRÁFICAS
    # =========================
    out_dir = GRAFICAS_DIR / municipio
    out_dir.mkdir(exist_ok=True)

    # Día +1
    y_real_nivel = y_nivel_test[:,0]
    y_pred_nivel = pred_nivel[:,0]

    y_real_caudal = y_caudal_real[:,0]
    y_pred_caudal = pred_caudal[:,0]

    # 🔹 NIVEL
    plt.figure()
    plt.plot(y_real_nivel, label="Real")
    plt.plot(y_pred_nivel, label="Pred")
    plt.legend()
    plt.title("Nivel (Día +1)")
    plt.savefig(out_dir / "nivel_linea.png")
    plt.close()

    # 🔹 CAUDAL
    plt.figure()
    plt.plot(y_real_caudal, label="Real")
    plt.plot(y_pred_caudal, label="Pred")
    plt.legend()
    plt.title("Caudal (Día +1)")
    plt.savefig(out_dir / "caudal_linea.png")
    plt.close()

    # 🔹 SCATTER NIVEL
    plt.figure()
    plt.scatter(y_real_nivel, y_pred_nivel, alpha=0.5)
    plt.plot([y_real_nivel.min(), y_real_nivel.max()],
             [y_real_nivel.min(), y_real_nivel.max()])
    plt.title("Scatter Nivel")
    plt.savefig(out_dir / "nivel_scatter.png")
    plt.close()

    # 🔹 SCATTER CAUDAL
    plt.figure()
    plt.scatter(y_real_caudal, y_pred_caudal, alpha=0.5)
    plt.plot([y_real_caudal.min(), y_real_caudal.max()],
             [y_real_caudal.min(), y_real_caudal.max()])
    plt.title("Scatter Caudal")
    plt.savefig(out_dir / "caudal_scatter.png")
    plt.close()

    # 🔹 PICOS
    mask = y_real_nivel > UMBRAL
    if np.sum(mask) > 0:
        plt.figure()
        plt.plot(y_real_nivel[mask], label="Real")
        plt.plot(y_pred_nivel[mask], label="Pred")
        plt.legend()
        plt.title("Picos (Nivel)")
        plt.savefig(out_dir / "picos.png")
        plt.close()

# =========================
# MAIN
# =========================
def main():
    archivos = list(DATA_DIR.glob("*.csv"))

    print("📊 Evaluando modelos...")

    for archivo in archivos:
        print(f"\nEvaluando {archivo.stem}...")
        evaluar_municipio(archivo)

if __name__ == "__main__":
    main()
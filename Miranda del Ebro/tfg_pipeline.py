import json
from pathlib import Path
import numpy as np
import pandas as pd

IN_CSV = Path("Miranda_del_Ebro.csv")

OUT_DATA = Path("TFG_Miranda_features_labels.csv")
OUT_MODEL = Path("TFG_model_logreg.json")
OUT_REPORT = Path("TFG_report.txt")

# ---------- utilidades ----------
def future_risk(overflow: np.ndarray, H: int) -> np.ndarray:
    """
    y[i]=1 si hay desbordamiento en (i, i+H] (futuro), excluyendo el día i.
    """
    n = len(overflow)
    c = np.zeros(n + 1, dtype=int)
    c[1:] = np.cumsum(overflow)
    idx = np.arange(n)
    start = idx + 1
    end = np.minimum(n, idx + H + 1)
    return ((c[end] - c[start]) > 0).astype(int)

def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-z))

def average_precision(y_true, y_score):
    # AP (PR-AUC) sin sklearn
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / np.maximum(tp + fp, 1)
    # AP = suma de precisiones en cada positivo / num_positivos
    pos = tp[-1]
    if pos == 0:
        return 0.0
    return float((precision * y_true).sum() / pos)

def metrics_at_threshold(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
    return {
        "thr": float(thr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "alerts": int(y_pred.sum()),
        "cm": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    }

# ---------- carga ----------
df = pd.read_csv(IN_CSV)
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)

overflow = df["overflow"].astype(int).to_numpy()
df["y_riesgo_1d"] = future_risk(overflow, 1)   # riesgo en 24h (día siguiente)
df["y_riesgo_7d"] = future_risk(overflow, 7)   # riesgo en 7 días

# ---------- features (diarias, compatibles con “tiempo real” agregando lecturas) ----------
feat = df[["fecha","nivel_m","caudal_m3s","lluvia_mm","overflow","y_riesgo_1d","y_riesgo_7d"]].copy()

# lags
for col in ["nivel_m","caudal_m3s","lluvia_mm"]:
    for L in [1,2,7]:
        feat[f"{col}_lag{L}"] = feat[col].shift(L)

# tendencias
feat["dnivel_1"] = feat["nivel_m"].diff(1)
feat["dcaudal_1"] = feat["caudal_m3s"].diff(1)

# ventanas (rolling)
for w in [3,7,14]:
    for col in ["nivel_m","caudal_m3s"]:
        r = feat[col].rolling(w)
        feat[f"{col}_mean{w}"] = r.mean()
        feat[f"{col}_max{w}"]  = r.max()
    r = feat["lluvia_mm"].rolling(w)
    feat[f"lluvia_mm_sum{w}"] = r.sum()
    feat[f"lluvia_mm_max{w}"] = r.max()

# estacionalidad
feat["month"] = feat["fecha"].dt.month
feat["dayofyear"] = feat["fecha"].dt.dayofyear
feat["weekday"] = feat["fecha"].dt.weekday

feat = feat.dropna().reset_index(drop=True)

# guarda dataset
feat.to_csv(OUT_DATA, index=False)

# ---------- split temporal ----------
split_date = pd.Timestamp("2024-01-01")
train = feat[feat["fecha"] < split_date].copy()
test  = feat[feat["fecha"] >= split_date].copy()

X_cols = [c for c in feat.columns if c not in ["fecha","overflow","y_riesgo_1d","y_riesgo_7d"]]
y_col = "y_riesgo_7d"  # para alerta temprana (más positivos que 1d)

X_train = train[X_cols].to_numpy(dtype=float)
y_train = train[y_col].to_numpy(dtype=int)
X_test  = test[X_cols].to_numpy(dtype=float)
y_test  = test[y_col].to_numpy(dtype=int)

# ---------- modelo: Logistic Regression por numpy (rápido y exportable) ----------
# estandariza
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)
sd[sd == 0] = 1.0
Xtr = (X_train - mu) / sd
Xte = (X_test  - mu) / sd

# añade bias
Xtr_b = np.c_[np.ones(len(Xtr)), Xtr]
Xte_b = np.c_[np.ones(len(Xte)), Xte]

# pesos por desbalanceo
pos = y_train.sum()
neg = len(y_train) - pos
pos_w = (neg / max(pos,1))
w = np.where(y_train == 1, pos_w, 1.0)

# grad descent
rng = np.random.default_rng(42)
w0 = rng.normal(0, 0.01, size=Xtr_b.shape[1])
lr = 0.05
l2 = 0.001
for _ in range(2500):
    p = sigmoid(Xtr_b @ w0)
    grad = (Xtr_b.T @ ((p - y_train) * w)) / len(y_train)
    grad[1:] += l2 * w0[1:]  # no regularizar el bias
    w0 -= lr * grad

proba_test = sigmoid(Xte_b @ w0)

# ---------- baseline por reglas (para y_riesgo_1d) ----------
# Regla: alerta mañana si (nivel>=3.0) OR (lluvia_3d>=15 & nivel>=1.4 & subida>=0.05)
# (usa variables en el día t para anticipar t+1)
baseline = (
    (feat["nivel_m"] >= 3.0) |
    ((feat["lluvia_mm_sum3"] >= 15.0) & (feat["nivel_m"] >= 1.4) & (feat["dnivel_1"] >= 0.05))
).astype(int).to_numpy()

# métricas baseline contra y_riesgo_1d en el mismo tramo test
feat_test = feat[feat["fecha"] >= split_date].copy()
y1_test = feat_test["y_riesgo_1d"].to_numpy(dtype=int)
baseline_test = baseline[feat["fecha"] >= split_date]

def basic_counts(y_true, y_pred):
    tp = int(((y_true==1)&(y_pred==1)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    return tp, fp, fn, float(prec), float(rec), int(y_pred.sum()), int(y_true.sum())

b_tp,b_fp,b_fn,b_prec,b_rec,b_alerts,b_pos = basic_counts(y1_test, baseline_test)

# ---------- reporte ----------
ap = average_precision(y_test, proba_test)
lines = []
lines.append("TFG - Predicción temprana de desbordamiento (Miranda del Ebro)")
lines.append("")
lines.append(f"Dataset: {IN_CSV.name}")
lines.append(f"Filas (post-features): {len(feat)} | Features: {len(X_cols)}")
lines.append(f"Split temporal: train < {split_date.date()} | test >= {split_date.date()}")
lines.append("")
lines.append(f"Objetivo principal (ML): y_riesgo_7d  | Positivos train={int(y_train.sum())} test={int(y_test.sum())}")
lines.append(f"PR-AUC (Average Precision) en test: {ap:.4f}")
lines.append("Métricas por umbral (test):")
for thr in [0.5,0.3,0.2,0.1]:
    m = metrics_at_threshold(y_test, proba_test, thr)
    lines.append(f"  thr={thr:.1f}  prec={m['precision']:.3f}  rec={m['recall']:.3f}  f1={m['f1']:.3f}  alerts={m['alerts']}  cm={m['cm']}")
lines.append("")
lines.append("Baseline (reglas) para y_riesgo_1d (alerta a 24h) en test:")
lines.append(f"  prec={b_prec:.3f}  rec={b_rec:.3f}  alerts={b_alerts}  positivos={b_pos}  (tp={b_tp}, fp={b_fp}, fn={b_fn})")
lines.append("")
lines.append("Sugerencia práctica:")
lines.append("- Empieza con thr=0.2 o 0.3 y añade histéresis: alertar si p>thr en 2 ejecuciones seguidas.")
lines.append("- Luego ajusta thr buscando el equilibrio entre 'no perder crecidas' (recall alto) y falsas alarmas.")
OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

# ---------- exporta modelo a JSON ----------
model_json = {
    "feature_names": X_cols,
    "scaler_mean": mu.tolist(),
    "scaler_std": sd.tolist(),
    "coef_with_bias": w0.tolist(),  # incluye bias en coef_with_bias[0]
    "target": y_col,
    "split_date": str(split_date.date()),
}
OUT_MODEL.write_text(json.dumps(model_json, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"OK -> {OUT_DATA} | {OUT_MODEL} | {OUT_REPORT}")
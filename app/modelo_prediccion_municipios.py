from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

try:
    from app.model_pipeline import (
        add_model_features,
        inverse_caudal_log,
        inverse_scaled_column,
        model_feature_columns,
        utc_now_iso,
    )
except ImportError:
    from model_pipeline import (
        add_model_features,
        inverse_caudal_log,
        inverse_scaled_column,
        model_feature_columns,
        utc_now_iso,
    )


os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets_modelo_municipios"
MODEL_DIR = BASE_DIR / "modelos_municipios"
MODEL_DIR.mkdir(exist_ok=True)

VENTANA = 14
HORIZONTE = 7
MODEL_VERSION = "lstm_multihorizon_v2"


def _import_tensorflow():
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    return {
        "tf": tf,
        "Adam": Adam,
        "Dense": Dense,
        "Dropout": Dropout,
        "EarlyStopping": EarlyStopping,
        "Huber": Huber,
        "Input": Input,
        "LSTM": LSTM,
        "Model": Model,
        "ReduceLROnPlateau": ReduceLROnPlateau,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        libs = _import_tensorflow()
        libs["tf"].random.set_seed(seed)
    except Exception:
        pass


def crear_ventanas(
    x_data: np.ndarray,
    nivel_scaled: np.ndarray,
    caudal_scaled: np.ndarray,
    nivel_real: np.ndarray,
    caudal_real: np.ndarray,
    ventana: int,
    horizonte: int,
):
    x_windows = []
    y_nivel = []
    y_caudal = []
    y_nivel_real = []
    y_caudal_real = []
    target_start = []
    last_nivel = []
    last_caudal = []

    for i in range(len(x_data) - ventana - horizonte + 1):
        start = i + ventana
        end = start + horizonte
        x_windows.append(x_data[i : i + ventana])
        y_nivel.append(nivel_scaled[start:end])
        y_caudal.append(caudal_scaled[start:end])
        y_nivel_real.append(nivel_real[start:end])
        y_caudal_real.append(caudal_real[start:end])
        target_start.append(start)
        last_nivel.append(nivel_real[start - 1])
        last_caudal.append(caudal_real[start - 1])

    return {
        "X": np.asarray(x_windows),
        "y_nivel": np.asarray(y_nivel),
        "y_caudal": np.asarray(y_caudal),
        "y_nivel_real": np.asarray(y_nivel_real),
        "y_caudal_real": np.asarray(y_caudal_real),
        "target_start": np.asarray(target_start),
        "last_nivel": np.asarray(last_nivel),
        "last_caudal": np.asarray(last_caudal),
    }


def _split_masks(target_start: np.ndarray, n_rows: int, train_ratio: float, val_ratio: float):
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    train_mask = target_start < train_end
    val_mask = (target_start >= train_end) & (target_start < val_end)
    test_mask = target_start >= val_end
    return train_end, val_end, train_mask, val_mask, test_mask


def _build_model(input_shape, horizonte: int, libs: dict):
    inputs = libs["Input"](shape=input_shape)
    x = libs["LSTM"](96, return_sequences=True, dropout=0.15)(inputs)
    x = libs["LSTM"](48, dropout=0.15)(x)
    x = libs["Dense"](96, activation="relu")(x)
    x = libs["Dropout"](0.15)(x)
    x = libs["Dense"](48, activation="relu")(x)

    output_nivel = libs["Dense"](horizonte, name="nivel")(x)
    output_caudal = libs["Dense"](horizonte, name="caudal")(x)
    model = libs["Model"](inputs=inputs, outputs=[output_nivel, output_caudal])
    model.compile(
        optimizer=libs["Adam"](learning_rate=0.0004),
        loss={"nivel": libs["Huber"](delta=0.5), "caudal": libs["Huber"](delta=0.5)},
        metrics={"nivel": ["mae"], "caudal": ["mae"]},
    )
    return model


def _as_float(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _metric_block(y_true_nivel, y_pred_nivel, y_true_caudal, y_pred_caudal):
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    if len(y_true_nivel) == 0:
        return None

    overall = {
        "mae_nivel_m": _as_float(mean_absolute_error(y_true_nivel.ravel(), y_pred_nivel.ravel())),
        "rmse_nivel_m": _as_float(np.sqrt(mean_squared_error(y_true_nivel.ravel(), y_pred_nivel.ravel()))),
        "mae_caudal_m3s": _as_float(mean_absolute_error(y_true_caudal.ravel(), y_pred_caudal.ravel())),
        "rmse_caudal_m3s": _as_float(np.sqrt(mean_squared_error(y_true_caudal.ravel(), y_pred_caudal.ravel()))),
    }
    by_horizon = []
    for day in range(y_true_nivel.shape[1]):
        by_horizon.append(
            {
                "day": day + 1,
                "mae_nivel_m": _as_float(mean_absolute_error(y_true_nivel[:, day], y_pred_nivel[:, day])),
                "rmse_nivel_m": _as_float(np.sqrt(mean_squared_error(y_true_nivel[:, day], y_pred_nivel[:, day]))),
                "mae_caudal_m3s": _as_float(mean_absolute_error(y_true_caudal[:, day], y_pred_caudal[:, day])),
                "rmse_caudal_m3s": _as_float(np.sqrt(mean_squared_error(y_true_caudal[:, day], y_pred_caudal[:, day]))),
            }
        )
    return {"overall": overall, "by_horizon": by_horizon}


def _predict_real(model, scaler_nivel, scaler_caudal, x_data: np.ndarray):
    pred_nivel_scaled, pred_caudal_scaled = model.predict(x_data, verbose=0)
    pred_nivel = inverse_scaled_column(scaler_nivel, pred_nivel_scaled)
    pred_caudal = inverse_caudal_log(scaler_caudal, pred_caudal_scaled)
    return pred_nivel, pred_caudal


def _baseline_persistence(last_nivel: np.ndarray, last_caudal: np.ndarray, horizonte: int):
    return (
        np.repeat(last_nivel.reshape(-1, 1), horizonte, axis=1),
        np.repeat(last_caudal.reshape(-1, 1), horizonte, axis=1),
    )


def _training_history(history) -> list[dict]:
    rows = []
    keys = sorted(history.history)
    for index in range(len(history.history.get("loss", []))):
        row = {"epoch": index + 1}
        for key in keys:
            row[key] = _as_float(history.history[key][index])
        rows.append(row)
    return rows


def _json_dump(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def entrenar_municipio(path_csv: Path, args) -> dict | None:
    try:
        from sklearn.preprocessing import RobustScaler
    except ImportError as exc:
        raise RuntimeError(
            "Falta scikit-learn en el entorno. Instala las dependencias con "
            "pip install -r requirements.txt antes de entrenar."
        ) from exc

    municipio = path_csv.stem.lower()
    print(f"\nMunicipio: {municipio}")

    try:
        raw_df = pd.read_csv(path_csv)
    except Exception as exc:
        print(f"  No se pudo leer el CSV: {exc}")
        return None

    df = add_model_features(raw_df)
    if len(df) < args.min_rows:
        print(f"  Datos insuficientes ({len(df)} filas limpias).")
        return None

    features = model_feature_columns(df)
    if not features:
        print("  No hay variables de entrada.")
        return None

    train_end = int(len(df) * args.train_ratio)
    if train_end <= VENTANA + HORIZONTE:
        print("  Split de entrenamiento demasiado corto.")
        return None

    scaler_X = RobustScaler()
    scaler_nivel = RobustScaler()
    scaler_caudal = RobustScaler()

    train_df = df.iloc[:train_end]
    scaler_X.fit(train_df[features].values)
    scaler_nivel.fit(train_df[["nivel_m"]].values)
    scaler_caudal.fit(train_df[["caudal_log"]].values)

    x_scaled = scaler_X.transform(df[features].values)
    nivel_scaled = scaler_nivel.transform(df[["nivel_m"]].values).ravel()
    caudal_scaled = scaler_caudal.transform(df[["caudal_log"]].values).ravel()

    windows = crear_ventanas(
        x_scaled,
        nivel_scaled,
        caudal_scaled,
        df["nivel_m"].values,
        df["caudal_m3s"].values,
        VENTANA,
        HORIZONTE,
    )
    if len(windows["X"]) < args.min_windows:
        print(f"  Ventanas insuficientes ({len(windows['X'])}).")
        return None

    train_end, val_end, train_mask, val_mask, test_mask = _split_masks(
        windows["target_start"], len(df), args.train_ratio, args.val_ratio
    )
    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        print(
            "  Split temporal invalido: "
            f"train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}"
        )
        return None

    try:
        libs = _import_tensorflow()
    except ImportError as exc:
        raise RuntimeError(
            "Falta TensorFlow en el entorno. Instala las dependencias con "
            "pip install -r requirements.txt antes de entrenar."
        ) from exc
    model = _build_model((VENTANA, windows["X"].shape[2]), HORIZONTE, libs)

    callbacks = [
        libs["EarlyStopping"](monitor="val_loss", patience=args.patience, restore_best_weights=True),
        libs["ReduceLROnPlateau"](monitor="val_loss", factor=0.5, patience=max(3, args.patience // 2), min_lr=1e-5),
    ]

    history = model.fit(
        windows["X"][train_mask],
        {"nivel": windows["y_nivel"][train_mask], "caudal": windows["y_caudal"][train_mask]},
        validation_data=(
            windows["X"][val_mask],
            {"nivel": windows["y_nivel"][val_mask], "caudal": windows["y_caudal"][val_mask]},
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=args.verbose,
    )

    pred_val_nivel, pred_val_caudal = _predict_real(model, scaler_nivel, scaler_caudal, windows["X"][val_mask])
    pred_test_nivel, pred_test_caudal = _predict_real(model, scaler_nivel, scaler_caudal, windows["X"][test_mask])
    baseline_val_nivel, baseline_val_caudal = _baseline_persistence(
        windows["last_nivel"][val_mask], windows["last_caudal"][val_mask], HORIZONTE
    )
    baseline_test_nivel, baseline_test_caudal = _baseline_persistence(
        windows["last_nivel"][test_mask], windows["last_caudal"][test_mask], HORIZONTE
    )

    validation_metrics = _metric_block(
        windows["y_nivel_real"][val_mask],
        pred_val_nivel,
        windows["y_caudal_real"][val_mask],
        pred_val_caudal,
    )
    test_metrics = _metric_block(
        windows["y_nivel_real"][test_mask],
        pred_test_nivel,
        windows["y_caudal_real"][test_mask],
        pred_test_caudal,
    )
    baseline_validation = _metric_block(
        windows["y_nivel_real"][val_mask],
        baseline_val_nivel,
        windows["y_caudal_real"][val_mask],
        baseline_val_caudal,
    )
    baseline_test = _metric_block(
        windows["y_nivel_real"][test_mask],
        baseline_test_nivel,
        windows["y_caudal_real"][test_mask],
        baseline_test_caudal,
    )

    out_dir = MODEL_DIR / municipio
    out_dir.mkdir(exist_ok=True)
    model.save(out_dir / "modelo.keras")
    with open(out_dir / "scaler_X.pkl", "wb") as file:
        pickle.dump(scaler_X, file)
    with open(out_dir / "scaler_nivel.pkl", "wb") as file:
        pickle.dump(scaler_nivel, file)
    with open(out_dir / "scaler_caudal.pkl", "wb") as file:
        pickle.dump(scaler_caudal, file)
    with open(out_dir / "features.pkl", "wb") as file:
        pickle.dump(features, file)

    metrics = {
        "site_id": municipio,
        "model_version": MODEL_VERSION,
        "trained_at": utc_now_iso(),
        "dataset": {
            "rows_raw": int(len(raw_df)),
            "rows_clean": int(len(df)),
            "first_date": str(df["fecha"].iloc[0]),
            "last_date": str(df["fecha"].iloc[-1]),
        },
        "window": {"input_days": VENTANA, "horizon_days": HORIZONTE},
        "features": features,
        "splits": {
            "train_rows_until": str(df["fecha"].iloc[train_end - 1]),
            "validation_rows_until": str(df["fecha"].iloc[val_end - 1]),
            "train_windows": int(train_mask.sum()),
            "validation_windows": int(val_mask.sum()),
            "test_windows": int(test_mask.sum()),
        },
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
            "baseline_persistence_validation": baseline_validation,
            "baseline_persistence_test": baseline_test,
        },
        "training": {
            "epochs_requested": int(args.epochs),
            "epochs_trained": int(len(history.history.get("loss", []))),
            "batch_size": int(args.batch_size),
            "loss": "Huber",
            "scaler": "RobustScaler",
            "target_transform": {"caudal": "log1p", "nivel": "identity"},
        },
    }
    _json_dump(out_dir / "metrics.json", metrics)
    _json_dump(out_dir / "training_history.json", _training_history(history))

    test_overall = test_metrics["overall"]
    print(
        "  Test MAE nivel={:.3f} m | RMSE nivel={:.3f} m | "
        "MAE caudal={:.3f} m3/s | RMSE caudal={:.3f} m3/s".format(
            test_overall["mae_nivel_m"],
            test_overall["rmse_nivel_m"],
            test_overall["mae_caudal_m3s"],
            test_overall["rmse_caudal_m3s"],
        )
    )
    print(f"  Modelo guardado en {out_dir}")
    return metrics


def _select_files(sites: list[str] | None) -> list[Path]:
    files = sorted(DATA_DIR.glob("*.csv"))
    if not sites:
        return files
    wanted = {site.lower() for site in sites}
    return [path for path in files if path.stem.lower() in wanted]


def main():
    parser = argparse.ArgumentParser(description="Entrena modelos LSTM por municipio con split temporal limpio.")
    parser.add_argument("--site", action="append", help="ID de municipio. Repetible. Por defecto: todos.")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--min-rows", type=int, default=180)
    parser.add_argument("--min-windows", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 0.95:
        raise SystemExit("Usa ratios validos: train > 0, val > 0 y train + val < 0.95.")

    _set_seed(args.seed)
    files = _select_files(args.site)
    if not files:
        raise SystemExit("No se encontraron datasets para entrenar.")

    print("Entrenando modelos predictivos por municipio...")
    summary = []
    for path in files:
        try:
            metrics = entrenar_municipio(path, args)
            if metrics:
                summary.append(
                    {
                        "site_id": metrics["site_id"],
                        "trained_at": metrics["trained_at"],
                        "dataset_last_date": metrics["dataset"]["last_date"],
                        "test": metrics["metrics"]["test"]["overall"],
                    }
                )
        except Exception as exc:
            print(f"  Error entrenando {path.stem}: {exc}")

    if summary:
        _json_dump(MODEL_DIR / "training_summary.json", summary)
        print(f"\nResumen guardado en {MODEL_DIR / 'training_summary.json'}")
    else:
        print("\nNo se entreno ningun modelo.")


if __name__ == "__main__":
    main()

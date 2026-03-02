# app.py
from pathlib import Path
from datetime import datetime
import asyncio
import json

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, requests
from fastapi.responses import HTMLResponse

from saih_opendata import fetch_saih_latest


POLL_SECONDS = 20

app = FastAPI()

# ---------------------------
# Modelo (cargado, listo)
# ---------------------------
MODEL = json.loads(Path("TFG_model_logreg.json").read_text(encoding="utf-8"))
FEATS = MODEL["feature_names"]
MU = np.array(MODEL["scaler_mean"], dtype=float)
SD = np.array(MODEL["scaler_std"], dtype=float)
COEF = np.array(MODEL["coef_with_bias"], dtype=float)

# ---------------------------
# Estado + conexiones WS
# ---------------------------
clients: set[WebSocket] = set()
last_payload = None


def sigmoid(z: float) -> float:
    z = np.clip(z, -50, 50)
    return float(1 / (1 + np.exp(-z)))


def predict_prob(feature_dict: dict) -> float:
    """
    Devuelve probabilidad del modelo logístico exportado (si feature_dict contiene todas FEATS).
    Nota: en 'tiempo real' todavía no lo usamos hasta construir features rolling consistentes.
    """
    x = np.array([feature_dict[f] for f in FEATS], dtype=float)
    xz = (x - MU) / np.where(SD == 0, 1.0, SD)
    z = float(COEF[0] + np.dot(xz, COEF[1:]))
    return sigmoid(z)


async def broadcast(payload: dict) -> None:
    """
    Envía el payload a todos los clientes conectados al WebSocket.
    Limpia conexiones caídas.
    """
    dead = []
    msg = json.dumps(payload, ensure_ascii=False)

    for c in list(clients):
        try:
            await c.send_text(msg)
        except Exception:
            dead.append(c)

    for d in dead:
        clients.discard(d)


# ---------------------------
# Rutas
# ---------------------------
@app.get("/")
def home():
    return HTMLResponse(Path("index.html").read_text(encoding="utf-8"))


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)

    try:
        # manda último estado al conectar
        if last_payload is not None:
            await websocket.send_text(json.dumps(last_payload, ensure_ascii=False))

        # mantenemos viva la conexión (aunque el cliente no envíe nada)
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        clients.discard(websocket)
    except Exception:
        clients.discard(websocket)


@app.get("/ingest_demo")
async def ingest_demo():
    """
    Simula que entra un dato.
    Útil para probar la UI sin depender de SAIH.
    """
    global last_payload

    now = datetime.now().isoformat(timespec="seconds")

    fake_features = {f: 0.0 for f in FEATS}
    prob = predict_prob(fake_features)

    last_payload = {
        "ts": now,
        "nivel_m": None,
        "caudal_m3s": None,
        "tendencia_nivel": None,
        "tendencia_caudal": None,
        "lluvia_mm": None,
        "prob_riesgo_7d": prob,
        "source": "demo"
    }

    await broadcast(last_payload)
    return last_payload


# ---------------------------
# Polling SAIH OpenData
# ---------------------------
last_ts_data = None  # último ts del dato (SAIH)

async def poll_loop():
    global last_payload, last_ts_data

    while True:
        try:
            latest = fetch_saih_latest()
            ts = latest.get("ts")

            is_new = (ts != last_ts_data)
            if is_new:
                last_ts_data = ts

            last_payload = {
                "ts": ts,  # timestamp del dato SAIH
                "refreshed_at": datetime.now().isoformat(timespec="seconds"),  # cuándo refrescó tu backend
                "is_new": is_new,
                "nivel_m": latest["nivel_m"],
                "caudal_m3s": latest["caudal_m3s"],
                "tendencia_nivel": latest["tendencia_nivel"],
                "tendencia_caudal": latest["tendencia_caudal"],
                "lluvia_mm": None,
                "prob_riesgo_7d": None,
                "source": "saih_opendata"
            }

            await broadcast(last_payload)
            print("[OK] tick:", last_payload["refreshed_at"], "| data_ts:", ts, "| new:", is_new)

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                print("[RATE LIMIT] 429 -> espero 60s")
                await asyncio.sleep(60)
                continue
            print("[ERROR]", repr(e))

        await asyncio.sleep(POLL_SECONDS)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(poll_loop())
import base64
import json
import os
from dotenv import load_dotenv

load_dotenv()
import traceback

import firebase_admin
from firebase_admin import credentials, messaging


try:
    firebase_path = os.getenv("FIREBASE_CREDENTIALS")
    firebase_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
    firebase_json_base64 = os.getenv("FIREBASE_CREDENTIALS_JSON_BASE64")

    print("🔥 Firebase path:", firebase_path)

    firebase_cred_source = None

    if firebase_json_base64:
        firebase_cred_source = json.loads(base64.b64decode(firebase_json_base64).decode("utf-8"))
    elif firebase_json:
        firebase_cred_source = json.loads(firebase_json)
    elif firebase_path:
        firebase_cred_source = firebase_path

    if not firebase_cred_source:
        raise RuntimeError(
            "Configura FIREBASE_CREDENTIALS, FIREBASE_CREDENTIALS_JSON o FIREBASE_CREDENTIALS_JSON_BASE64"
        )

    cred = credentials.Certificate(firebase_cred_source)

    firebase_admin.initialize_app(cred)

    print("✅ Firebase inicializado")

except Exception as e:
    print("❌ Error Firebase:", e)




from pathlib import Path
from datetime import datetime
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, Set, Optional

import requests
import pandas as pd
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from app.api.saih_opendata import fetch_saih_signals
from app.api.aemet_opendata import (
    fetch_aemet_municipio_horaria,
    extract_rain_forecast_mm,
    extract_prob_precip_summary,
)

from app.prediccion_individual import predecir_semana_municipio
from app.core.config import SITES, collect_all_tags

# from api.saih_opendata import fetch_saih_signals
# from api.aemet_opendata import (
#     fetch_aemet_municipio_horaria,
#     extract_rain_forecast_mm,
#     extract_prob_precip_summary,
# )

# from prediccion_individual import predecir_semana_municipio
# from core.config import SITES, collect_all_tags

from fastapi.staticfiles import StaticFiles
from app import alertas

from collections import defaultdict
from pathlib import Path

# =========================
# TOKENS PERSISTENTES
# =========================
from pathlib import Path
from collections import defaultdict
import json

# 🔥 usar ruta absoluta estable
BASE_DIR = Path(__file__).resolve().parent

TOKENS_FILE = Path(os.getenv("TOKENS_FILE", BASE_DIR / "tokens.json")).resolve()
TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)

print("📁 TOKENS_FILE:", TOKENS_FILE)


def cargar_tokens():

    try:

        # =========================
        # CREAR ARCHIVO SI NO EXISTE
        # =========================
        if not TOKENS_FILE.exists():

            print("⚠️ tokens.json no existe, creando...")

            with open(TOKENS_FILE, "w", encoding="utf-8") as f:

                json.dump({}, f)

            return defaultdict(set)

        # =========================
        # LEER ARCHIVO
        # =========================
        with open(TOKENS_FILE, "r", encoding="utf-8") as f:

            data = json.load(f)

        print(f"✅ Tokens cargados: {len(data)} municipios")

        return defaultdict(
            set,
            {
                k: set(v)
                for k, v in data.items()
            }
        )

    except Exception as e:

        print("❌ Error cargando tokens:", repr(e))

        return defaultdict(set)


def guardar_tokens():

    try:

        # =========================
        # CONVERTIR SET → LIST
        # =========================
        serializable = {
            k: list(v)
            for k, v in tokens.items()
        }

        tmp_file = TOKENS_FILE.with_suffix(TOKENS_FILE.suffix + ".tmp")

        with open(tmp_file, "w", encoding="utf-8") as f:

            json.dump(
                serializable,
                f,
                indent=2,
                ensure_ascii=False
            )

        os.replace(tmp_file, TOKENS_FILE)

        total = sum(len(v) for v in serializable.values())

        print(f"💾 Tokens guardados ({total} tokens)")

    except Exception as e:

        print("❌ Error guardando tokens:", repr(e))


# =========================
# CARGAR TOKENS AL INICIO
# =========================
tokens = cargar_tokens()


def limpiar_tokens_invalidos(invalid_tokens):
    invalid_tokens = set(t for t in (invalid_tokens or set()) if t)

    if not invalid_tokens:
        return 0

    removed = 0

    for site_tokens in tokens.values():
        before = len(site_tokens)
        site_tokens.difference_update(invalid_tokens)
        removed += before - len(site_tokens)

    if removed:
        print(
            f"[PUSH CLEANUP] Eliminadas {removed} suscripciones invalidas "
            f"({len(invalid_tokens)} tokens unicos)"
        )
        guardar_tokens()

    return removed



# ---------------------------
# Config
# ---------------------------
# SAIH rate limit: 5/min → 20s = 3/min (seguro)
POLL_SECONDS = 20

# AEMET: refresco real cada 30 min, comprobación cada 60s
AEMET_REFRESH_SECONDS = 1800
AEMET_CHECK_SECONDS = 60
AEMET_ERROR_RETRY_SECONDS = int(os.getenv("AEMET_ERROR_RETRY_SECONDS", "300"))
STARTUP_BACKGROUND_DELAY_SECONDS = int(os.getenv("STARTUP_BACKGROUND_DELAY_SECONDS", "30"))
IA_REFRESH_SECONDS = int(os.getenv("IA_REFRESH_SECONDS", "3600"))
IA_WORKERS = int(os.getenv("IA_WORKERS", "1"))
IA_PROCESS_POOL_ENABLED = os.getenv("IA_PROCESS_POOL_ENABLED", "1").lower() in {"1", "true", "yes"}
IA_REFRESH_ON_WS = os.getenv("IA_REFRESH_ON_WS", "1").lower() in {"1", "true", "yes"}
IA_EXECUTOR = ProcessPoolExecutor(max_workers=IA_WORKERS) if IA_PROCESS_POOL_ENABLED else None

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

SITES_BY_ID = {s["id"]: s for s in SITES}

# Cache simple del dataset para IA
_dataset_modelo_cache: Optional[pd.DataFrame] = None


def _filter_alert_sites(site_ids: Optional[str]):
    if not site_ids:
        return SITES

    requested = [
        site_id.strip()
        for site_id in site_ids.split(",")
        if site_id.strip()
    ]

    selected = [
        SITES_BY_ID[site_id]
        for site_id in requested
        if site_id in SITES_BY_ID
    ]

    return selected or SITES


# ---------------------------
# WS state
# ---------------------------
clients: Set[WebSocket] = set()
ws_site: Dict[WebSocket, str] = {}
ws_last_ts: Dict[WebSocket, Optional[str]] = {}

# ---------------------------
# Caches
# ---------------------------
def _default_aemet() -> Dict[str, Any]:
    return {
        "aemet_refreshed_at": None,
        "aemet_error": None,
        "aemet_mm_6h_sum": 0.0,
        "aemet_mm_24h_sum": 0.0,
        "aemet_mm_6h_max": 0.0,
        "aemet_mm_24h_max": 0.0,
        "aemet_mm_next_hours": [],
        "aemet_prob_6h_max": None,
        "aemet_prob_24h_max": None,
    }

# AEMET cache por sitio (con _epoch interno)
aemet_cache_by_site: Dict[str, Dict[str, Any]] = {}

# “inflight”: evita múltiples llamadas AEMET concurrentes para el mismo sitio
aemet_inflight: set[str] = set()

# SAIH cache por sitio (último nivel/caudal/tendencias)
saih_cache_by_site: Dict[str, Dict[str, Any]] = {}

# IA cache por sitio
ia_cache_by_site: Dict[str, Dict[str, Any]] = {}
ia_inflight: set[str] = set()
ia_epoch_by_site: Dict[str, float] = {}

def _default_ia() -> Dict[str, Any]:
    return {
        "ia_refreshed_at": None,
        "ia_error": None,
        "pred_semana": [],
    }

def _init_caches():
    for s in SITES:
        sid = s["id"]
        saih_cache_by_site.setdefault(sid, {
            "ts": None,
            "nivel_m": None,
            "caudal_m3s": None,
            "tendencia_nivel": None,
            "tendencia_caudal": None,
        })
        aemet_cache_by_site.setdefault(sid, {**_default_aemet(), "_epoch": None})
        ia_cache_by_site.setdefault(sid, _default_ia())

_init_caches()


# ---------------------------
# HTTP routes
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent

@app.get("/")
def home():
    return HTMLResponse((BASE_DIR / "index.html").read_text(encoding="utf-8"))

@app.get("/api/sites")
def api_sites():
    return JSONResponse([
        {
            "id": s["id"],
            "name": s["name"],
            "lat": s.get("lat"),
            "lon": s.get("lon"),
        }
        for s in SITES
    ])


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/push-debug")
async def push_debug(data: dict):
    safe_data = {
        "event": data.get("event"),
        "permission": data.get("permission"),
        "secureContext": data.get("secureContext"),
        "serviceWorker": data.get("serviceWorker"),
        "pushManager": data.get("pushManager"),
        "firebaseMessagingSupported": data.get("firebaseMessagingSupported"),
        "selectedSitesCount": data.get("selectedSitesCount"),
        "errorCode": data.get("errorCode"),
        "errorMessage": data.get("errorMessage"),
        "tokenPrefix": data.get("tokenPrefix"),
        "href": data.get("href"),
        "userAgent": data.get("userAgent"),
    }
    print("[PUSH DEBUG]", json.dumps(safe_data, ensure_ascii=False))
    return {"ok": True}



@app.post("/api/token")
async def save_token(data: dict):

    print("📩 Token recibido")

    token = (data.get("token") or "").strip()
    sites = data.get("sites", [])
    user_agent = (data.get("userAgent") or "")[:180]
    client_platform = (data.get("platform") or "")[:80]

    if not token:

        print("❌ Token vacío")

        return JSONResponse({"ok": False, "error": "token_empty"}, status_code=400)

    if not isinstance(sites, list):

        print("âŒ Sites invÃ¡lidos")

        return JSONResponse({"ok": False, "error": "sites_must_be_list"}, status_code=400)

    print(
        f"[PUSH TOKEN] prefix={token[:15]} "
        f"sites={len(sites)} platform={client_platform or '-'} ua={user_agent or '-'}"
    )

    valid_sites = []

    for site in sites:

        if site in SITES_BY_ID and site not in valid_sites:

            valid_sites.append(site)

        else:

            print(f"âš ï¸ Site ignorado: {site!r}")

    # eliminar token previo
    for site_tokens in tokens.values():
        site_tokens.discard(token)

    # guardar nuevo
    for site in valid_sites:

        tokens[site].add(token)

        print(f"🔥 Token guardado en {site}")

    # 🔥 GUARDAR EN DISCO
    guardar_tokens()

    total = sum(len(v) for v in tokens.values())

    return {
        "ok": True,
        "sites": valid_sites,
        "token_saved": bool(valid_sites),
        "total_tokens": total,
    }


@app.post("/api/test-token")
async def test_token(data: dict):
    token = (data.get("token") or "").strip()

    if not token:
        return JSONResponse({"ok": False, "error": "token_empty"}, status_code=400)

    print(f"[PUSH TEST] Enviando prueba solo a {token[:15]}")

    result = await asyncio.to_thread(
        alertas.enviar_notificacion,
        {token},
        "Prueba de alerta Rio Ebro",
        "Esta notificacion va solo a este dispositivo.",
    )

    removed = limpiar_tokens_invalidos(result.get("invalid_tokens"))

    return {
        "ok": True,
        "sent": result.get("sent", 0),
        "invalid_subscriptions_removed": removed,
        "tokenPrefix": token[:15],
    }


@app.get("/test-alerts-now")
async def test_alerts_now(sites: Optional[str] = None):

    alertas.ULTIMO_ENVIO = None
    alert_sites = _filter_alert_sites(sites)

    result = await asyncio.to_thread(
        alertas.enviar_alertas_diarias,
        tokens,
        alert_sites
    )

    removed = limpiar_tokens_invalidos(result.get("invalid_tokens"))

    return {
        "status": "alertas enviadas",
        "sent": result.get("sent", 0),
        "processed_sites": result.get("processed_sites", 0),
        "sites": [site["id"] for site in alert_sites],
        "invalid_subscriptions_removed": removed,
    }


@app.get("/test-alert")
async def test_alert(sites: Optional[str] = None):
    return await test_alerts_now(sites=sites)


@app.get("/firebase-messaging-sw.js")
def sw():
    return Response(
        ((BASE_DIR / "firebase-messaging-sw.js").read_text(encoding="utf-8")),
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(BASE_DIR / "manifest.webmanifest", media_type="application/manifest+json")


# ---------------------------
# Helpers
# ---------------------------
def _aemet_public_cache(site_id: str) -> Dict[str, Any]:
    c = aemet_cache_by_site.get(site_id)
    if not c:
        return _default_aemet()
    return {k: v for k, v in c.items() if not k.startswith("_")}

def _ia_public_cache(site_id: str) -> Dict[str, Any]:
    c = ia_cache_by_site.get(site_id)
    if not c:
        return _default_ia()
    return c


def _build_payload(site_id: str, forced_is_new: Optional[bool] = None) -> Dict[str, Any]:
    site = SITES_BY_ID.get(site_id, {"id": site_id, "name": site_id})

    sc = saih_cache_by_site.get(site_id, {})
    ts = sc.get("ts")

    payload = {
        "site_id": site_id,
        "site_name": site.get("name", site_id),

        "lat": site.get("lat"),
        "lon": site.get("lon"),
        "is_selected": True,

        "ts": ts,
        "refreshed_at": datetime.now().isoformat(timespec="seconds"),
        "is_new": forced_is_new if forced_is_new is not None else False,
        "source": "saih_opendata",

        "nivel_m": sc.get("nivel_m"),
        "caudal_m3s": sc.get("caudal_m3s"),
        "tendencia_nivel": sc.get("tendencia_nivel"),
        "tendencia_caudal": sc.get("tendencia_caudal"),

        **_aemet_public_cache(site_id),
        **_ia_public_cache(site_id),
    }
    return payload

def _chunk(lst: list[str], n: int) -> list[list[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]


async def _run_prediction(site_id: str):
    if IA_EXECUTOR is not None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(IA_EXECUTOR, predecir_semana_municipio, site_id)

    return await asyncio.to_thread(predecir_semana_municipio, site_id)


async def refresh_ia_for_site(site_id: str, force: bool = False) -> bool:
    if site_id in ia_inflight:
        print(f"IA ya en curso para {site_id}")
        return False

    now_epoch = datetime.now().timestamp()
    last_epoch = ia_epoch_by_site.get(site_id)

    if not force and last_epoch is not None and (now_epoch - last_epoch) < IA_REFRESH_SECONDS:
        return False

    ia_inflight.add(site_id)

    print("🚀 IA para:", site_id)

    try:
        pred = await _run_prediction(site_id)

        if pred is None:
            pred = []

        ia_cache_by_site[site_id] = {
            "ia_refreshed_at": datetime.now().isoformat(timespec="seconds"),
            "ia_error": None,
            "pred_semana": pred,
        }
        ia_epoch_by_site[site_id] = now_epoch

        return True

    except Exception as e:
        ia_cache_by_site[site_id] = {
            "ia_refreshed_at": datetime.now().isoformat(timespec="seconds"),
            "ia_error": repr(e),
            "pred_semana": [],
        }

        traceback.print_exc()
        return False

    finally:
        ia_inflight.discard(site_id)

async def refresh_aemet_for_site(site_id: str, force: bool = False) -> bool:
    """
    Refresca AEMET para un site.
    - force=True: refresca aunque no haya vencido TTL (útil al seleccionar)
    Devuelve True si ha actualizado cache (ok o error).
    """
    if site_id in aemet_inflight:
        return False

    site = SITES_BY_ID.get(site_id)
    if not site:
        return False

    muni = (site.get("aemet_muni") or "").strip()
    if not muni:
        return False

    now_epoch = datetime.now().timestamp()
    cur = aemet_cache_by_site.get(site_id, {**_default_aemet(), "_epoch": None})
    last_epoch = cur.get("_epoch")

    if not force:
        ttl = AEMET_ERROR_RETRY_SECONDS if cur.get("aemet_error") else AEMET_REFRESH_SECONDS
        if last_epoch is not None and (now_epoch - float(last_epoch)) < ttl:
            return False

    aemet_inflight.add(site_id)
    try:
        data = await asyncio.to_thread(fetch_aemet_municipio_horaria, muni)
        mm = extract_rain_forecast_mm(data)
        pb = extract_prob_precip_summary(data)

        aemet_cache_by_site[site_id] = {
            "_epoch": now_epoch,
            "aemet_refreshed_at": datetime.now().isoformat(timespec="seconds"),
            "aemet_error": None,
            **mm,
            **pb,
        }
        return True

    except Exception as e:
        prev = aemet_cache_by_site.get(site_id, {**_default_aemet(), "_epoch": now_epoch})
        aemet_cache_by_site[site_id] = {
            **prev,
            "_epoch": now_epoch,
            "aemet_refreshed_at": datetime.now().isoformat(timespec="seconds"),
            "aemet_error": str(e),
        }
        return True

    finally:
        aemet_inflight.discard(site_id)

def send_notification(title: str, body: str):

    all_tokens = set()

    for site_tokens in tokens.values():

        all_tokens.update(site_tokens)

    print(f"📤 Enviando push a {len(all_tokens)} dispositivos")

    for token in list(all_tokens):

        try:

            message = messaging.Message(

                notification=messaging.Notification(
                    title=title,
                    body=body,
                ),

                token=token,
            )

            response = messaging.send(message)

            print("✅ PUSH OK:", response)

        except Exception as e:

            print("❌ Error enviando:", e)

            # 🔥 ELIMINAR TOKENS INVÁLIDOS
            for site_tokens in tokens.values():

                site_tokens.discard(token)

            guardar_tokens()


# ---------------------------
# WebSocket
# ---------------------------
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)

    default_site = SITES[0]["id"] if SITES else None
    if default_site:
        ws_site[websocket] = default_site
        ws_last_ts[websocket] = None

        # Envío inmediato al conectar
        await websocket.send_text(json.dumps(_build_payload(default_site, forced_is_new=True), ensure_ascii=False))

        # Refrescos inmediatos en background
        async def _refresh_default():
            updated_aemet = await refresh_aemet_for_site(default_site, force=False)
            updated_ia = await refresh_ia_for_site(default_site) if IA_REFRESH_ON_WS else False
            if (updated_aemet or updated_ia) and websocket in clients and ws_site.get(websocket) == default_site:
                await websocket.send_text(json.dumps(_build_payload(default_site, forced_is_new=True), ensure_ascii=False))
        asyncio.create_task(_refresh_default())

    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                continue

            if data.get("type") == "set_site":
                sid = data.get("site")
                if sid in SITES_BY_ID:
                    ws_site[websocket] = sid
                    ws_last_ts[websocket] = None

                    # 1) envío inmediato (cache SAIH + cache AEMET + cache IA existente)
                    await websocket.send_text(json.dumps(_build_payload(sid, forced_is_new=True), ensure_ascii=False))

                    # 2) refrescos inmediatos
                    async def _refresh_and_push(site_id: str):
                        updated_aemet = await refresh_aemet_for_site(site_id, force=False)
                        updated_ia = await refresh_ia_for_site(site_id) if IA_REFRESH_ON_WS else False
                        if updated_aemet or updated_ia:
                            if websocket in clients and ws_site.get(websocket) == site_id:
                                await websocket.send_text(json.dumps(_build_payload(site_id, forced_is_new=True), ensure_ascii=False))
                    asyncio.create_task(_refresh_and_push(sid))

    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)
        ws_site.pop(websocket, None)
        ws_last_ts.pop(websocket, None)


# ---------------------------
# Loops
# ---------------------------
async def _refresh_saih_cache_once():
    """
    Prefetch global, pero en BATCHES para evitar URL gigante y timeouts.
    Además, si un batch falla, no machacamos el cache con None: conservamos el último dato válido.
    """
    try:
        tags = collect_all_tags()
        if not tags:
            return

        BATCH_SIZE = 20
        all_signals: Dict[str, Dict[str, Any]] = {}

        for batch in _chunk(tags, BATCH_SIZE):
            try:
                signals = await asyncio.to_thread(fetch_saih_signals, batch)
                all_signals.update(signals)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    await asyncio.sleep(60)
                print("[SAIH ERROR batch HTTP]", repr(e))
            except Exception as e:
                print("[SAIH ERROR batch]", repr(e))

            await asyncio.sleep(0.2)

        for s in SITES:
            sid = s["id"]
            nivel_tag = (s.get("saih") or {}).get("nivel", "") or ""
            caudal_tag = (s.get("saih") or {}).get("caudal", "") or ""

            nivel = all_signals.get(nivel_tag, {}) if nivel_tag else {}
            caudal = all_signals.get(caudal_tag, {}) if caudal_tag else {}

            ts = (nivel.get("fecha") or caudal.get("fecha")) if (nivel or caudal) else None
            prev = saih_cache_by_site.get(sid, {})

            saih_cache_by_site[sid] = {
                "ts": ts or prev.get("ts"),
                "nivel_m": (nivel.get("valor") if nivel else prev.get("nivel_m")),
                "caudal_m3s": (caudal.get("valor") if caudal else prev.get("caudal_m3s")),
                "tendencia_nivel": (nivel.get("tendencia") if nivel else prev.get("tendencia_nivel")),
                "tendencia_caudal": (caudal.get("tendencia") if caudal else prev.get("tendencia_caudal")),
            }

    except Exception as e:
        print("[SAIH ERROR]", repr(e))

async def _push_to_clients_from_cache():
    for ws in list(clients):
        sid = ws_site.get(ws)
        if not sid:
            continue

        ts = saih_cache_by_site.get(sid, {}).get("ts")
        last_ts = ws_last_ts.get(ws)
        is_new = (ts is not None and ts != last_ts)
        ws_last_ts[ws] = ts

        payload = _build_payload(sid, forced_is_new=is_new)

        try:
            await ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            clients.discard(ws)
            ws_site.pop(ws, None)
            ws_last_ts.pop(ws, None)

async def poll_saih_loop():
    await _refresh_saih_cache_once()
    await _push_to_clients_from_cache()

    while True:
        await _refresh_saih_cache_once()
        await _push_to_clients_from_cache()
        await asyncio.sleep(POLL_SECONDS)

async def poll_aemet_loop():
    """
    Mantiene el cache fresco con TTL para los sitios activos.
    """
    while True:
        try:
            active_sites = set(ws_site.values())
            now_epoch = datetime.now().timestamp()

            for sid in active_sites:
                site = SITES_BY_ID.get(sid)
                if not site:
                    continue

                muni = (site.get("aemet_muni") or "").strip()
                if not muni:
                    continue

                cur = aemet_cache_by_site.get(sid, {**_default_aemet(), "_epoch": None})
                last_epoch = cur.get("_epoch")
                if last_epoch is not None and (now_epoch - float(last_epoch)) < AEMET_REFRESH_SECONDS:
                    continue

                await refresh_aemet_for_site(sid, force=False)

        except Exception as e:
            print("[AEMET LOOP ERROR]", repr(e))

        await asyncio.sleep(AEMET_CHECK_SECONDS)

# async def poll_ia_loop():
#     """
#     Refresca la predicción IA para los sitios activos.
#     No hace falta muy rápido: cada 5 min está bien.
#     """
#     while True:
#         try:
#             active_sites = set(ws_site.values())
#             for sid in active_sites:
#                 await refresh_ia_for_site(sid)
#         except Exception as e:
#             print("[IA LOOP ERROR]", repr(e))

#         await asyncio.sleep(3600)

async def poll_alertas_loop():

    while True:

        try:

            result = await asyncio.to_thread(
                alertas.enviar_alertas_diarias,
                tokens,
                SITES
            )

            limpiar_tokens_invalidos(result.get("invalid_tokens"))

        except Exception as e:

            print("[ALERTAS ERROR]", e)

        # comprobar cada minuto
        await asyncio.sleep(60)


async def run_background_loop(name: str, loop_factory):
    if STARTUP_BACKGROUND_DELAY_SECONDS > 0:
        await asyncio.sleep(STARTUP_BACKGROUND_DELAY_SECONDS)

    try:
        await loop_factory()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"[{name} FATAL]", repr(e))
        traceback.print_exc()


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(run_background_loop("SAIH LOOP", poll_saih_loop))
    asyncio.create_task(run_background_loop("AEMET LOOP", poll_aemet_loop))
    #asyncio.create_task(poll_ia_loop())
    asyncio.create_task(run_background_loop("ALERTAS LOOP", poll_alertas_loop))


@app.on_event("shutdown")
async def on_shutdown():
    if IA_EXECUTOR is not None:
        IA_EXECUTOR.shutdown(wait=False, cancel_futures=True)

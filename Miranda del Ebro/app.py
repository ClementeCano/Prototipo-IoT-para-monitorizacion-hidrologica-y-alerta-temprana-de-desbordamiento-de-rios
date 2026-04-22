import traceback

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from datetime import datetime
import asyncio
import json
from typing import Dict, Any, Set, Optional

import requests
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from saih_opendata import fetch_saih_signals
from aemet_opendata import (
    fetch_aemet_municipio_horaria,
    extract_rain_forecast_mm,
    extract_prob_precip_summary,
)

from prediccion import predecir_semana


# ---------------------------
# Config
# ---------------------------
# SAIH rate limit: 5/min → 20s = 3/min (seguro)
POLL_SECONDS = 20

# AEMET: refresco real cada 30 min, comprobación cada 60s
AEMET_REFRESH_SECONDS = 1800
AEMET_CHECK_SECONDS = 60

app = FastAPI()

SITES = json.loads(Path("sites.json").read_text(encoding="utf-8"))
SITES_BY_ID = {s["id"]: s for s in SITES}

# Cache simple del dataset para IA
_dataset_modelo_cache: Optional[pd.DataFrame] = None


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
@app.get("/")
def home():
    return HTMLResponse(Path("index.html").read_text(encoding="utf-8"))

@app.get("/api/sites")
def api_sites():
    return JSONResponse([{"id": s["id"], "name": s["name"]} for s in SITES])


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

def _load_dataset_modelo() -> pd.DataFrame:
    global _dataset_modelo_cache

    if _dataset_modelo_cache is None:
        BASE_DIR = Path(__file__).resolve().parent
        path = BASE_DIR / "salidas" / "dataset_modelo_final.csv"

        print("📂 Cargando dataset:", path)
        print("EXISTE:", path.exists())

        _dataset_modelo_cache = pd.read_csv(path)

    return _dataset_modelo_cache

def _build_payload(site_id: str, forced_is_new: Optional[bool] = None) -> Dict[str, Any]:
    site = SITES_BY_ID.get(site_id, {"id": site_id, "name": site_id})

    sc = saih_cache_by_site.get(site_id, {})
    ts = sc.get("ts")

    payload = {
        "site_id": site_id,
        "site_name": site.get("name", site_id),

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

def _collect_all_tags() -> list[str]:
    """Prefetch de TODOS los sitios: cambio instantáneo sin esperar tick."""
    tags: list[str] = []
    for s in SITES:
        nivel = (s.get("saih") or {}).get("nivel", "") or ""
        caudal = (s.get("saih") or {}).get("caudal", "") or ""
        if nivel:
            tags.append(nivel)
        if caudal:
            tags.append(caudal)

    out: list[str] = []
    seen = set()
    for t in tags:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def _chunk(lst: list[str], n: int) -> list[list[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def _normalize_site_name_for_model(site_name: str) -> str:
    """
    Ajusta algunos nombres para que coincidan con el dataset_modelo.csv.
    Cambia aquí si tus nombres reales son distintos.
    """
    mapping = {
        "Ascó": "Asco",
        "Castejón": "Castejon",
    }
    return mapping.get(site_name, site_name)

async def refresh_ia_for_site(site_id: str) -> bool:
    print("🚀 llamando a IA para:", site_id)
    """
    Calcula la predicción IA para el sitio actual usando el dataset unificado.
    """
    if predecir_semana is None:
        ia_cache_by_site[site_id] = {
            "ia_refreshed_at": datetime.now().isoformat(timespec="seconds"),
            "ia_error": "No se pudo importar prediccion.py",
            "pred_semana": [],
        }
        return False

    site = SITES_BY_ID.get(site_id)
    if not site:
        return False

    site_name = _normalize_site_name_for_model(site.get("name", site_id))

    try:
        df_modelo = _load_dataset_modelo()

        # Intenta primero con site_name; si la función no lo acepta, cae al fallback
        try:
            pred = predecir_semana(df_modelo, site_name=site_name)
        except TypeError:
            pred = predecir_semana(df_modelo)

        # Validación ligera
        if pred is None:
            pred = []

        ia_cache_by_site[site_id] = {
            "ia_refreshed_at": datetime.now().isoformat(timespec="seconds"),
            "ia_error": None,
            "pred_semana": pred,
        }
        return True

    except Exception as e:
        ia_cache_by_site[site_id] = {
            "ia_refreshed_at": datetime.now().isoformat(timespec="seconds"),
            "ia_error": repr(e),
            "pred_semana": [],
        }
        print("📊 pred:", pred)
        traceback.print_exc()
        return False

async def refresh_aemet_for_site(site_id: str, force: bool = True) -> bool:
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
        if last_epoch is not None and (now_epoch - float(last_epoch)) < AEMET_REFRESH_SECONDS:
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
            "aemet_error": repr(e),
        }
        return True

    finally:
        aemet_inflight.discard(site_id)


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
            updated_aemet = await refresh_aemet_for_site(default_site, force=True)
            updated_ia = await refresh_ia_for_site(default_site)
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
                        updated_aemet = await refresh_aemet_for_site(site_id, force=True)
                        updated_ia = await refresh_ia_for_site(site_id)
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
        tags = _collect_all_tags()
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

async def poll_ia_loop():
    """
    Refresca la predicción IA para los sitios activos.
    No hace falta muy rápido: cada 5 min está bien.
    """
    while True:
        try:
            active_sites = set(ws_site.values())
            for sid in active_sites:
                await refresh_ia_for_site(sid)
        except Exception as e:
            print("[IA LOOP ERROR]", repr(e))

        await asyncio.sleep(300)

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(poll_saih_loop())
    asyncio.create_task(poll_aemet_loop())
    asyncio.create_task(poll_ia_loop())
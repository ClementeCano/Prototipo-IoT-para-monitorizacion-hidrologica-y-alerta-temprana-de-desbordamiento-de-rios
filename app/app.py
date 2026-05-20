import base64
from io import BytesIO
import json
import os
import sys
import unicodedata
from dotenv import load_dotenv

load_dotenv()
import traceback

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
from datetime import date, datetime, timedelta
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, Set, Optional
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from fastapi.staticfiles import StaticFiles

try:
    from app.api.saih_opendata import fetch_saih_history, fetch_saih_signals
    from app.api.aemet_opendata import (
        fetch_aemet_municipio_horaria,
        extract_rain_forecast_mm,
        extract_prob_precip_summary,
    )
    from app.prediccion_individual import predecir_semana_municipio
    from app.core.config import SITES, collect_all_tags
    from app import alertas
    from app.user_store import UserStore, UserStoreError
except ImportError:
    from app.api.saih_opendata import fetch_saih_history, fetch_saih_signals
    from app.api.aemet_opendata import (
        fetch_aemet_municipio_horaria,
        extract_rain_forecast_mm,
        extract_prob_precip_summary,
    )
    from app.prediccion_individual import predecir_semana_municipio
    from app.core.config import SITES, collect_all_tags
    from app import alertas
    from app.user_store import UserStore, UserStoreError

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
        return True

    except Exception as e:

        print("❌ Error guardando tokens:", repr(e))
        return False


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

    try:
        removed += user_store.remove_invalid_tokens(invalid_tokens)
    except Exception as e:
        print("[PUSH CLEANUP] Error limpiando tokens de usuarios:", repr(e))

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
ALERT_CHECK_SECONDS = max(1, int(os.getenv("ALERT_CHECK_SECONDS", "5")))
STARTUP_BACKGROUND_DELAY_SECONDS = int(os.getenv("STARTUP_BACKGROUND_DELAY_SECONDS", "30"))
IA_REFRESH_SECONDS = int(os.getenv("IA_REFRESH_SECONDS", "3600"))
IA_WORKERS = int(os.getenv("IA_WORKERS", "1"))
IA_PROCESS_POOL_ENABLED = os.getenv("IA_PROCESS_POOL_ENABLED", "1").lower() in {"1", "true", "yes"}
IA_REFRESH_ON_WS = os.getenv("IA_REFRESH_ON_WS", "1").lower() in {"1", "true", "yes"}
IA_EXECUTOR = ProcessPoolExecutor(max_workers=IA_WORKERS) if IA_PROCESS_POOL_ENABLED else None
HISTORY_DOWNLOAD_MAX_DAYS = int(os.getenv("HISTORY_DOWNLOAD_MAX_DAYS", "366"))

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

SITES_BY_ID = {s["id"]: s for s in SITES}
user_store = UserStore(sites_by_id=SITES_BY_ID)

SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "rio_session")
SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "0").lower() in {"1", "true", "yes"}
SESSION_COOKIE_MAX_AGE = int(os.getenv("SESSION_COOKIE_MAX_AGE", str(60 * 60 * 24 * 30)))


def _session_token(request: Request) -> str:
    return request.cookies.get(SESSION_COOKIE_NAME, "")


def _require_user(request: Request) -> dict:
    user = user_store.get_user_by_session(_session_token(request))

    if not user:
        raise HTTPException(status_code=401, detail="auth_required")

    return user


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=True,
        secure=SESSION_COOKIE_SECURE,
        samesite="lax",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        httponly=True,
        secure=SESSION_COOKIE_SECURE,
        samesite="lax",
    )

# Cache simple del dataset para IA
_dataset_modelo_cache: Optional[pd.DataFrame] = None


def _filter_alert_sites(site_ids: Optional[str], allow_all: bool = False):
    if not site_ids:
        if allow_all:
            return SITES

        return []

    if site_ids.lower().strip() == "all":
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

    return selected


def _today_madrid() -> date:
    try:
        return datetime.now(ZoneInfo("Europe/Madrid")).date()
    except Exception:
        return date.today()


def _parse_iso_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat((value or "").strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name}_invalid") from exc


def _slugify_filename(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    value = normalized.encode("ascii", "ignore").decode("ascii")
    safe = []

    for char in value.lower():
        if char.isalnum():
            safe.append(char)
        elif char in {" ", "-", "_"}:
            safe.append("_")

    return "_".join("".join(safe).split("_")) or "historico"


def _build_history_dataframe(
    site: dict[str, Any],
    variable: str,
    granularity: str,
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    signal_to_column = {}

    if variable in {"nivel", "both"}:
        signal_to_column[(site.get("saih") or {}).get("nivel")] = "nivel_m"

    if variable in {"caudal", "both"}:
        signal_to_column[(site.get("saih") or {}).get("caudal")] = "caudal_m3s"

    rows_by_date: dict[str, dict[str, Any]] = {}

    for record in records:
        signal = record.get("senal")
        column = signal_to_column.get(signal)

        if not column:
            continue

        timestamp = record.get("fecha")
        if not timestamp:
            continue

        row = rows_by_date.setdefault(timestamp, {
            "fecha": timestamp,
            "municipio": site.get("name"),
        })
        row[column] = record.get("valor")

    columns = ["fecha", "municipio"]

    if variable in {"nivel", "both"}:
        columns.append("nivel_m")

    if variable in {"caudal", "both"}:
        columns.append("caudal_m3s")

    df = pd.DataFrame(rows_by_date.values())

    if df.empty:
        return pd.DataFrame(columns=columns)

    df["fecha_dt"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha_dt"]).sort_values("fecha_dt").reset_index(drop=True)

    for column in columns:
        if column not in df.columns:
            df[column] = None

    value_columns = [column for column in columns if column not in {"fecha", "municipio"}]
    rule = "h" if granularity == "hourly" else "D"
    date_format = "%Y-%m-%d %H:00:00" if granularity == "hourly" else "%Y-%m-%d"

    grouped = (
        df.set_index("fecha_dt")[value_columns]
        .resample(rule)
        .mean()
        .dropna(how="all")
        .reset_index()
    )

    if grouped.empty:
        return pd.DataFrame(columns=columns)

    grouped["fecha"] = grouped["fecha_dt"].dt.strftime(date_format)
    grouped["municipio"] = site.get("name")

    for column in value_columns:
        grouped[column] = grouped[column].round(3)

    return grouped[columns]


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


@app.get("/api/history/download")
async def api_history_download(
    request: Request,
    site_id: str,
    start_date: str,
    end_date: Optional[str] = None,
    variable: str = "both",
    granularity: str = "hourly",
    file_format: str = "xlsx",
):
    _require_user(request)

    site = SITES_BY_ID.get((site_id or "").strip())

    if not site:
        return JSONResponse({"ok": False, "error": "site_not_found"}, status_code=404)

    variable = (variable or "").strip().lower()
    if variable not in {"nivel", "caudal", "both"}:
        return JSONResponse({"ok": False, "error": "variable_invalid"}, status_code=400)

    granularity = (granularity or "").strip().lower()
    if granularity not in {"hourly", "daily"}:
        return JSONResponse({"ok": False, "error": "granularity_invalid"}, status_code=400)

    file_format = (file_format or "").strip().lower()
    if file_format == "excel":
        file_format = "xlsx"

    if file_format not in {"csv", "xlsx"}:
        return JSONResponse({"ok": False, "error": "format_invalid"}, status_code=400)

    max_end_date = _today_madrid() - timedelta(days=1)
    range_start = _parse_iso_date(start_date, "start_date")
    range_end = _parse_iso_date(end_date, "end_date") if end_date else max_end_date

    if range_start > range_end:
        return JSONResponse(
            {
                "ok": False,
                "error": "date_range_invalid",
                "message": "La fecha desde no puede ser posterior a la fecha hasta.",
            },
            status_code=400,
        )

    if range_end > max_end_date:
        return JSONResponse(
            {
                "ok": False,
                "error": "end_date_in_future",
                "message": "La fecha hasta debe ser, como máximo, el día anterior a hoy.",
            },
            status_code=400,
        )

    days_count = (range_end - range_start).days + 1

    if days_count > HISTORY_DOWNLOAD_MAX_DAYS:
        return JSONResponse(
            {
                "ok": False,
                "error": "date_range_too_large",
                "message": f"El rango maximo permitido es de {HISTORY_DOWNLOAD_MAX_DAYS} dias.",
            },
            status_code=400,
        )

    tags = []
    saih_config = site.get("saih") or {}

    if variable in {"nivel", "both"}:
        tags.append(saih_config.get("nivel"))

    if variable in {"caudal", "both"}:
        tags.append(saih_config.get("caudal"))

    tags = [tag for tag in tags if tag]

    if not tags:
        return JSONResponse({"ok": False, "error": "site_without_requested_signals"}, status_code=400)

    try:
        records = await asyncio.to_thread(fetch_saih_history, tags, range_start, range_end)
    except Exception as e:
        print("[SAIH HISTORY ERROR]", repr(e))
        return JSONResponse(
            {
                "ok": False,
                "error": "saih_history_error",
                "message": str(e),
            },
            status_code=502,
        )

    df = _build_history_dataframe(site, variable, granularity, records)

    if df.empty:
        return JSONResponse(
            {
                "ok": False,
                "error": "history_without_data",
                "message": "SAIH no ha devuelto datos para ese rango.",
            },
            status_code=404,
        )

    base_filename = (
        f"historico_{_slugify_filename(site.get('name', site_id))}_"
        f"{variable}_{granularity}_{range_start.isoformat()}_{range_end.isoformat()}"
    )

    if file_format == "csv":
        content = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        media_type = "text/csv; charset=utf-8"
        filename = f"{base_filename}.csv"
    else:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Historico")
        content = buffer.getvalue()
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = f"{base_filename}.xlsx"

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-History-Start-Date": range_start.isoformat(),
            "X-History-End-Date": range_end.isoformat(),
            "X-History-Granularity": granularity,
            "X-History-Rows": str(len(df)),
        },
    )


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/users/me")
def api_current_user(request: Request):
    user = user_store.get_public_user_by_session(_session_token(request))

    return {
        "ok": True,
        "authenticated": user is not None,
        "user": user,
    }


@app.post("/api/users/register")
async def api_register(data: dict, response: Response):
    try:
        user = user_store.create_user(
            data.get("name", ""),
            data.get("email", ""),
            data.get("password", ""),
        )
        session_token = user_store.create_session(user["id"])
        _set_session_cookie(response, session_token)

        return {"ok": True, "user": user}

    except UserStoreError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/users/login")
async def api_login(data: dict, response: Response):
    try:
        user = user_store.authenticate(data.get("email", ""), data.get("password", ""))
    except UserStoreError:
        user = None

    if not user:
        return JSONResponse({"ok": False, "error": "invalid_credentials"}, status_code=401)

    session_token = user_store.create_session(user["id"])
    _set_session_cookie(response, session_token)

    return {"ok": True, "user": user}


@app.post("/api/users/logout")
async def api_logout(request: Request, response: Response):
    user_store.delete_session(_session_token(request))
    _clear_session_cookie(response)

    return {"ok": True}


@app.put("/api/users/preferences")
async def api_update_preferences(data: dict, request: Request):
    user = _require_user(request)

    try:
        public_user = user_store.update_preferences(user["id"], data)
        return {"ok": True, "user": public_user}
    except UserStoreError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.put("/api/users/profile")
async def api_update_profile(data: dict, request: Request):
    user = _require_user(request)
    preferences = data.get("preferences")

    if preferences is not None and not isinstance(preferences, dict):
        return JSONResponse({"ok": False, "error": "preferences_invalid"}, status_code=400)

    try:
        public_user = user_store.update_profile(
            user["id"],
            name=data.get("name"),
            preferences=preferences,
        )
        return {"ok": True, "user": public_user}
    except UserStoreError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.get("/api/email/config")
def api_email_config():
    return {
        "ok": True,
        "smtp": alertas.smtp_config_status(),
        "message": alertas.smtp_config_message(),
    }


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
        "selectedSites": data.get("selectedSites"),
        "errorCode": data.get("errorCode"),
        "errorMessage": data.get("errorMessage"),
        "tokenPrefix": data.get("tokenPrefix"),
        "href": data.get("href"),
        "userAgent": data.get("userAgent"),
    }
    print("[PUSH DEBUG]", json.dumps(safe_data, ensure_ascii=False))
    return {"ok": True}



def _persist_push_subscription(
    user: dict,
    token: str,
    sites: list,
    user_agent: str = "",
    platform: str = "",
) -> dict:
    valid_sites = []

    for site in sites:
        if site in SITES_BY_ID and site not in valid_sites:
            valid_sites.append(site)
        else:
            print(f"⚠️ Site ignorado: {site!r}")

    public_user = user_store.save_push_subscription(
        user["id"],
        token,
        valid_sites,
        user_agent=user_agent,
        platform=platform,
    )

    for site_tokens in tokens.values():
        site_tokens.discard(token)

    for site in valid_sites:
        tokens[site].add(token)
        print(f"🔥 Token guardado en {site}")

    if not guardar_tokens():
        raise RuntimeError("tokens_storage_error")

    return {
        "sites": valid_sites,
        "token_saved": bool(valid_sites),
        "total_tokens": sum(len(v) for v in tokens.values()),
        "user": public_user,
    }


@app.post("/api/token")
async def save_token(data: dict, request: Request):

    print("📩 Token recibido")

    user = _require_user(request)

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

    try:
        public_user = user_store.save_push_subscription(
            user["id"],
            token,
            valid_sites,
            user_agent=user_agent,
            platform=client_platform,
        )
    except UserStoreError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    except Exception as e:
        print(f"[PUSH TOKEN ERROR] users_file={user_store.path} error={repr(e)}")
        return JSONResponse(
            {
                "ok": False,
                "error": "push_storage_error",
                "message": "No se ha podido guardar el token push en el usuario.",
            },
            status_code=500,
        )

    # eliminar token previo
    for site_tokens in tokens.values():
        site_tokens.discard(token)

    # guardar nuevo
    for site in valid_sites:

        tokens[site].add(token)

        print(f"🔥 Token guardado en {site}")

    # 🔥 GUARDAR EN DISCO
    if not guardar_tokens():
        return JSONResponse(
            {
                "ok": False,
                "error": "tokens_storage_error",
                "message": "No se ha podido guardar el token push en el servidor.",
            },
            status_code=500,
        )

    total = sum(len(v) for v in tokens.values())

    return {
        "ok": True,
        "sites": valid_sites,
        "token_saved": bool(valid_sites),
        "total_tokens": total,
        "user": public_user,
    }


@app.post("/api/test-token")
async def test_token(data: dict, request: Request):
    _require_user(request)

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
        "errors": result.get("errors", []),
        "tokenPrefix": token[:15],
    }


@app.post("/api/test-selected-alerts")
async def test_selected_alerts(data: dict, request: Request):
    user = _require_user(request)
    token = (data.get("token") or "").strip()
    sites = data.get("sites", [])
    user_agent = (data.get("userAgent") or "")[:180]
    client_platform = (data.get("platform") or "")[:80]

    if not isinstance(sites, list) or not sites:
        return JSONResponse({"ok": False, "error": "sites_required"}, status_code=400)

    valid_sites = []

    for site_id in sites:
        if site_id in SITES_BY_ID and site_id not in valid_sites:
            valid_sites.append(site_id)

    if not valid_sites:
        return JSONResponse({"ok": False, "error": "no_valid_sites"}, status_code=400)

    alert_sites = [SITES_BY_ID[site_id] for site_id in valid_sites]
    preferences = {
        **(user.get("preferences") or {}),
        "sites": valid_sites,
    }
    temp_user = {
        **user,
        "preferences": preferences,
        "devices": user.get("devices", []),
    }

    subscription = None

    if preferences.get("notification_channel") != "email":
        if not token:
            return JSONResponse({"ok": False, "error": "token_empty"}, status_code=400)

        try:
            subscription = _persist_push_subscription(
                user,
                token,
                valid_sites,
                user_agent=user_agent,
                platform=client_platform,
            )
        except UserStoreError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        except RuntimeError as e:
            if str(e) == "tokens_storage_error":
                return JSONResponse(
                    {
                        "ok": False,
                        "error": "tokens_storage_error",
                        "message": "No se ha podido guardar el token push en el servidor.",
                    },
                    status_code=500,
                )
            raise

        temp_user["devices"] = [{"token": token}]

    print(f"[ALERT TEST] user={user.get('email')} channel={preferences.get('notification_channel')} sites={valid_sites}")

    result = await asyncio.to_thread(
        alertas.enviar_alerta_usuario,
        temp_user,
        alert_sites,
    )

    removed = limpiar_tokens_invalidos(result.get("invalid_tokens"))

    return {
        "ok": True,
        "sent": result.get("sent", 0),
        "processed_sites": result.get("processed_sites", 0),
        "sites": valid_sites,
        "invalid_subscriptions_removed": removed,
        "channel": preferences.get("notification_channel"),
        "token_saved": subscription.get("token_saved") if subscription else None,
        "errors": result.get("errors", []),
        "tokenPrefix": token[:15] if token else None,
    }


@app.get("/test-alerts-now")
async def test_alerts_now(request: Request, sites: Optional[str] = None, all_sites: bool = False):
    user = _require_user(request)

    alert_sites = _filter_alert_sites(sites, allow_all=all_sites)

    if not alert_sites:
        selected_sites = (user.get("preferences") or {}).get("sites", [])
        alert_sites = [
            SITES_BY_ID[site_id]
            for site_id in selected_sites
            if site_id in SITES_BY_ID
        ]

    if not alert_sites:
        return JSONResponse(
            {
                "ok": False,
                "error": "sites_required",
                "message": "Indica municipios, por ejemplo /test-alerts-now?sites=miranda,palazuelos. Para probar todos usa ?sites=all.",
            },
            status_code=400,
        )

    site_ids = [site["id"] for site in alert_sites]
    temp_user = {
        **user,
        "preferences": {
            **(user.get("preferences") or {}),
            "sites": site_ids,
        },
    }

    result = await asyncio.to_thread(
        alertas.enviar_alerta_usuario,
        temp_user,
        alert_sites,
    )

    removed = limpiar_tokens_invalidos(result.get("invalid_tokens"))

    return {
        "ok": True,
        "status": "alertas enviadas",
        "sent": result.get("sent", 0),
        "processed_sites": result.get("processed_sites", 0),
        "sites": site_ids,
        "channel": (temp_user.get("preferences") or {}).get("notification_channel"),
        "errors": result.get("errors", []),
        "invalid_subscriptions_removed": removed,
    }


@app.get("/test-alert")
async def test_alert(request: Request, sites: Optional[str] = None, all_sites: bool = False):
    return await test_alerts_now(request=request, sites=sites, all_sites=all_sites)


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
            due_users = user_store.users_due_for_alert()

            if not due_users:
                await asyncio.sleep(ALERT_CHECK_SECONDS)
                continue

            result = await asyncio.to_thread(
                alertas.enviar_alertas_usuarios,
                due_users,
                SITES
            )

            limpiar_tokens_invalidos(result.get("invalid_tokens"))

            for user_id, user_result in result.get("per_user", {}).items():
                if int(user_result.get("sent", 0)) > 0:
                    user_store.mark_alert_result(user_id, user_result)
                else:
                    print(
                        "[ALERTAS LOOP] No se marca como enviada "
                        f"user_id={user_id} reason={user_result.get('reason')} "
                        f"errors={user_result.get('errors')}"
                    )

        except Exception as e:

            print("[ALERTAS ERROR]", e)

        await asyncio.sleep(ALERT_CHECK_SECONDS)


async def run_background_loop(name: str, loop_factory, startup_delay: Optional[int] = None):
    delay = STARTUP_BACKGROUND_DELAY_SECONDS if startup_delay is None else startup_delay

    if delay > 0:
        await asyncio.sleep(delay)

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
    asyncio.create_task(run_background_loop("ALERTAS LOOP", poll_alertas_loop, startup_delay=0))


@app.on_event("shutdown")
async def on_shutdown():
    if IA_EXECUTOR is not None:
        IA_EXECUTOR.shutdown(wait=False, cancel_futures=True)

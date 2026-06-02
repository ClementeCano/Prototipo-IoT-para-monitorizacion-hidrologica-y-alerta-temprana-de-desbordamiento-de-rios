import base64
from io import BytesIO
import json
import os
import sys
import unicodedata
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
APP_DIR = Path(__file__).resolve().parent
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
        firebase_candidate = Path(firebase_path)

        if firebase_candidate.is_absolute():
            firebase_cred_source = str(firebase_candidate)
        else:
            cwd_candidate = Path.cwd() / firebase_candidate
            app_candidate = APP_DIR / firebase_candidate
            firebase_cred_source = str(
                cwd_candidate
                if cwd_candidate.exists()
                else app_candidate
                if app_candidate.exists()
                else firebase_candidate
            )

    if not firebase_cred_source:
        raise RuntimeError(
            "Configura FIREBASE_CREDENTIALS, FIREBASE_CREDENTIALS_JSON o FIREBASE_CREDENTIALS_JSON_BASE64"
        )

    cred = credentials.Certificate(firebase_cred_source)

    firebase_admin.initialize_app(cred)

    print("✅ Firebase inicializado")

except Exception as e:
    print("❌ Error Firebase:", e)

from datetime import date, datetime, timedelta
import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, Set, Optional
from zoneinfo import ZoneInfo

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
    from app.prediction_store import create_prediction_store
    from app.core.config import SITES, collect_all_tags
    from app import alertas
    from app.user_store import UserStoreError, create_user_store
except ImportError:
    from app.api.saih_opendata import fetch_saih_history, fetch_saih_signals
    from app.api.aemet_opendata import (
        fetch_aemet_municipio_horaria,
        extract_rain_forecast_mm,
        extract_prob_precip_summary,
    )
    from app.prediccion_individual import predecir_semana_municipio
    from app.prediction_store import create_prediction_store
    from app.core.config import SITES, collect_all_tags
    from app import alertas
    from app.user_store import UserStoreError, create_user_store

from collections import defaultdict

# =========================
# TOKENS PERSISTENTES
# =========================

# 🔥 usar ruta absoluta estable
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR)).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

TOKENS_FILE = Path(os.getenv("TOKENS_FILE", DATA_DIR / "tokens.json")).resolve()
TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
print("DATA_DIR:", DATA_DIR)

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
# SAIH rate-limit: se refresca en segundo plano y se evita llamar por cada click.
POLL_SECONDS = int(os.getenv("SAIH_POLL_SECONDS", "180"))
SAIH_BATCH_SIZE = max(1, int(os.getenv("SAIH_BATCH_SIZE", "20")))
SAIH_BATCH_DELAY_SECONDS = float(os.getenv("SAIH_BATCH_DELAY_SECONDS", "12"))
SAIH_REQUEST_MAX_SECONDS = float(os.getenv("SAIH_REQUEST_MAX_SECONDS", "15"))
SAIH_RATE_LIMIT_COOLDOWN_SECONDS = int(os.getenv("SAIH_RATE_LIMIT_COOLDOWN_SECONDS", "300"))
SAIH_REFRESH_ON_WS = os.getenv("SAIH_REFRESH_ON_WS", "0").lower() in {"1", "true", "yes"}
SAIH_SITE_REFRESH_MIN_SECONDS = int(os.getenv("SAIH_SITE_REFRESH_MIN_SECONDS", "300"))

# AEMET: refresco real cada 30 min, comprobación cada 60s
AEMET_REFRESH_SECONDS = 1800
AEMET_CHECK_SECONDS = 60
AEMET_ERROR_RETRY_SECONDS = int(os.getenv("AEMET_ERROR_RETRY_SECONDS", "300"))
SAIH_STALE_HOURS = int(os.getenv("SAIH_STALE_HOURS", "72"))
ALERT_CHECK_SECONDS = max(1, int(os.getenv("ALERT_CHECK_SECONDS", "5")))
STARTUP_BACKGROUND_DELAY_SECONDS = int(os.getenv("STARTUP_BACKGROUND_DELAY_SECONDS", "30"))
IA_REFRESH_SECONDS = int(os.getenv("IA_REFRESH_SECONDS", "3600"))
IA_WORKERS = int(os.getenv("IA_WORKERS", "1"))
IA_PROCESS_POOL_ENABLED = os.getenv("IA_PROCESS_POOL_ENABLED", "1").lower() in {"1", "true", "yes"}
IA_REFRESH_ON_WS = os.getenv("IA_REFRESH_ON_WS", "0").lower() in {"1", "true", "yes"}
IA_EXECUTOR = ProcessPoolExecutor(max_workers=IA_WORKERS) if IA_PROCESS_POOL_ENABLED else None
HISTORY_DOWNLOAD_MAX_DAYS = int(os.getenv("HISTORY_DOWNLOAD_MAX_DAYS", "366"))
PREDICTION_EVAL_LOOKBACK_DAYS = int(os.getenv("PREDICTION_EVAL_LOOKBACK_DAYS", "30"))
PREDICTION_EVAL_CHECK_SECONDS = int(os.getenv("PREDICTION_EVAL_CHECK_SECONDS", "3600"))
PREDICTION_EVAL_INCLUDE_TODAY = os.getenv("PREDICTION_EVAL_INCLUDE_TODAY", "1").lower() in {"1", "true", "yes"}
PREDICTION_EVAL_MIN_REFRESH_SECONDS = int(os.getenv("PREDICTION_EVAL_MIN_REFRESH_SECONDS", "600"))
PREDICTION_DAILY_REFRESH_ENABLED = os.getenv("PREDICTION_DAILY_REFRESH_ENABLED", "1").lower() in {"1", "true", "yes"}
PREDICTION_DAILY_REFRESH_HOUR = int(os.getenv("PREDICTION_DAILY_REFRESH_HOUR", "6"))
PREDICTION_DAILY_REFRESH_MINUTE = int(os.getenv("PREDICTION_DAILY_REFRESH_MINUTE", "0"))
PREDICTION_DAILY_CHECK_SECONDS = int(os.getenv("PREDICTION_DAILY_CHECK_SECONDS", "600"))
PREDICTION_DAILY_STARTUP_GRACE_SECONDS = int(os.getenv("PREDICTION_DAILY_STARTUP_GRACE_SECONDS", "1800"))

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

SITES_BY_ID = {s["id"]: s for s in SITES}
user_store = create_user_store(sites_by_id=SITES_BY_ID)
print("USER_STORE:", getattr(user_store, "storage_backend", "json"), user_store.path)
prediction_store = create_prediction_store()
print("PREDICTION_STORE:", getattr(prediction_store, "storage_backend", "json"), prediction_store.path)

try:
    stored_token_map = user_store.token_site_map()
    if stored_token_map:
        tokens = defaultdict(set, stored_token_map)
        guardar_tokens()
except Exception as e:
    print("[PUSH TOKEN SYNC] Error sincronizando tokens desde usuarios:", repr(e))

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


def _now_madrid() -> datetime:
    try:
        return datetime.now(ZoneInfo(os.getenv("ALERT_TIMEZONE", "Europe/Madrid")))
    except Exception:
        return datetime.now()


def _today_madrid() -> date:
    return _now_madrid().date()


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


def _prediction_actual_tags(site: dict[str, Any]) -> list[str]:
    saih_config = site.get("saih") or {}
    return [
        tag
        for tag in [saih_config.get("nivel"), saih_config.get("caudal")]
        if tag
    ]


def _number_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _daily_actuals_from_records(site: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    df = _build_history_dataframe(site, "both", "daily", records)
    actuals: dict[str, dict[str, Any]] = {}

    if df.empty:
        return actuals

    for _, row in df.iterrows():
        day = str(row.get("fecha") or "").strip()
        if not day:
            continue

        actuals[day] = {
            "nivel_m": _number_or_none(row.get("nivel_m")),
            "caudal_m3s": _number_or_none(row.get("caudal_m3s")),
            "observed_at": day,
        }

    return actuals


def _parse_saih_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None

    try:
        return parsed.to_pydatetime().replace(tzinfo=None)
    except Exception:
        return None


def _saih_signal_issue(signal: dict[str, Any], name: str) -> Optional[str]:
    if not signal:
        return f"{name} sin respuesta"

    if signal.get("valor") is None:
        return f"{name} sin valor publicado"

    timestamp = _parse_saih_timestamp(signal.get("fecha"))
    if not timestamp:
        return f"{name} sin fecha valida"

    age_hours = (datetime.now() - timestamp).total_seconds() / 3600
    if SAIH_STALE_HOURS > 0 and age_hours > SAIH_STALE_HOURS:
        return f"{name} desactualizado desde {signal.get('fecha')}"

    return None


def _is_saih_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "too many requests" in text


def _saih_rate_limit_seconds_left() -> int:
    if not saih_rate_limit_until:
        return 0

    seconds = int((saih_rate_limit_until - datetime.now()).total_seconds())
    return max(0, seconds)


def _set_saih_rate_limit(reason: str) -> None:
    global saih_rate_limit_until

    until = datetime.now() + timedelta(seconds=SAIH_RATE_LIMIT_COOLDOWN_SECONDS)
    if not saih_rate_limit_until or until > saih_rate_limit_until:
        saih_rate_limit_until = until

    message = (
        "SAIH Ebro ha limitado temporalmente las peticiones. "
        "No es un fallo de la web; se reintentara automaticamente."
    )

    for sid, prev in list(saih_cache_by_site.items()):
        saih_cache_by_site[sid] = {
            **prev,
            "saih_error": message,
            "saih_error_detail": reason,
        }

    print(
        "[SAIH RATE LIMIT] cooldown="
        f"{SAIH_RATE_LIMIT_COOLDOWN_SECONDS}s reason={reason}"
    )


async def refresh_prediction_actuals_for_site(site_id: str) -> dict[str, Any]:
    site = SITES_BY_ID.get(site_id)
    if not site:
        return {"checked": False, "error": "site_not_found"}

    cooldown_left = _saih_rate_limit_seconds_left()
    if cooldown_left > 0:
        return {
            "checked": False,
            "updated": 0,
            "skipped": "saih_rate_limited",
            "next_check_seconds": cooldown_left,
        }

    today = _today_madrid()
    max_date = today if PREDICTION_EVAL_INCLUDE_TODAY else today - timedelta(days=1)
    refresh_date = today if PREDICTION_EVAL_INCLUDE_TODAY else None
    pending_dates = await asyncio.to_thread(
        prediction_store.pending_actual_dates,
        site_id,
        max_date,
        PREDICTION_EVAL_LOOKBACK_DAYS,
        refresh_date,
    )

    if not pending_dates:
        return {"checked": True, "pending_dates": 0, "updated": 0}

    now_epoch = datetime.now().timestamp()
    last_epoch = prediction_actual_refresh_epoch_by_site.get(site_id)

    if (
        last_epoch is not None
        and PREDICTION_EVAL_MIN_REFRESH_SECONDS > 0
        and (now_epoch - last_epoch) < PREDICTION_EVAL_MIN_REFRESH_SECONDS
    ):
        return {
            "checked": True,
            "pending_dates": len(pending_dates),
            "updated": 0,
            "skipped": "recently_checked",
            "next_check_seconds": round(PREDICTION_EVAL_MIN_REFRESH_SECONDS - (now_epoch - last_epoch)),
        }

    tags = _prediction_actual_tags(site)
    if not tags:
        return {
            "checked": False,
            "pending_dates": len(pending_dates),
            "updated": 0,
            "error": "site_without_saih_signals",
        }

    range_start = min(pending_dates)
    range_end = max(pending_dates)

    try:
        prediction_actual_refresh_epoch_by_site[site_id] = now_epoch
        records = await asyncio.to_thread(fetch_saih_history, tags, range_start, range_end)
        actuals = _daily_actuals_from_records(site, records)
        updated = await asyncio.to_thread(prediction_store.update_actuals, site_id, actuals)
    except Exception as e:
        print("[PREDICTION EVAL ERROR]", site_id, repr(e))
        if _is_saih_rate_limit_error(e):
            _set_saih_rate_limit(str(e))

        return {
            "checked": False,
            "pending_dates": len(pending_dates),
            "updated": 0,
            "from": range_start.isoformat(),
            "to": range_end.isoformat(),
            "error": str(e),
        }

    return {
        "checked": True,
        "pending_dates": len(pending_dates),
        "updated": updated,
        "from": range_start.isoformat(),
        "to": range_end.isoformat(),
    }


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
saih_rate_limit_until: Optional[datetime] = None
saih_refresh_epoch_by_site: Dict[str, float] = {}

# IA cache por sitio
ia_cache_by_site: Dict[str, Dict[str, Any]] = {}
ia_inflight: set[str] = set()
ia_epoch_by_site: Dict[str, float] = {}
ia_reliability_cache_by_site: Dict[str, Dict[str, Any]] = {}
daily_prediction_dates_run: set[str] = set()
prediction_actual_refresh_epoch_by_site: Dict[str, float] = {}

def _default_ia(store_checked: bool = False) -> Dict[str, Any]:
    return {
        "ia_refreshed_at": None,
        "ia_error": None,
        "pred_semana": [],
        "pred_semana_source": None,
        "pred_semana_store_checked": store_checked,
    }


def _stored_ia(site_id: str) -> Dict[str, Any]:
    try:
        forecast = prediction_store.latest_forecast(site_id)
    except Exception as e:
        print("[PREDICTION FORECAST LOAD ERROR]", site_id, repr(e))
        return _default_ia(store_checked=True)

    if not forecast:
        return _default_ia(store_checked=True)

    return {
        "ia_refreshed_at": forecast.get("issued_at"),
        "ia_error": None,
        "pred_semana": forecast.get("predictions") or [],
        "pred_semana_source": "persisted",
        "pred_semana_store_checked": True,
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
            "saih_error": None,
            "saih_error_detail": None,
        })
        aemet_cache_by_site.setdefault(sid, {**_default_aemet(), "_epoch": None})
        ia_cache_by_site.setdefault(sid, _stored_ia(sid))

_init_caches()


# ---------------------------
# HTTP routes
# ---------------------------
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


@app.get("/api/prediction/reliability/{site_id}")
async def api_prediction_reliability(site_id: str):
    if site_id not in SITES_BY_ID:
        raise HTTPException(status_code=404, detail="site_not_found")

    update_result = await refresh_prediction_actuals_for_site(site_id)
    result = await asyncio.to_thread(prediction_store.evaluation, site_id, 30)
    result["generated_at"] = datetime.now().isoformat(timespec="seconds")
    result["storage_backend"] = getattr(prediction_store, "storage_backend", "json")
    result["update"] = update_result

    return {"ok": True, "site_id": site_id, **result}


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
        return JSONResponse(
            {
                "ok": False,
                "error": "site_without_requested_signals",
                "message": "Este municipio no tiene senales SAIH para el dato solicitado. No es un fallo de la web.",
            },
            status_code=400,
        )

    try:
        records = await asyncio.to_thread(fetch_saih_history, tags, range_start, range_end)
    except Exception as e:
        print("[SAIH HISTORY ERROR]", repr(e))
        return JSONResponse(
            {
                "ok": False,
                "error": "saih_history_error",
                "message": (
                    "No se ha podido descargar el historico porque la API de SAIH Ebro "
                    "no ha respondido correctamente. No es un fallo de la web; prueba mas tarde "
                    "o con otro tramo de fechas."
                ),
                "detail": str(e),
            },
            status_code=502,
        )

    df = _build_history_dataframe(site, variable, granularity, records)

    if df.empty:
        return JSONResponse(
            {
                "ok": False,
                "error": "history_without_data",
                "message": (
                    "SAIH Ebro no ha devuelto datos para ese municipio, variable o tramo de fechas. "
                    "No es un fallo de la web; ese periodo puede no tener datos publicados."
                ),
            },
            status_code=404,
        )

    history_warning = ""
    try:
        observed_days = {
            pd.Timestamp(value).date()
            for value in pd.to_datetime(df["fecha"], errors="coerce").dropna()
        }
        days_count = (range_end - range_start).days + 1

        if len(observed_days) < days_count:
            missing_days = days_count - len(observed_days)
            history_warning = (
                f"SAIH Ebro no ha devuelto datos para {missing_days} dia(s) del tramo seleccionado; "
                "el archivo incluye solo los registros disponibles."
            )
    except Exception:
        history_warning = ""

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
            **({"X-History-Warning": history_warning} if history_warning else {}),
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


@app.get("/api/users/downloads")
def api_user_downloads(request: Request, limit: int = 50):
    user = _require_user(request)

    try:
        downloads = user_store.list_downloads(user["id"], limit=limit)
        return {"ok": True, "downloads": downloads}
    except UserStoreError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/users/downloads")
async def api_record_download(data: dict, request: Request):
    user = _require_user(request)

    try:
        download = user_store.record_download(user["id"], data)
        return {"ok": True, "download": download}
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

    print("[PUSH TOKEN] Token recibido")

    user = _require_user(request)

    token = (data.get("token") or "").strip()
    sites = data.get("sites", [])
    user_agent = (data.get("userAgent") or "")[:180]
    client_platform = (data.get("platform") or "")[:80]

    if not token:

        print("[PUSH TOKEN] Token vacio")

        return JSONResponse({"ok": False, "error": "token_empty"}, status_code=400)

    if not isinstance(sites, list):

        print("[PUSH TOKEN] Sites invalidos")

        return JSONResponse({"ok": False, "error": "sites_must_be_list"}, status_code=400)

    print(
        f"[PUSH TOKEN] prefix={token[:15]} "
        f"sites={len(sites)} platform={client_platform or '-'} ua={user_agent or '-'}"
    )

    try:
        subscription = _persist_push_subscription(
            user,
            token,
            sites,
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
    except Exception as e:
        print(f"[PUSH TOKEN ERROR] user_store={user_store.path} error={repr(e)}")
        return JSONResponse(
            {
                "ok": False,
                "error": "push_storage_error",
                "message": "No se ha podido guardar el token push en el usuario.",
            },
            status_code=500,
        )

    return {
        "ok": True,
        **subscription,
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
        stored = _stored_ia(site_id)
        ia_cache_by_site[site_id] = stored
        return stored

    if not c.get("pred_semana") and not c.get("pred_semana_store_checked"):
        stored = _stored_ia(site_id)
        if stored.get("pred_semana"):
            ia_cache_by_site[site_id] = stored
            return stored
        ia_cache_by_site[site_id] = {**c, **stored}
        return ia_cache_by_site[site_id]

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
        "saih_error": sc.get("saih_error"),
        "saih_error_detail": sc.get("saih_error_detail"),

        **_aemet_public_cache(site_id),
        **_ia_public_cache(site_id),
    }
    return payload

def _chunk(lst: list[str], n: int) -> list[list[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]


async def _run_prediction(site_id: str, use_live_saih: Optional[bool] = None):
    if IA_EXECUTOR is not None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(IA_EXECUTOR, predecir_semana_municipio, site_id, use_live_saih)

    return await asyncio.to_thread(predecir_semana_municipio, site_id, use_live_saih)


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
        pred = await _run_prediction(site_id, use_live_saih=True if force else None)

        if pred is None:
            pred = []

        stored_predictions = 0
        stored_forecast = 0
        site = SITES_BY_ID.get(site_id)

        if site and pred:
            try:
                stored_forecast = await asyncio.to_thread(
                    prediction_store.store_forecast,
                    site,
                    pred,
                )
                stored_predictions = await asyncio.to_thread(
                    prediction_store.store_prediction,
                    site,
                    pred,
                )
                ia_reliability_cache_by_site.pop(site_id, None)
            except Exception as e:
                print("[PREDICTION STORE ERROR]", site_id, repr(e))

        ia_cache_by_site[site_id] = {
            "ia_refreshed_at": datetime.now().isoformat(timespec="seconds"),
            "ia_error": None,
            "pred_semana": pred,
            "pred_semana_persistida": stored_predictions,
            "pred_semana_forecast_persistido": stored_forecast,
            "pred_semana_source": "fresh",
            "pred_semana_store_checked": True,
        }
        ia_epoch_by_site[site_id] = now_epoch

        return True

    except Exception as e:
        ia_cache_by_site[site_id] = {
            "ia_refreshed_at": datetime.now().isoformat(timespec="seconds"),
            "ia_error": repr(e),
            "pred_semana": [],
            "pred_semana_source": None,
            "pred_semana_store_checked": True,
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
                        updated_saih = await refresh_saih_for_site(site_id) if SAIH_REFRESH_ON_WS else False
                        if updated_saih:
                            if websocket in clients and ws_site.get(websocket) == site_id:
                                await websocket.send_text(json.dumps(_build_payload(site_id, forced_is_new=True), ensure_ascii=False))

                        updated_aemet = await refresh_aemet_for_site(site_id, force=False)
                        if updated_aemet:
                            if websocket in clients and ws_site.get(websocket) == site_id:
                                await websocket.send_text(json.dumps(_build_payload(site_id, forced_is_new=True), ensure_ascii=False))

                        updated_ia = await refresh_ia_for_site(site_id) if IA_REFRESH_ON_WS else False
                        if updated_ia:
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
def _update_saih_cache_for_site(
    site: dict[str, Any],
    all_signals: dict[str, dict[str, Any]],
    batch_errors: Optional[list[str]] = None,
) -> None:
    sid = site["id"]
    nivel_tag = (site.get("saih") or {}).get("nivel", "") or ""
    caudal_tag = (site.get("saih") or {}).get("caudal", "") or ""

    nivel = all_signals.get(nivel_tag, {}) if nivel_tag else {}
    caudal = all_signals.get(caudal_tag, {}) if caudal_tag else {}

    prev = saih_cache_by_site.get(sid, {})
    missing_signals = []
    nivel_issue = _saih_signal_issue(nivel, "nivel") if nivel_tag else None
    caudal_issue = _saih_signal_issue(caudal, "caudal") if caudal_tag else None
    valid_nivel = bool(nivel) and nivel_issue is None
    valid_caudal = bool(caudal) and caudal_issue is None
    ts = (
        (nivel.get("fecha") if valid_nivel else None)
        or (caudal.get("fecha") if valid_caudal else None)
    )

    if nivel_issue:
        missing_signals.append(nivel_issue)
    if caudal_issue:
        missing_signals.append(caudal_issue)

    if not nivel_tag and not caudal_tag:
        saih_error = "Este municipio no tiene senales SAIH configuradas."
        saih_error_detail = "site_without_saih_signals"
    elif missing_signals:
        saih_error = (
            "SAIH Ebro no esta devolviendo datos validos para este municipio. "
            "No es un fallo de la web."
        )
        saih_error_detail = "; ".join([*missing_signals, *(batch_errors or [])])
    else:
        saih_error = None
        saih_error_detail = None

    saih_cache_by_site[sid] = {
        "ts": ts or prev.get("ts"),
        "nivel_m": (nivel.get("valor") if valid_nivel else prev.get("nivel_m")),
        "caudal_m3s": (caudal.get("valor") if valid_caudal else prev.get("caudal_m3s")),
        "tendencia_nivel": (nivel.get("tendencia") if valid_nivel else prev.get("tendencia_nivel")),
        "tendencia_caudal": (caudal.get("tendencia") if valid_caudal else prev.get("tendencia_caudal")),
        "saih_error": saih_error,
        "saih_error_detail": saih_error_detail,
    }


async def refresh_saih_for_site(site_id: str) -> bool:
    cooldown_left = _saih_rate_limit_seconds_left()
    if cooldown_left > 0:
        return False

    now_epoch = datetime.now().timestamp()
    last_epoch = saih_refresh_epoch_by_site.get(site_id)
    if (
        last_epoch is not None
        and SAIH_SITE_REFRESH_MIN_SECONDS > 0
        and (now_epoch - last_epoch) < SAIH_SITE_REFRESH_MIN_SECONDS
    ):
        return False

    site = SITES_BY_ID.get(site_id)
    if not site:
        return False

    tags = [
        tag
        for tag in [
            (site.get("saih") or {}).get("nivel"),
            (site.get("saih") or {}).get("caudal"),
        ]
        if tag
    ]

    if not tags:
        _update_saih_cache_for_site(site, {}, [])
        return True

    try:
        saih_refresh_epoch_by_site[site_id] = now_epoch
        signals = await asyncio.to_thread(
            fetch_saih_signals,
            tags,
            (3, 8),
            1,
            SAIH_REQUEST_MAX_SECONDS,
        )
        _update_saih_cache_for_site(site, signals, [])
    except Exception as e:
        print("[SAIH SITE ERROR]", site_id, repr(e))
        if _is_saih_rate_limit_error(e):
            _set_saih_rate_limit(str(e))

        prev = saih_cache_by_site.get(site_id, {})
        saih_cache_by_site[site_id] = {
            **prev,
            "saih_error": (
                "SAIH Ebro ha limitado temporalmente las peticiones. No es un fallo de la web."
                if _is_saih_rate_limit_error(e)
                else "SAIH Ebro no esta respondiendo ahora mismo. No es un fallo de la web."
            ),
            "saih_error_detail": str(e),
        }

    return True


async def _refresh_saih_cache_once():
    """
    Prefetch global, pero en BATCHES para evitar URL gigante y timeouts.
    Además, si un batch falla, no machacamos el cache con None: conservamos el último dato válido.
    """
    try:
        cooldown_left = _saih_rate_limit_seconds_left()
        if cooldown_left > 0:
            print(f"[SAIH RATE LIMIT] refresco global omitido, quedan {cooldown_left}s")
            return

        tags = collect_all_tags()
        if not tags:
            return

        all_signals: Dict[str, Dict[str, Any]] = {}
        batch_errors: list[str] = []
        batches = _chunk(tags, SAIH_BATCH_SIZE)

        for index, batch in enumerate(batches):
            try:
                signals = await asyncio.to_thread(
                    fetch_saih_signals,
                    batch,
                    (3, 8),
                    1,
                    SAIH_REQUEST_MAX_SECONDS,
                )
                all_signals.update(signals)
            except Exception as e:
                print("[SAIH ERROR batch]", repr(e))
                batch_errors.append(str(e))
                if _is_saih_rate_limit_error(e):
                    _set_saih_rate_limit(str(e))
                    break

            if SAIH_BATCH_DELAY_SECONDS > 0 and index < len(batches) - 1:
                await asyncio.sleep(SAIH_BATCH_DELAY_SECONDS)

        for s in SITES:
            _update_saih_cache_for_site(s, all_signals, batch_errors)

    except Exception as e:
        print("[SAIH ERROR]", repr(e))
        if _is_saih_rate_limit_error(e):
            _set_saih_rate_limit(str(e))

        for sid, prev in list(saih_cache_by_site.items()):
            saih_cache_by_site[sid] = {
                **prev,
                "saih_error": (
                    "SAIH Ebro ha limitado temporalmente las peticiones. No es un fallo de la web."
                    if _is_saih_rate_limit_error(e)
                    else "SAIH Ebro no esta respondiendo ahora mismo. No es un fallo de la web."
                ),
                "saih_error_detail": str(e),
            }

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
        await asyncio.sleep(POLL_SECONDS)
        await _refresh_saih_cache_once()
        await _push_to_clients_from_cache()

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

async def poll_prediction_evaluation_loop():
    while True:
        try:
            for site in SITES:
                await refresh_prediction_actuals_for_site(site["id"])
        except Exception as e:
            print("[PREDICTION EVAL LOOP ERROR]", repr(e))

        await asyncio.sleep(PREDICTION_EVAL_CHECK_SECONDS)


async def poll_daily_prediction_loop():
    startup_checked = False

    while True:
        try:
            if not PREDICTION_DAILY_REFRESH_ENABLED:
                await asyncio.sleep(PREDICTION_DAILY_CHECK_SECONDS)
                continue

            now = _now_madrid()
            today_key = now.date().isoformat()
            scheduled = now.replace(
                hour=PREDICTION_DAILY_REFRESH_HOUR,
                minute=PREDICTION_DAILY_REFRESH_MINUTE,
                second=0,
                microsecond=0,
            )

            if not startup_checked:
                startup_checked = True
                startup_limit = scheduled + timedelta(seconds=PREDICTION_DAILY_STARTUP_GRACE_SECONDS)

                if now > startup_limit:
                    daily_prediction_dates_run.add(today_key)
                    print(
                        "[PREDICTION DAILY] Saltando batch inicial tras arranque tardio "
                        f"({today_key}); se ejecutara en la proxima ventana programada."
                    )

            if now >= scheduled and today_key not in daily_prediction_dates_run:
                print(f"[PREDICTION DAILY] Guardando predicciones D+1 para {today_key}")

                for site in SITES:
                    await refresh_ia_for_site(site["id"], force=True)
                    await asyncio.sleep(0.5)

                daily_prediction_dates_run.add(today_key)

        except Exception as e:
            print("[PREDICTION DAILY LOOP ERROR]", repr(e))

        await asyncio.sleep(PREDICTION_DAILY_CHECK_SECONDS)


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
    asyncio.create_task(run_background_loop("PREDICTION EVAL LOOP", poll_prediction_evaluation_loop))
    asyncio.create_task(run_background_loop("PREDICTION DAILY LOOP", poll_daily_prediction_loop))
    #asyncio.create_task(poll_ia_loop())
    asyncio.create_task(run_background_loop("ALERTAS LOOP", poll_alertas_loop, startup_delay=0))


@app.on_event("shutdown")
async def on_shutdown():
    if IA_EXECUTOR is not None:
        IA_EXECUTOR.shutdown(wait=False, cancel_futures=True)

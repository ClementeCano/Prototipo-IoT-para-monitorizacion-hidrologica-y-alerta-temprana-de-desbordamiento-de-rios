import os
import time
import logging
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Any, List

import certifi
import requests
from requests.adapters import HTTPAdapter

import urllib3
from urllib3.exceptions import InsecureRequestWarning

try:
    from app.env_utils import env_bool, env_float, env_int, env_value
    from app.logging_config import configure_logging
except ImportError:
    from env_utils import env_bool, env_float, env_int, env_value
    from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# Opcional: en algunos Windows evita problemas de certificados
try:
    import truststore  # type: ignore
    truststore.inject_into_ssl()
except Exception:
    pass

URL = "https://www.saihebro.com/datos/apiopendata"
HISTORY_REQUEST_DELAY_SECONDS = env_float("SAIH_HISTORY_REQUEST_DELAY_SECONDS", 0.1)
SAIH_RATE_LIMIT_SLEEP_SECONDS = env_float("SAIH_RATE_LIMIT_SLEEP_SECONDS", 45)
SAIH_RATE_LIMIT_ABORT_THRESHOLD_SECONDS = env_float("SAIH_RATE_LIMIT_ABORT_THRESHOLD_SECONDS", 3600)
SAIH_SSL_MODE = env_value("SAIH_SSL_MODE", "auto").strip().lower()
SAIH_VERIFY_SSL = env_bool("SAIH_VERIFY_SSL", True)
SAIH_HTTP_RETRIES = max(0, env_int("SAIH_HTTP_RETRIES", 0))
_WARNED_INSECURE_SSL = False
_WORKING_VERIFY = None


def _build_session() -> requests.Session:
    session = requests.Session()

    retries = urllib3.Retry(
        total=SAIH_HTTP_RETRIES,
        connect=SAIH_HTTP_RETRIES,
        read=SAIH_HTTP_RETRIES,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retries,
        pool_connections=20,
        pool_maxsize=20,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    })

    # ✅ AQUÍ está la clave
    session.verify = certifi.where()

    return session


_SESSION = _build_session()


def _verify_candidates():
    if _WORKING_VERIFY is not None:
        return [_WORKING_VERIFY]

    if not SAIH_VERIFY_SSL or SAIH_SSL_MODE in {"0", "false", "no", "insecure", "disabled"}:
        return [False]

    if SAIH_SSL_MODE in {"strict", "certifi"}:
        return [certifi.where()]

    if SAIH_SSL_MODE == "system":
        return [True]

    return [certifi.where(), True, False]


def _bounded_timeout(timeout, remaining: float | None):
    if remaining is None:
        return timeout

    remaining = max(0.1, remaining)
    if isinstance(timeout, tuple):
        return (min(float(timeout[0]), remaining), min(float(timeout[1]), remaining))
    return min(float(timeout), remaining)


def _retry_after_seconds(response, fallback: float, max_sleep: float) -> tuple[float, float | None]:
    parsed_seconds = None
    try:
        header = response.headers.get("Retry-After") if response is not None else None
        if header:
            try:
                parsed_seconds = float(header)
            except ValueError:
                retry_at = parsedate_to_datetime(header)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                parsed_seconds = max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        pass

    if parsed_seconds is None:
        parsed_seconds = float(fallback)

    max_sleep = max(1.0, float(max_sleep))
    wait_seconds = min(max(1.0, parsed_seconds), max_sleep)
    return wait_seconds, parsed_seconds


def _rate_limit_message(server_wait_seconds: float | None, attempts: int | None = None) -> str:
    retry = (
        f"Retry-After={server_wait_seconds:.0f}s"
        if server_wait_seconds is not None
        else "sin Retry-After"
    )
    suffix = (
        f" Se agotaron los {attempts} intentos configurados."
        if attempts is not None and attempts > 0
        else ""
    )
    return (
        f"SAIH esta limitando las peticiones (429, {retry})."
        f"{suffix} Espera antes de reintentar o aumenta el intervalo entre peticiones."
    )


def _bounded_sleep(seconds: float, started: float, max_seconds: float | None) -> None:
    if max_seconds is not None:
        remaining = max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            return
        seconds = min(seconds, max(0.1, remaining))
    time.sleep(seconds)


def _safe_get(
    url: str,
    params: dict,
    timeout=(6, 20),
    attempts: int = 2,
    max_seconds: float | None = None,
    rate_limit_sleep_seconds: float | None = None,
    rate_limit_abort_threshold_seconds: float | None = None,
):
    global _WARNED_INSECURE_SSL, _WORKING_VERIFY
    last_error = None
    started = time.monotonic()

    for verify in _verify_candidates():
        if max_seconds is not None and (time.monotonic() - started) > max_seconds:
            raise TimeoutError(f"SAIH request supero el limite de {max_seconds:.1f}s")

        if verify is False and not _WARNED_INSECURE_SSL:
            urllib3.disable_warnings(InsecureRequestWarning)
            logger.warning("SAIH SSL: usando fallback sin verificacion SSL para saihebro.com")
            _WARNED_INSECURE_SSL = True

        for attempt in range(1, max(1, attempts) + 1):
            if max_seconds is not None and (time.monotonic() - started) > max_seconds:
                raise TimeoutError(f"SAIH request supero el limite de {max_seconds:.1f}s")

            remaining = None if max_seconds is None else max_seconds - (time.monotonic() - started)
            try:
                r = _SESSION.get(
                    url,
                    params=params,
                    timeout=_bounded_timeout(timeout, remaining),
                    verify=verify,
                )
                if r.status_code == 429:
                    max_wait = (
                        rate_limit_sleep_seconds
                        if rate_limit_sleep_seconds is not None
                        else SAIH_RATE_LIMIT_SLEEP_SECONDS
                    )
                    wait_seconds, server_wait_seconds = _retry_after_seconds(
                        r,
                        fallback=max_wait,
                        max_sleep=max_wait,
                    )
                    abort_threshold = (
                        rate_limit_abort_threshold_seconds
                        if rate_limit_abort_threshold_seconds is not None
                        else SAIH_RATE_LIMIT_ABORT_THRESHOLD_SECONDS
                    )
                    if server_wait_seconds is not None and server_wait_seconds >= abort_threshold:
                        raise RuntimeError(
                            "SAIH ha activado un limite temporal largo "
                            f"(Retry-After={server_wait_seconds:.0f}s). "
                            "No se espera automaticamente para evitar bloquear la actualizacion; "
                            "prueba mas tarde o actualiza menos municipios."
                        )
                    if attempt >= max(1, attempts):
                        raise RuntimeError(_rate_limit_message(server_wait_seconds, max(1, attempts)))
                    logger.warning(
                        "SAIH rate limit 429. Reintentando en %.1fs (SAIH pidio %.1fs, intento %s/%s)",
                        wait_seconds,
                        server_wait_seconds or wait_seconds,
                        attempt,
                        max(1, attempts),
                    )
                    _bounded_sleep(wait_seconds, started, max_seconds)
                    continue
                r.raise_for_status()
                _WORKING_VERIFY = verify
                return r.json()

            except requests.exceptions.SSLError as e:
                last_error = e
                break

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < max(1, attempts):
                    response = getattr(e, "response", None)
                    status_code = getattr(response, "status_code", None)
                    if status_code == 429:
                        max_wait = (
                            rate_limit_sleep_seconds
                            if rate_limit_sleep_seconds is not None
                            else SAIH_RATE_LIMIT_SLEEP_SECONDS
                        )
                        wait_seconds, server_wait_seconds = _retry_after_seconds(
                            response,
                            fallback=max_wait,
                            max_sleep=max_wait,
                        )
                        abort_threshold = (
                            rate_limit_abort_threshold_seconds
                            if rate_limit_abort_threshold_seconds is not None
                            else SAIH_RATE_LIMIT_ABORT_THRESHOLD_SECONDS
                        )
                        if server_wait_seconds is not None and server_wait_seconds >= abort_threshold:
                            raise RuntimeError(
                                "SAIH ha activado un limite temporal largo "
                                f"(Retry-After={server_wait_seconds:.0f}s). "
                                "No se espera automaticamente para evitar bloquear la actualizacion; "
                                "prueba mas tarde o actualiza menos municipios."
                            ) from e
                    else:
                        wait_seconds = attempt * 1.5
                    _bounded_sleep(wait_seconds, started, max_seconds)
                else:
                    break

    if isinstance(last_error, requests.exceptions.HTTPError):
        response = getattr(last_error, "response", None)
        if getattr(response, "status_code", None) == 429:
            max_wait = (
                rate_limit_sleep_seconds
                if rate_limit_sleep_seconds is not None
                else SAIH_RATE_LIMIT_SLEEP_SECONDS
            )
            _, server_wait_seconds = _retry_after_seconds(response, fallback=max_wait, max_sleep=max_wait)
            raise RuntimeError(_rate_limit_message(server_wait_seconds, max(1, attempts))) from last_error

    error_type = type(last_error).__name__ if last_error is not None else "desconocido"
    raise RuntimeError(f"Error conexion SAIH ({error_type}). Revisa conectividad, certificados o rate limit.")


def fetch_saih_signals(
    tags: List[str],
    request_timeout=None,
    request_attempts: int = 1,
    max_seconds: float | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    1 llamada para muchas señales.
    Devuelve:
      { TAG: {fecha, valor, tendencia, unidades, descripcion} }
    """

    apikey = env_value("SAIH_APIKEY", "")
    if not apikey:
        raise RuntimeError("Falta SAIH_APIKEY (en .env o variable de entorno).")

    tags = [t for t in tags if t]
    if not tags:
        return {}

    params = {
        "senal": ",".join(tags),
        "inicio": "",
        "apikey": apikey,
    }

    data = _safe_get(
        URL,
        params,
        timeout=request_timeout or (4, 10),
        attempts=request_attempts,
        max_seconds=max_seconds,
    )

    if not isinstance(data, list):
        raise RuntimeError(f"Formato SAIH inesperado: {type(data)}")

    out: Dict[str, Dict[str, Any]] = {}

    for item in data:
        tag = item.get("senal")
        if not tag:
            continue

        out[tag] = {
            "fecha": item.get("fecha"),
            "valor": item.get("valor"),
            "tendencia": item.get("tendencia"),
            "unidades": item.get("unidades"),
            "descripcion": item.get("descripcion"),
        }

    return out


def _iter_days(start_date: date, end_date: date):
    current = start_date

    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _extract_signal_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        items = data.get("senales", [])
    else:
        items = data

    if not isinstance(items, list):
        raise RuntimeError(f"Formato SAIH inesperado: {type(data)}")

    return [
        item
        for item in items
        if isinstance(item, dict) and item.get("senal")
    ]


def fetch_saih_history(
    tags: List[str],
    start_date: date,
    end_date: date,
    request_timeout=None,
    request_attempts: int = 2,
    max_seconds: float | None = None,
    request_delay_seconds: float | None = None,
    rate_limit_sleep_seconds: float | None = None,
    rate_limit_abort_threshold_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """
    Descarga datos historicos de SAIH por dias completos.
    La API de SAIH solo devuelve las 24 horas posteriores a cada fecha de inicio,
    por eso se itera dia a dia.
    """

    apikey = env_value("SAIH_APIKEY", "")
    if not apikey:
        raise RuntimeError("Falta SAIH_APIKEY (en .env o variable de entorno).")

    tags = [tag for tag in tags if tag]
    if not tags:
        return []

    if start_date > end_date:
        return []

    records: list[dict[str, Any]] = []
    started = time.monotonic()
    timeout = request_timeout or (8, 30)

    for day in _iter_days(start_date, end_date):
        if max_seconds is not None and (time.monotonic() - started) > max_seconds:
            raise TimeoutError(f"SAIH history supero el limite de {max_seconds:.1f}s")

        params = {
            "senal": ",".join(tags),
            "inicio": day.isoformat(),
            "apikey": apikey,
        }

        remaining = None if max_seconds is None else max_seconds - (time.monotonic() - started)
        data = _safe_get(
            URL,
            params,
            timeout=timeout,
            attempts=request_attempts,
            max_seconds=remaining,
            rate_limit_sleep_seconds=rate_limit_sleep_seconds,
            rate_limit_abort_threshold_seconds=rate_limit_abort_threshold_seconds,
        )
        records.extend(_extract_signal_items(data))

        delay = HISTORY_REQUEST_DELAY_SECONDS if request_delay_seconds is None else request_delay_seconds
        if delay > 0:
            time.sleep(delay)

    return records

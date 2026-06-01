import os
import time
from datetime import date, timedelta
from typing import Dict, Any, List

import certifi
import requests
from requests.adapters import HTTPAdapter

import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Opcional: en algunos Windows evita problemas de certificados
try:
    import truststore  # type: ignore
    truststore.inject_into_ssl()
except Exception:
    pass

URL = "https://www.saihebro.com/datos/apiopendata"
HISTORY_REQUEST_DELAY_SECONDS = float(os.getenv("SAIH_HISTORY_REQUEST_DELAY_SECONDS", "0.1"))
SAIH_SSL_MODE = os.getenv("SAIH_SSL_MODE", "auto").strip().lower()
SAIH_VERIFY_SSL = os.getenv("SAIH_VERIFY_SSL", "1").lower() not in {"0", "false", "no"}
SAIH_HTTP_RETRIES = max(0, int(os.getenv("SAIH_HTTP_RETRIES", "0")))
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


def _safe_get(url: str, params: dict, timeout=(6, 20), attempts: int = 2, max_seconds: float | None = None):
    global _WARNED_INSECURE_SSL, _WORKING_VERIFY
    last_error = None
    started = time.monotonic()

    for verify in _verify_candidates():
        if max_seconds is not None and (time.monotonic() - started) > max_seconds:
            raise TimeoutError(f"SAIH request supero el limite de {max_seconds:.1f}s")

        if verify is False and not _WARNED_INSECURE_SSL:
            urllib3.disable_warnings(InsecureRequestWarning)
            print("[SAIH SSL] Usando fallback sin verificacion SSL para saihebro.com")
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
                r.raise_for_status()
                _WORKING_VERIFY = verify
                return r.json()

            except requests.exceptions.SSLError as e:
                last_error = e
                break

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < 3:
                    time.sleep(attempt * 1.5)
                else:
                    break

    raise RuntimeError(f"Error conexion SAIH: {last_error}")


def fetch_saih_signals(tags: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    1 llamada para muchas señales.
    Devuelve:
      { TAG: {fecha, valor, tendencia, unidades, descripcion} }
    """

    apikey = os.getenv("SAIH_APIKEY", "")
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

    data = _safe_get(URL, params)

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
) -> list[dict[str, Any]]:
    """
    Descarga datos historicos de SAIH por dias completos.
    La API de SAIH solo devuelve las 24 horas posteriores a cada fecha de inicio,
    por eso se itera dia a dia.
    """

    apikey = os.getenv("SAIH_APIKEY", "")
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
        )
        records.extend(_extract_signal_items(data))

        if HISTORY_REQUEST_DELAY_SECONDS > 0:
            time.sleep(HISTORY_REQUEST_DELAY_SECONDS)

    return records

import os
import time
from datetime import date, timedelta
from typing import Dict, Any, List

import certifi
import requests
from requests.adapters import HTTPAdapter

import urllib3

# Opcional: en algunos Windows evita problemas de certificados
try:
    import truststore  # type: ignore
    truststore.inject_into_ssl()
except Exception:
    pass

URL = "https://www.saihebro.com/datos/apiopendata"
HISTORY_REQUEST_DELAY_SECONDS = float(os.getenv("SAIH_HISTORY_REQUEST_DELAY_SECONDS", "0.1"))


def _build_session() -> requests.Session:
    session = requests.Session()

    retries = urllib3.Retry(
        total=4,
        connect=4,
        read=4,
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


def _safe_get(url: str, params: dict, timeout=(6, 20)):
    last_error = None
    #print("🔥 SAIH CON SSL ACTIVADO")

    for attempt in range(1, 4):
        try:
            r = _SESSION.get(
                url,
                params=params,
                timeout=timeout,
                verify=False,  # DESACTIVADO para evitar errores SSL en algunos entornos (usar con precaución)
            )
            r.raise_for_status()
            return r.json()

        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < 3:
                time.sleep(attempt * 1.5)
            else:
                break

    raise RuntimeError(f"❌ Error conexión SAIH: {last_error}")


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


def fetch_saih_history(tags: List[str], start_date: date, end_date: date) -> list[dict[str, Any]]:
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

    for day in _iter_days(start_date, end_date):
        params = {
            "senal": ",".join(tags),
            "inicio": day.isoformat(),
            "apikey": apikey,
        }

        data = _safe_get(URL, params, timeout=(8, 30))
        records.extend(_extract_signal_items(data))

        if HISTORY_REQUEST_DELAY_SECONDS > 0:
            time.sleep(HISTORY_REQUEST_DELAY_SECONDS)

    return records

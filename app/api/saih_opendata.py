import os
import time
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

    return session


_SESSION = _build_session()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _safe_get(url: str, params: dict, timeout=(6, 20)):
    last_error = None

    for attempt in range(1, 4):
        try:
            r = _SESSION.get(
                url,
                params=params,
                timeout=timeout,
                verify=False,  # 🔥 aquí
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
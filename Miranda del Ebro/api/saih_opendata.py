import os
import requests
from typing import Dict, Any, List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Opcional: en algunos Windows evita problemas de certificados
try:
    import truststore  # type: ignore
    truststore.inject_into_ssl()
except Exception:
    pass

URL = "https://www.saihebro.com/datos/apiopendata"

# Session reutilizable (mejor rendimiento + retries)
_session = requests.Session()
_retry = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=0.6,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET",),
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=20, pool_maxsize=20)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def fetch_saih_signals(tags: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    1 llamada para muchas señales. Devuelve:
      { TAG: {fecha, valor, tendencia, unidades, descripcion} }
    """
    apikey = os.getenv("SAIH_APIKEY", "")
    if not apikey:
        raise RuntimeError("Falta SAIH_APIKEY (en .env o variable de entorno).")

    tags = [t for t in tags if t]
    if not tags:
        return {}

    # timeout separado: (connect, read)
    r = _session.get(
        URL,
        params={"senal": ",".join(tags), "inicio": "", "apikey": apikey},
        timeout=(4, 12),
        headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()

    data = r.json()  # lista [{senal, fecha, valor, ...}]
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
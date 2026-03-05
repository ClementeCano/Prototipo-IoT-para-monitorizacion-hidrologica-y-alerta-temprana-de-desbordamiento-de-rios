import os
import requests
from typing import Dict, Any, List

# Opcional: en algunos Windows evita problemas de certificados
try:
    import truststore  # type: ignore
    truststore.inject_into_ssl()
except Exception:
    pass

URL = "https://www.saihebro.com/datos/apiopendata"


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

    r = requests.get(
        URL,
        params={"senal": ",".join(tags), "inicio": "", "apikey": apikey},
        timeout=30,
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
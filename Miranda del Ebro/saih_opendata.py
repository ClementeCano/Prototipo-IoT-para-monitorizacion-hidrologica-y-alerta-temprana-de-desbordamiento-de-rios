import truststore
truststore.inject_into_ssl()

import os
import requests
from typing import Dict, Any, List, Optional

URL = "https://www.saihebro.com/datos/apiopendata"
TAG_NIVEL = "A001L17NRIO1"
TAG_CAUDAL = "A001L65QRIO1"

def fetch_saih_latest() -> Dict[str, Any]:
    apikey = os.getenv("SAIH_APIKEY", "")
    if not apikey:
        raise RuntimeError("Falta SAIH_APIKEY (ponlo como variable de entorno).")

    senales = f"{TAG_NIVEL},{TAG_CAUDAL}"
    r = requests.get(URL, params={"senal": senales, "inicio": "", "apikey": apikey}, timeout=30)
    r.raise_for_status()

    data: List[Dict[str, Any]] = r.json()  # [{senal, fecha, valor, ...}, ...]

    out = {
        "ts": None,
        "nivel_m": None,
        "caudal_m3s": None,
        "tendencia_nivel": None,
        "tendencia_caudal": None,
    }

    for item in data:
        if item.get("senal") == TAG_NIVEL:
            out["nivel_m"] = float(item.get("valor")) if item.get("valor") is not None else None
            out["ts"] = item.get("fecha") or out["ts"]
            out["tendencia_nivel"] = item.get("tendencia")
        elif item.get("senal") == TAG_CAUDAL:
            out["caudal_m3s"] = float(item.get("valor")) if item.get("valor") is not None else None
            # si no hay ts aún, usa el del caudal
            out["ts"] = out["ts"] or item.get("fecha")
            out["tendencia_caudal"] = item.get("tendencia")

    return out
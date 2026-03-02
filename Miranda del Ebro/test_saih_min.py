import truststore
truststore.inject_into_ssl()

import json
import requests

META_URL = "https://www.saihebro.com/api/grafica/getMetaDatosSenalesEstacion"
DATA_URL = "https://www.saihebro.com/api/datos-graficas/obtenerGraficaHistorica"

TAG_NIVEL = "A001L17NRIO1"
TAG_CAUDAL = "A001L65QRIO1"

def session_min():
    s = requests.Session()
    # Headers mínimos “tipo navegador”
    s.headers.update({
        "Accept": "*/*",
        "Content-Type": "application/json",
        "Origin": "https://www.saihebro.com",
        "Referer": "https://www.saihebro.com/",
        "X-Requested-With": "XMLHttpRequest",
        "User-Agent": "Mozilla/5.0"
    })
    return s

def fetch_meta(s, tag, days=7):
    r = s.get(META_URL, params={"tag": tag, "cambio_periodo": days}, timeout=30)
    print("META status:", r.status_code)
    if r.status_code != 200:
        print(r.text[:800])
        raise SystemExit
    return r.json()

def fetch_series(s, tag, meta):
    meta_key = f"{tag}|VALOR"
    md = meta.get("metaData", {})
    if meta_key not in md:
        print("No está", meta_key, "en metaData. Keys:", list(md.keys())[:10])
        raise SystemExit

    payload = {
        "fechaIni": meta["fechaIni"],
        "fechaFin": meta["fechaFin"],
        "metaData": {meta_key: md[meta_key]},
        "senalesSeleccionadas": tag,
        "tipoConsolidado": meta.get("tipoConsolidado", "quinceminutal"),
    }

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    r = s.post(DATA_URL, data=body, timeout=30)

    print("DATA status:", r.status_code, "| content-type:", r.headers.get("content-type"))
    if r.status_code != 200:
        print("TEXT:", r.text[:800])
        raise SystemExit

    data = r.json()
    print("JSON type:", type(data))
    if isinstance(data, dict):
        print("keys:", list(data.keys())[:30])
    print("sample:", json.dumps(data, ensure_ascii=False)[:1200])

if __name__ == "__main__":
    s = session_min()
    meta = fetch_meta(s, TAG_NIVEL, days=7)

    print("\n--- CAUDAL ---")
    fetch_series(s, TAG_CAUDAL, meta)

    print("\n--- NIVEL ---")
    fetch_series(s, TAG_NIVEL, meta)
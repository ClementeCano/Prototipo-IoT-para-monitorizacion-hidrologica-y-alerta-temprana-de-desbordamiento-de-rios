import truststore
import json
truststore.inject_into_ssl()

import requests

APIKEY = "cd19780c00ce7c7f62bb6696db85edd0"
URL = "https://www.saihebro.com/datos/apiopendata"
SENALES = "A001L17NRIO1,A001L65QRIO1"

r = requests.get(URL, params={"senal": SENALES, "inicio": "", "apikey": APIKEY}, timeout=30)
print("status:", r.status_code)
print("content-type:", r.headers.get("content-type"))
print("text sample:", r.text[:800])
print(json.dumps(r.json(), ensure_ascii=False, indent=2)[:2000])
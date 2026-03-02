# test_saih.py
import truststore
truststore.inject_into_ssl()

import json
import requests
from saih_auth import parse_windows_curl

url, headers, cookies, payload = parse_windows_curl("saih_curl.txt")

# Limpia headers que no interesa forzar
headers.pop("Content-Length", None)
headers.pop("Host", None)

s = requests.Session()
s.headers.update(headers)
s.cookies.update(cookies)

r = s.post(url, json=payload, timeout=30)
print("status:", r.status_code)
print("content-type:", r.headers.get("content-type"))

# Si no es 200, imprime texto (suele ser HTML/401)
if r.status_code != 200:
    print("text sample:", r.text[:800])
else:
    data = r.json()
    if isinstance(data, dict):
        print("keys:", list(data.keys())[:30])
    print("sample:", json.dumps(data, ensure_ascii=False)[:1200])

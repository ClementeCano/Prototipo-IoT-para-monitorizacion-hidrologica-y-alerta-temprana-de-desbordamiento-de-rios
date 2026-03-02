# saih_auth.py
import re, json
from pathlib import Path

def parse_windows_curl(path="saih_curl.txt"):
    txt = Path(path).read_text(encoding="utf-8")

    # Quita escapes de cmd: ^" -> ", ^{ -> {, ^| -> |, etc.
    txt = txt.replace("\r", "")
    txt = txt.replace("^\n", " ")
    txt = txt.replace("^", "")

    # URL
    m = re.search(r'curl\s+"([^"]+)"', txt)
    if not m:
        raise RuntimeError("No pude encontrar la URL dentro del cURL. Asegúrate de pegar el comando completo.")
    url = m.group(1)

    # Headers (-H "K: V")
    headers = {}
    for h in re.findall(r'-H\s+"([^"]+)"', txt):
        if ":" in h:
            k, v = h.split(":", 1)
            headers[k.strip()] = v.strip()

    # Cookies (-b "a=1; b=2; ...")
    cookies = {}
    m = re.search(r'-b\s+"([^"]*)"', txt)
    if m:
        cookie_str = m.group(1)
        for chunk in cookie_str.split(";"):
            chunk = chunk.strip()
            if "=" in chunk:
                k, v = chunk.split("=", 1)
                cookies[k.strip()] = v.strip()

    # Payload (--data-raw "{...}")
    payload = None
    m = re.search(r'--data-raw\s+"([^"]+)"', txt)
    if m:
        data_str = m.group(1)
        try:
            payload = json.loads(data_str)
        except json.JSONDecodeError:
            payload = data_str  # por si viniera raro

    return url, headers, cookies, payload

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


# =========================================================
# CARGA .ENV
# =========================================================
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

API_KEY = os.getenv("AEMET_APIKEY", "").strip()

if not API_KEY:
    raise RuntimeError("No se ha encontrado AEMET_APIKEY en el .env")


# =========================================================
# CONFIG
# =========================================================
BASE_URL = "https://opendata.aemet.es/opendata"

HEADERS = {
    "accept": "application/json",
    "api_key": API_KEY,
}

OUTPUT_DIR = Path("datos_lluvia_excel")
OUTPUT_DIR.mkdir(exist_ok=True)

# Control de peticiones
MIN_SECONDS_BETWEEN_REQUESTS = 2.2
TIMEOUT = 30
MAX_RETRIES = 6

_last_request_time = 0.0


# =========================================================
# SITIOS (TUS INDICATIVOS)
# =========================================================
SITIOS = [
    {"nombre": "Arroyo", "indicativo": "9012E"},
    {"nombre": "Ascó", "indicativo": "9961X"},
    {"nombre": "Castejón", "indicativo": "9293X"},
    {"nombre": "Gelsa", "indicativo": "9510X"},
    {"nombre": "Logroño", "indicativo": "9170"},
    {"nombre": "Mendavia", "indicativo": "9170"},
    {"nombre": "Miranda del Ebro", "indicativo": "9069C"},
    {"nombre": "Palazuelos", "indicativo": "2465"},
    {"nombre": "Reinosa-Nestares", "indicativo": "9001D"},
    {"nombre": "Riosequillo", "indicativo": "9051"},
    {"nombre": "Tortosa", "indicativo": "9981A"},
    {"nombre": "Tudela", "indicativo": "9302Y"},
    {"nombre": "Villafranca de Ebro", "indicativo": "9510X"},
    {"nombre": "Zaragoza", "indicativo": "9434P"},
]


# =========================================================
# INTERVALOS (6 MESES)
# =========================================================
def generar_intervalos():
    return [
        ("2020-01-01T00:00:00UTC", "2020-06-30T23:59:59UTC"),
        ("2020-07-01T00:00:00UTC", "2020-12-31T23:59:59UTC"),
        ("2021-01-01T00:00:00UTC", "2021-06-30T23:59:59UTC"),
        ("2021-07-01T00:00:00UTC", "2021-12-31T23:59:59UTC"),
        ("2022-01-01T00:00:00UTC", "2022-06-30T23:59:59UTC"),
        ("2022-07-01T00:00:00UTC", "2022-12-31T23:59:59UTC"),
        ("2023-01-01T00:00:00UTC", "2023-06-30T23:59:59UTC"),
        ("2023-07-01T00:00:00UTC", "2023-12-31T23:59:59UTC"),
        ("2024-01-01T00:00:00UTC", "2024-06-30T23:59:59UTC"),
        ("2024-07-01T00:00:00UTC", "2024-12-31T23:59:59UTC"),
        ("2025-01-01T00:00:00UTC", "2025-06-30T23:59:59UTC"),
        ("2025-07-01T00:00:00UTC", "2025-12-31T23:59:59UTC"),
        ("2026-01-01T00:00:00UTC", "2026-01-31T23:59:59UTC"),
    ]


# =========================================================
# RATE LIMIT
# =========================================================
def esperar():
    global _last_request_time
    ahora = time.time()
    diff = ahora - _last_request_time

    if diff < MIN_SECONDS_BETWEEN_REQUESTS:
        time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - diff)

    _last_request_time = time.time()


# =========================================================
# REQUEST ROBUSTA
# =========================================================
def get_json(url, headers=None):
    ultimo_error = None

    for intento in range(1, MAX_RETRIES + 1):
        try:
            esperar()
            r = requests.get(url, headers=headers, timeout=TIMEOUT)

            if r.status_code == 429:
                wait = min(20 * intento, 120)
                print(f"   ⏳ 429 → esperando {wait}s")
                time.sleep(wait)
                continue

            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")

            return r.json()

        except Exception as e:
            ultimo_error = e
            time.sleep(2 * intento)

    raise RuntimeError(f"Error request: {ultimo_error}")


def descargar_tramo(indicativo, ini, fin):
    url = f"{BASE_URL}/api/valores/climatologicos/diarios/datos/fechaini/{ini}/fechafin/{fin}/estacion/{indicativo}"

    meta = get_json(url, HEADERS)
    return get_json(meta["datos"])


# =========================================================
# PRECIPITACIÓN
# =========================================================
def parsear_prec(x):
    if x is None:
        return None

    s = str(x).strip()

    if not s:
        return None

    if s.lower() == "ip":
        return 0.0

    s = s.replace(",", ".")

    try:
        return float(s)
    except:
        return None


# =========================================================
# DESCARGA COMPLETA
# =========================================================
def descargar_historico(indicativo):
    filas = []

    for ini, fin in generar_intervalos():
        print(f"   ⏳ {ini[:10]} -> {fin[:10]}")

        try:
            datos = descargar_tramo(indicativo, ini, fin)
        except:
            continue

        for d in datos:
            if "fecha" in d:
                filas.append({
                    "fecha": d["fecha"],
                    "lluvia": parsear_prec(d.get("prec"))
                })

    df = pd.DataFrame(filas)

    if not df.empty:
        df["fecha"] = pd.to_datetime(df["fecha"])
    else:
        df = pd.DataFrame(columns=["fecha", "lluvia"])

    # =====================================================
    # GENERAR TODAS LAS FECHAS
    # =====================================================
    fechas = pd.date_range("2020-01-01", "2026-01-31", freq="D")
    df_full = pd.DataFrame({"fecha": fechas})

    df = df_full.merge(df, on="fecha", how="left")

    # =====================================================
    # FORMATO FINAL
    # =====================================================
    df = df.sort_values("fecha")

    # fecha DD/MM/AAAA
    df["fecha"] = df["fecha"].dt.strftime("%d/%m/%Y")

    return df


# =========================================================
# MAIN
# =========================================================
def main():
    print("🌧️ Descargando lluvia...")

    cache = {}

    for sitio in SITIOS:
        nombre = sitio["nombre"]
        ind = sitio["indicativo"]

        print(f"\n📍 {nombre} ({ind})")

        if ind in cache:
            df = cache[ind].copy()
            print("   ⚡ usando cache")
        else:
            df = descargar_historico(ind)
            cache[ind] = df.copy()

        df.to_excel(OUTPUT_DIR / f"{nombre}.xlsx", index=False)

        print("   ✅ guardado")


if __name__ == "__main__":
    main()
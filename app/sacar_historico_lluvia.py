from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("AEMET_APIKEY", "").strip()
if not API_KEY:
    raise RuntimeError("Falta AEMET_APIKEY en el archivo .env")

BASE_URL = "https://opendata.aemet.es/opendata"
HEADERS = {
    "accept": "application/json",
    "api_key": API_KEY,
}

OUTPUT_DIR = BASE_DIR / "datos_lluvia_aemet"
OUTPUT_DIR.mkdir(exist_ok=True)

FECHA_INICIO = "2020-01-01"
FECHA_FIN = "2026-04-26"

TIMEOUT = 30
MAX_RETRIES = 6
MIN_SECONDS_BETWEEN_REQUESTS = 2.2
_last_request_time = 0.0

# =========================================================
# MUNICIPIOS Y ESTACIONES
# =========================================================
SITIOS = [
    {"municipio": "Arroyo", "indicativo": "9012E"},
    {"municipio": "Ascó", "indicativo": "9961X"},
    {"municipio": "Castejón", "indicativo": "9293X"},
    {"municipio": "Gelsa", "indicativo": "9510X"},
    {"municipio": "Logroño", "indicativo": "9170"},
    {"municipio": "Mendavia", "indicativo": "9170"},
    {"municipio": "Miranda del Ebro", "indicativo": "9069C"},
    {"municipio": "Palazuelos", "indicativo": "2465"},
    {"municipio": "Reinosa-Nestares", "indicativo": "9001D"},
    {"municipio": "Riosequillo", "indicativo": "9051"},
    {"municipio": "Tortosa", "indicativo": "9981A"},
    {"municipio": "Tudela", "indicativo": "9302Y"},
    {"municipio": "Villafranca de Ebro", "indicativo": "9510X"},
    {"municipio": "Zaragoza", "indicativo": "9434P"},
]

# =========================================================
# UTILIDADES
# =========================================================
def generar_intervalos_semestrales():
    intervalos = []
    inicio = pd.Timestamp(FECHA_INICIO)
    fin_global = pd.Timestamp(FECHA_FIN)

    actual = inicio
    while actual <= fin_global:
        fin = min(actual + pd.DateOffset(months=6) - pd.DateOffset(days=1), fin_global)
        ini_txt = actual.strftime("%Y-%m-%dT00:00:00UTC")
        fin_txt = fin.strftime("%Y-%m-%dT23:59:59UTC")
        intervalos.append((ini_txt, fin_txt))
        actual = fin + pd.DateOffset(days=1)

    return intervalos


def esperar_rate_limit():
    global _last_request_time
    ahora = time.time()
    diff = ahora - _last_request_time
    if diff < MIN_SECONDS_BETWEEN_REQUESTS:
        time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - diff)
    _last_request_time = time.time()


def get_json(url: str, headers: dict | None = None):
    ultimo_error = None

    for intento in range(1, MAX_RETRIES + 1):
        try:
            esperar_rate_limit()
            r = requests.get(url, headers=headers, timeout=TIMEOUT)

            if r.status_code == 429:
                espera = min(20 * intento, 120)
                print(f"   429 recibido. Esperando {espera}s...")
                time.sleep(espera)
                continue

            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")

            return r.json()

        except Exception as e:
            ultimo_error = e
            time.sleep(2 * intento)

    raise RuntimeError(f"No se pudo obtener JSON: {ultimo_error}")


def descargar_tramo(indicativo: str, fechaini: str, fechafin: str):
    url = (
        f"{BASE_URL}/api/valores/climatologicos/diarios/datos/"
        f"fechaini/{fechaini}/fechafin/{fechafin}/estacion/{indicativo}"
    )
    meta = get_json(url, HEADERS)

    if "datos" not in meta:
        raise RuntimeError(f"Respuesta AEMET inválida: {meta}")

    return get_json(meta["datos"])


def parsear_precipitacion(valor):
    if valor is None:
        return None

    s = str(valor).strip()
    if not s:
        return None

    if s.lower() == "ip":
        return 0.0

    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def descargar_historico_lluvia(indicativo: str) -> pd.DataFrame:
    filas = []

    for ini, fin in generar_intervalos_semestrales():
        print(f"   Descargando {ini[:10]} -> {fin[:10]}")
        try:
            datos = descargar_tramo(indicativo, ini, fin)
        except Exception as e:
            print(f"   Aviso: no se pudo descargar el tramo ({e})")
            continue

        for d in datos:
            fecha = d.get("fecha")
            if fecha:
                filas.append(
                    {
                        "fecha": pd.to_datetime(fecha, errors="coerce"),
                        "lluvia_mm": parsear_precipitacion(d.get("prec")),
                    }
                )

    df = pd.DataFrame(filas, columns=["fecha", "lluvia_mm"])

    if not df.empty:
        df = df.dropna(subset=["fecha"]).drop_duplicates(subset=["fecha"]).sort_values("fecha")

    fechas_completas = pd.DataFrame(
        {"fecha": pd.date_range(FECHA_INICIO, FECHA_FIN, freq="D")}
    )

    df_final = fechas_completas.merge(df, on="fecha", how="left")
    df_final["fecha"] = df_final["fecha"].dt.strftime("%d/%m/%Y")

    return df_final


def main():
    cache_estacion = {}

    print("Descargando lluvia histórica limpia desde AEMET...")

    for sitio in SITIOS:
        municipio = sitio["municipio"]
        indicativo = sitio["indicativo"]

        print(f"\nMunicipio: {municipio} | Estación: {indicativo}")

        if indicativo in cache_estacion:
            df = cache_estacion[indicativo].copy()
            print("   Usando caché de esa estación")
        else:
            df = descargar_historico_lluvia(indicativo)
            cache_estacion[indicativo] = df.copy()

        salida = OUTPUT_DIR / f"{municipio}.xlsx"
        df.to_excel(salida, index=False)
        print(f"   Guardado en: {salida}")


if __name__ == "__main__":
    main()
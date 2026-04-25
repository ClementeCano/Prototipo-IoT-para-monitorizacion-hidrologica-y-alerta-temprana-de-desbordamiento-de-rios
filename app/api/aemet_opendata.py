import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import certifi
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Opcional: en algunos Windows evita problemas de certificados
try:
    import truststore  # type: ignore
    truststore.inject_into_ssl()
except Exception:
    pass

BASE = "https://opendata.aemet.es/opendata/api"
TZ = ZoneInfo("Europe/Madrid")


def _build_session() -> requests.Session:
    session = requests.Session()

    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        }
    )
    return session


_SESSION = _build_session()


def _get_json(url: str, timeout: int = 60, pause_before_retry: float = 1.0):
    last_error = None

    for attempt in range(1, 4):
        try:
            r = _SESSION.get(
                url,
                timeout=timeout,
                verify=certifi.where(),
            )
            r.raise_for_status()
            return r.json()

        except requests.exceptions.SSLError as e:
            last_error = RuntimeError(f"Error SSL al conectar con {url}: {e}")
            break

        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < 3:
                time.sleep(pause_before_retry * attempt)
            else:
                break

    raise RuntimeError(f"Fallo al pedir JSON a {url}: {last_error}")


def fetch_aemet_municipio_horaria(municipio: str) -> List[Dict[str, Any]]:
    api_key = os.getenv("AEMET_APIKEY", "")
    if not api_key:
        raise RuntimeError("Falta AEMET_APIKEY (en .env o variable de entorno).")

    url = f"{BASE}/prediccion/especifica/municipio/horaria/{municipio}?api_key={api_key}"
    meta = _get_json(url, timeout=60)

    datos_url = meta.get("datos")
    if not datos_url:
        raise RuntimeError(f"Respuesta AEMET sin 'datos': {meta}")

    data = _get_json(datos_url, timeout=60)
    if not isinstance(data, list):
        raise RuntimeError(f"Formato AEMET inesperado (no list): {type(data)}")

    return data


def _to_float_mm(x) -> float:
    if x is None:
        return 0.0
    s = str(x).strip()
    if s == "" or s.lower() == "null":
        return 0.0
    if s.lower() == "ip":
        return 0.0
    try:
        return float(s.replace(",", "."))
    except Exception:
        return 0.0


def _parse_periodo_to_interval(fecha_iso: str, periodo: str) -> Optional[Tuple[datetime, datetime]]:
    """
    Soporta:
      - "03" -> 03:00-03:59
      - "00-06" -> 00:00-05:59
      - "1319" -> 13:00-18:59
      - "1901" -> 19:00-00:59 (cruza medianoche)
    """
    try:
        base_date = datetime.fromisoformat(fecha_iso).replace(
            tzinfo=TZ, hour=0, minute=0, second=0, microsecond=0
        )
    except Exception:
        return None

    periodo = str(periodo).strip()

    if len(periodo) == 2 and periodo.isdigit():
        h = int(periodo)
        start = base_date.replace(hour=h)
        end = start + timedelta(hours=1) - timedelta(seconds=1)
        return start, end

    if "-" in periodo:
        a, b = periodo.split("-", 1)
        a = a.strip()
        b = b.strip()
        if a.isdigit() and b.isdigit() and len(a) == 2 and len(b) == 2:
            h1 = int(a)
            h2 = int(b)
            start = base_date.replace(hour=h1)
            end = base_date.replace(hour=h2) - timedelta(seconds=1)
            if end < start:
                end = (base_date + timedelta(days=1)).replace(hour=h2) - timedelta(seconds=1)
            return start, end

    if len(periodo) == 4 and periodo.isdigit():
        h1 = int(periodo[:2])
        h2 = int(periodo[2:])
        start = base_date.replace(hour=h1)
        end = base_date.replace(hour=h2) - timedelta(seconds=1)
        if end < start:
            end = (base_date + timedelta(days=1)).replace(hour=h2) - timedelta(seconds=1)
        return start, end

    return None


def extract_rain_forecast_mm(
    aemet_data: List[Dict[str, Any]],
    hours_ahead: int = 24,
    list_hours: int = 12,
) -> Dict[str, Any]:
    now = datetime.now(TZ)
    limit_6h = now + timedelta(hours=6)
    limit_24h = now + timedelta(hours=hours_ahead)
    limit_list = now + timedelta(hours=list_hours)

    item = aemet_data[0] if (aemet_data and isinstance(aemet_data[0], dict)) else None
    if not item:
        return {
            "aemet_mm_6h_sum": 0.0,
            "aemet_mm_24h_sum": 0.0,
            "aemet_mm_6h_max": 0.0,
            "aemet_mm_24h_max": 0.0,
            "aemet_mm_next_hours": [],
        }

    pred = item.get("prediccion", {})
    dias = pred.get("dia", [])
    if not dias:
        return {
            "aemet_mm_6h_sum": 0.0,
            "aemet_mm_24h_sum": 0.0,
            "aemet_mm_6h_max": 0.0,
            "aemet_mm_24h_max": 0.0,
            "aemet_mm_next_hours": [],
        }

    series: List[Tuple[datetime, float]] = []
    for d in dias:
        fecha = d.get("fecha")
        prec = d.get("precipitacion", [])
        if not fecha or not isinstance(prec, list):
            continue

        for p in prec:
            if not isinstance(p, dict):
                continue
            periodo = str(p.get("periodo", "")).strip()
            if len(periodo) == 2 and periodo.isdigit():
                hour = int(periodo)
                dt = datetime.fromisoformat(fecha).replace(
                    tzinfo=TZ,
                    hour=hour,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                if dt >= now:
                    series.append((dt, _to_float_mm(p.get("value"))))

    series.sort(key=lambda x: x[0])

    mm_6 = [mm for (dt, mm) in series if dt <= limit_6h]
    mm_24 = [mm for (dt, mm) in series if dt <= limit_24h]
    mm_list = [
        {"hora": dt.strftime("%Y-%m-%d %H:%M"), "mm": round(mm, 2)}
        for (dt, mm) in series
        if dt <= limit_list
    ]

    return {
        "aemet_mm_6h_sum": round(sum(mm_6), 2) if mm_6 else 0.0,
        "aemet_mm_24h_sum": round(sum(mm_24), 2) if mm_24 else 0.0,
        "aemet_mm_6h_max": round(max(mm_6), 2) if mm_6 else 0.0,
        "aemet_mm_24h_max": round(max(mm_24), 2) if mm_24 else 0.0,
        "aemet_mm_next_hours": mm_list,
    }


def extract_prob_precip_summary(
    aemet_data: List[Dict[str, Any]],
    hours_ahead: int = 24,
) -> Dict[str, Optional[int]]:
    now = datetime.now(TZ)
    limit_6h = now + timedelta(hours=6)
    limit_24h = now + timedelta(hours=hours_ahead)

    item = aemet_data[0] if (aemet_data and isinstance(aemet_data[0], dict)) else None
    if not item:
        return {"aemet_prob_6h_max": None, "aemet_prob_24h_max": None}

    pred = item.get("prediccion", {})
    dias = pred.get("dia", [])
    if not dias:
        return {"aemet_prob_6h_max": None, "aemet_prob_24h_max": None}

    probs_6: List[int] = []
    probs_24: List[int] = []

    for d in dias:
        fecha = d.get("fecha")
        pp = d.get("probPrecipitacion", [])
        if not fecha or not isinstance(pp, list):
            continue

        for p in pp:
            if not isinstance(p, dict):
                continue
            try:
                prob = int(p.get("value"))
            except Exception:
                continue

            interval = _parse_periodo_to_interval(fecha, str(p.get("periodo")))
            if not interval:
                continue

            start, end = interval

            if end >= now and start <= limit_6h:
                probs_6.append(prob)
            if end >= now and start <= limit_24h:
                probs_24.append(prob)

    return {
        "aemet_prob_6h_max": max(probs_6) if probs_6 else None,
        "aemet_prob_24h_max": max(probs_24) if probs_24 else None,
    }
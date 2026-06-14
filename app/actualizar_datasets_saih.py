import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from app.api.saih_opendata import fetch_saih_history
    from app.core.config import SITES
    from app.model_pipeline import BASE_COLUMNS, add_model_features
except ImportError:
    from api.saih_opendata import fetch_saih_history
    from core.config import SITES
    from model_pipeline import BASE_COLUMNS, add_model_features


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets_modelo_municipios"


def _site_map():
    return {site["id"]: site for site in SITES}


def _last_date(df: pd.DataFrame):
    if df.empty or "fecha" not in df.columns:
        return None

    dates = pd.to_datetime(df["fecha"], errors="coerce").dropna()
    if dates.empty:
        return None

    return dates.max().date()


def _records_to_daily(site: dict, records: list[dict]) -> pd.DataFrame:
    saih = site.get("saih") or {}
    signal_to_column = {
        saih.get("nivel"): "nivel_m",
        saih.get("caudal"): "caudal_m3s",
    }
    rows_by_ts = {}

    for record in records:
        column = signal_to_column.get(record.get("senal"))
        timestamp = record.get("fecha")
        if not column or not timestamp:
            continue

        row = rows_by_ts.setdefault(timestamp, {"fecha": timestamp})
        row[column] = record.get("valor")

    df = pd.DataFrame(rows_by_ts.values())
    if df.empty:
        return df

    df["fecha_dt"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha_dt"]).sort_values("fecha_dt")

    for column in ["nivel_m", "caudal_m3s"]:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    daily = (
        df.set_index("fecha_dt")[["nivel_m", "caudal_m3s"]]
        .resample("D")
        .mean()
        .dropna(how="any")
        .reset_index()
    )

    if daily.empty:
        return daily

    daily["fecha"] = daily["fecha_dt"].dt.strftime("%Y-%m-%d")
    daily["lluvia_mm"] = 0.0
    daily["desbordamiento"] = 0
    return daily[BASE_COLUMNS]


def update_site(
    site: dict,
    days_back: int,
    max_seconds: float,
    request_attempts: int,
    request_delay: float,
    rate_limit_sleep: float,
    rate_limit_abort_threshold: float,
) -> int:
    site_id = site["id"]
    path = DATASET_DIR / f"{site_id}.csv"
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=BASE_COLUMNS)

    end_date = date.today() - timedelta(days=1)
    last = _last_date(existing)
    start_date = (last + timedelta(days=1)) if last else (end_date - timedelta(days=days_back - 1))

    if start_date > end_date:
        print(f"{site_id}: ya estaba actualizado hasta {last}")
        return 0

    saih = site.get("saih") or {}
    tags = [tag for tag in [saih.get("nivel"), saih.get("caudal")] if tag]
    if len(tags) < 2:
        print(f"{site_id}: sin senales SAIH suficientes")
        return 0

    print(f"{site_id}: descargando {start_date} -> {end_date}")
    records = fetch_saih_history(
        tags,
        start_date,
        end_date,
        request_timeout=(3, 10),
        request_attempts=request_attempts,
        max_seconds=max_seconds,
        request_delay_seconds=request_delay,
        rate_limit_sleep_seconds=rate_limit_sleep,
        rate_limit_abort_threshold_seconds=rate_limit_abort_threshold,
    )
    daily = _records_to_daily(site, records)

    if daily.empty:
        print(f"{site_id}: SAIH no devolvio datos diarios completos")
        return 0

    base_existing = existing[[column for column in BASE_COLUMNS if column in existing.columns]].copy()
    for column in BASE_COLUMNS:
        if column not in base_existing.columns:
            base_existing[column] = 0

    combined = pd.concat([base_existing[BASE_COLUMNS], daily], ignore_index=True)
    combined = (
        combined.drop_duplicates(subset=["fecha"], keep="last")
        .sort_values("fecha")
        .reset_index(drop=True)
    )
    updated = add_model_features(combined)
    DATASET_DIR.mkdir(exist_ok=True)
    updated.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"{site_id}: {len(daily)} dias nuevos, dataset final {len(updated)} filas")
    return len(daily)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Actualiza datasets locales con historico SAIH reciente.")
    parser.add_argument("--site", action="append", help="ID de municipio. Repetible. Por defecto: todos.")
    parser.add_argument("--days", type=int, default=45, help="Dias hacia atras si el CSV esta vacio.")
    parser.add_argument("--max-seconds", type=float, default=900, help="Tiempo maximo por municipio.")
    parser.add_argument("--request-attempts", type=int, default=4, help="Reintentos por dia si SAIH limita o falla.")
    parser.add_argument("--request-delay", type=float, default=2.0, help="Pausa entre dias para evitar 429.")
    parser.add_argument("--rate-limit-sleep", type=float, default=45.0, help="Espera tras un 429 antes de reintentar.")
    parser.add_argument(
        "--rate-limit-abort-threshold",
        type=float,
        default=3600.0,
        help="Si SAIH pide esperar mas segundos que este umbral, no se bloquea el script.",
    )
    args = parser.parse_args()

    sites_by_id = _site_map()
    selected = args.site or sorted(sites_by_id)

    total = 0
    for site_id in selected:
        site = sites_by_id.get(site_id)
        if not site:
            print(f"{site_id}: municipio desconocido")
            continue
        try:
            total += update_site(
                site,
                max(1, args.days),
                max(1, args.max_seconds),
                max(1, args.request_attempts),
                max(0, args.request_delay),
                max(1, args.rate_limit_sleep),
                max(1, args.rate_limit_abort_threshold),
            )
        except Exception as exc:
            print(f"{site_id}: error actualizando dataset: {exc}")

    print(f"Total dias nuevos incorporados: {total}")


if __name__ == "__main__":
    main()

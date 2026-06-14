from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from app.actualizar_datasets_saih import DATASET_DIR, _last_date, _records_to_daily, _site_map
    from app.api.saih_opendata import fetch_saih_history
    from app.model_pipeline import BASE_COLUMNS, add_model_features
except ImportError:
    from actualizar_datasets_saih import DATASET_DIR, _last_date, _records_to_daily, _site_map
    from api.saih_opendata import fetch_saih_history
    from model_pipeline import BASE_COLUMNS, add_model_features


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _iter_days(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _site_tags(site: dict) -> list[str]:
    saih = site.get("saih") or {}
    return [tag for tag in [saih.get("nivel"), saih.get("caudal")] if tag]


def _load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=BASE_COLUMNS)


def _save_daily(site: dict, daily: pd.DataFrame) -> int:
    if daily.empty:
        return 0

    site_id = site["id"]
    path = DATASET_DIR / f"{site_id}.csv"
    existing = _load_existing(path)
    base_existing = existing[[column for column in BASE_COLUMNS if column in existing.columns]].copy()

    for column in BASE_COLUMNS:
        if column not in base_existing.columns:
            base_existing[column] = 0

    combined = pd.concat([base_existing[BASE_COLUMNS], daily[BASE_COLUMNS]], ignore_index=True)
    combined = (
        combined.drop_duplicates(subset=["fecha"], keep="last")
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    updated = add_model_features(combined)
    DATASET_DIR.mkdir(exist_ok=True)
    updated.to_csv(path, index=False, encoding="utf-8-sig")
    return len(daily)


def _pending_range(site_id: str, days_back: int, from_date: date | None, to_date: date | None) -> tuple[date, date] | None:
    path = DATASET_DIR / f"{site_id}.csv"
    existing = _load_existing(path)
    end_date = min(to_date or (date.today() - timedelta(days=1)), date.today() - timedelta(days=1))
    last = _last_date(existing)

    if from_date:
        start_date = from_date
    elif last:
        start_date = last + timedelta(days=1)
    else:
        start_date = end_date - timedelta(days=max(1, days_back) - 1)

    if start_date > end_date:
        return None
    return start_date, end_date


def update_site_slow(site: dict, args) -> int:
    site_id = site["id"]
    tags = _site_tags(site)
    if len(tags) < 2:
        print(f"{site_id}: sin senales SAIH suficientes")
        return 0

    pending = _pending_range(site_id, args.days, args.from_date, args.to_date)
    if not pending:
        print(f"{site_id}: ya estaba actualizado")
        return 0

    start_date, end_date = pending
    days = list(_iter_days(start_date, end_date))
    if args.max_days_per_site:
        days = days[: args.max_days_per_site]

    print(f"{site_id}: actualizacion lenta {days[0]} -> {days[-1]} ({len(days)} dias)")
    saved_days = 0

    for index, day in enumerate(days, start=1):
        print(f"{site_id}: [{index}/{len(days)}] solicitando {day}")
        if args.dry_run:
            continue

        records = fetch_saih_history(
            tags,
            day,
            day,
            request_timeout=(4, 12),
            request_attempts=args.request_attempts,
            max_seconds=args.max_seconds_per_request,
            request_delay_seconds=0,
            rate_limit_sleep_seconds=args.rate_limit_sleep,
            rate_limit_abort_threshold_seconds=args.rate_limit_abort_threshold,
        )
        daily = _records_to_daily(site, records)
        if daily.empty:
            print(f"{site_id}: {day} sin datos completos")
        else:
            saved = _save_daily(site, daily)
            saved_days += saved
            print(f"{site_id}: {day} guardado ({saved} dia)")

        if index < len(days) and args.interval_seconds > 0:
            print(f"{site_id}: esperando {args.interval_seconds:.0f}s antes de la siguiente peticion")
            time.sleep(args.interval_seconds)

    return saved_days


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Actualiza datasets SAIH de forma lenta, guardando cada dia y pausando entre peticiones."
    )
    parser.add_argument("--site", action="append", help="ID de municipio. Repetible. Por defecto: todos.")
    parser.add_argument("--days", type=int, default=45, help="Dias hacia atras si el CSV esta vacio.")
    parser.add_argument("--from-date", type=_parse_date, help="Fecha inicial YYYY-MM-DD. Opcional.")
    parser.add_argument("--to-date", type=_parse_date, help="Fecha final YYYY-MM-DD. Como maximo ayer.")
    parser.add_argument("--interval-seconds", type=float, default=120, help="Pausa entre peticiones diarias.")
    parser.add_argument("--site-delay-seconds", type=float, default=300, help="Pausa entre municipios.")
    parser.add_argument("--request-attempts", type=int, default=2, help="Reintentos por dia.")
    parser.add_argument("--rate-limit-sleep", type=float, default=60, help="Espera maxima local tras un 429 corto.")
    parser.add_argument(
        "--rate-limit-abort-threshold",
        type=float,
        default=900,
        help="Si SAIH pide esperar mas que este umbral, se detiene para no saturar.",
    )
    parser.add_argument("--max-seconds-per-request", type=float, default=180, help="Tiempo maximo por dia.")
    parser.add_argument("--max-days-per-site", type=int, default=0, help="Limita dias por municipio en esta ejecucion.")
    parser.add_argument("--continue-on-rate-limit", action="store_true", help="Sigue con otros municipios tras rate limit.")
    parser.add_argument("--dry-run", action="store_true", help="Muestra que haria sin llamar a SAIH.")
    args = parser.parse_args()

    sites_by_id = _site_map()
    selected = args.site or sorted(sites_by_id)
    total = 0

    for position, site_id in enumerate(selected, start=1):
        site = sites_by_id.get(site_id)
        if not site:
            print(f"{site_id}: municipio desconocido")
            continue

        try:
            total += update_site_slow(site, args)
        except KeyboardInterrupt:
            print("Actualizacion interrumpida por el usuario. Puedes reanudar mas tarde.")
            break
        except Exception as exc:
            print(f"{site_id}: error actualizando dataset: {exc}")
            error_text = str(exc).lower()
            if ("limite temporal largo" in error_text or "429" in error_text or "limitando" in error_text) and not args.continue_on_rate_limit:
                print("Se detiene para no seguir golpeando la API de SAIH.")
                break

        if position < len(selected) and args.site_delay_seconds > 0:
            print(f"Esperando {args.site_delay_seconds:.0f}s antes del siguiente municipio")
            time.sleep(args.site_delay_seconds)

    print(f"Total dias nuevos incorporados: {total}")


if __name__ == "__main__":
    main()

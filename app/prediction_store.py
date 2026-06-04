import json
import os
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Optional
from urllib.parse import quote_plus, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

try:
    from psycopg2 import pool
    from psycopg2.extras import Json, RealDictCursor
except ImportError:  # pragma: no cover - only used when Postgres extras are not installed.
    pool = None
    Json = None
    RealDictCursor = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR)).resolve()
PREDICTIONS_FILE = Path(os.getenv("PREDICTIONS_FILE", DATA_DIR / "predictions.json")).resolve()


class PredictionStoreError(ValueError):
    pass


def _utc_now() -> datetime:
    return datetime.utcnow().replace(microsecond=0)


def _iso_now() -> str:
    return _utc_now().isoformat() + "Z"


def _today() -> date:
    try:
        return datetime.now(ZoneInfo(os.getenv("ALERT_TIMEZONE", "Europe/Madrid"))).date()
    except Exception:
        return date.today()


def _date_from_issued_at(value: Optional[str]) -> date:
    if not value:
        return _today()

    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return _today()

    if parsed.tzinfo is not None:
        try:
            parsed = parsed.astimezone(ZoneInfo(os.getenv("ALERT_TIMEZONE", "Europe/Madrid")))
        except Exception:
            parsed = parsed.replace(tzinfo=None)

    return parsed.date()


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)

    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(ZoneInfo(os.getenv("ALERT_TIMEZONE", "Europe/Madrid"))).replace(tzinfo=None)

    return parsed.replace(microsecond=0)


def _hour_bucket(value: Any) -> Optional[datetime]:
    parsed = _parse_datetime(value)
    if not parsed:
        return None
    return parsed.replace(minute=0, second=0, microsecond=0)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        if number != number:
            return None
        return number
    except (TypeError, ValueError):
        return None


def _safe_database_label(database_url: str) -> str:
    try:
        parts = urlsplit(database_url)
        host = parts.hostname or "postgres"
        port = f":{parts.port}" if parts.port else ""
        database = parts.path or ""
        return urlunsplit((parts.scheme, f"{host}{port}", database, "", ""))
    except Exception:
        return "postgres"


def _database_url_from_parts() -> Optional[str]:
    name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    default_sslmode = "require" if host and "neon.tech" in host else ""
    sslmode = os.getenv("DB_SSLMODE", default_sslmode).strip()

    if not all([name, user, password, host]):
        return None

    query = f"?sslmode={quote_plus(sslmode)}" if sslmode else ""

    return (
        f"postgresql://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{quote_plus(name)}{query}"
    )


def _prediction_rows(site: dict[str, Any], predictions: list[dict[str, Any]], issued_at: Optional[str] = None) -> list[dict[str, Any]]:
    if not site:
        raise PredictionStoreError("site_required")

    site_id = str(site.get("id") or "").strip()
    if not site_id:
        raise PredictionStoreError("site_id_required")

    now = issued_at or _iso_now()
    issued_date = _date_from_issued_at(now)
    rows = []

    for point in predictions or []:
        if not isinstance(point, dict):
            continue

        nivel_pred = _float_or_none(point.get("nivel"))
        caudal_pred = _float_or_none(point.get("caudal"))

        if nivel_pred is None and caudal_pred is None:
            continue

        target_date = issued_date + timedelta(days=1)
        row_id = f"{site_id}:{issued_date.isoformat()}:{target_date.isoformat()}"

        rows.append({
            "id": row_id,
            "siteId": site_id,
            "site": str(site.get("name") or site_id),
            "issuedDate": issued_date.isoformat(),
            "issuedAt": now,
            "targetDate": target_date.isoformat(),
            "horizonDay": 1,
            "nivelPred": nivel_pred,
            "caudalPred": caudal_pred,
            "nivelReal": None,
            "caudalReal": None,
            "realObservedAt": None,
            "evaluatedAt": None,
            "updatedAt": now,
            "source": "model",
        })
        break

    return rows


def _metrics(points: list[dict[str, Any]]) -> dict[str, Any]:
    nivel_errors = [
        abs(point["nivel_real"] - point["nivel_pred"])
        for point in points
        if point.get("nivel_real") is not None and point.get("nivel_pred") is not None
    ]
    caudal_errors = [
        abs(point["caudal_real"] - point["caudal_pred"])
        for point in points
        if point.get("caudal_real") is not None and point.get("caudal_pred") is not None
    ]

    samples = sum(
        1
        for point in points
        if point.get("nivel_real") is not None or point.get("caudal_real") is not None
    )

    return {
        "nivel_mae": round(sum(nivel_errors) / len(nivel_errors), 3) if nivel_errors else None,
        "caudal_mae": round(sum(caudal_errors) / len(caudal_errors), 3) if caudal_errors else None,
        "samples": samples,
    }


def _public_point(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": record.get("targetDate"),
        "target_date": record.get("targetDate"),
        "issued_date": record.get("issuedDate"),
        "horizon_day": record.get("horizonDay"),
        "nivel_real": record.get("nivelReal"),
        "nivel_pred": record.get("nivelPred"),
        "caudal_real": record.get("caudalReal"),
        "caudal_pred": record.get("caudalPred"),
    }


def _forecast_points(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = []

    for point in predictions or []:
        if not isinstance(point, dict):
            continue

        nivel = _float_or_none(point.get("nivel"))
        caudal = _float_or_none(point.get("caudal"))

        if nivel is None and caudal is None:
            continue

        points.append({
            "nivel": nivel,
            "caudal": caudal,
        })

    return points


def _forecast_record(site: dict[str, Any], predictions: list[dict[str, Any]], issued_at: Optional[str] = None) -> Optional[dict[str, Any]]:
    if not site:
        raise PredictionStoreError("site_required")

    site_id = str(site.get("id") or "").strip()
    if not site_id:
        raise PredictionStoreError("site_id_required")

    points = _forecast_points(predictions)
    if not points:
        return None

    now = issued_at or _iso_now()
    return {
        "siteId": site_id,
        "site": str(site.get("name") or site_id),
        "issuedAt": now,
        "predictions": points[:7],
        "updatedAt": now,
        "source": "model",
    }


def _public_forecast(record: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not record:
        return None

    return {
        "site_id": record.get("siteId"),
        "site": record.get("site"),
        "issued_at": record.get("issuedAt"),
        "predictions": _forecast_points(record.get("predictions") or []),
        "updated_at": record.get("updatedAt"),
        "source": record.get("source") or "model",
    }


def _sample_record(site: dict[str, Any], nivel_m: Any, caudal_m3s: Any, observed_at: Any) -> Optional[dict[str, Any]]:
    site_id = str((site or {}).get("id") or "").strip()
    if not site_id:
        return None

    bucket = _hour_bucket(observed_at)
    if not bucket:
        return None

    nivel = _float_or_none(nivel_m)
    caudal = _float_or_none(caudal_m3s)
    if nivel is None and caudal is None:
        return None

    now = _iso_now()
    hour = bucket.isoformat(timespec="seconds")
    return {
        "id": f"{site_id}:{hour}",
        "siteId": site_id,
        "site": str((site or {}).get("name") or site_id),
        "sampleHour": hour,
        "observedAt": _parse_datetime(observed_at).isoformat(timespec="seconds"),
        "nivelM": nivel,
        "caudalM3s": caudal,
        "updatedAt": now,
    }


def _daily_record_from_samples(site_id: str, site_name: str, actual_date: date, samples: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    nivel_values = [_float_or_none(item.get("nivelM")) for item in samples]
    caudal_values = [_float_or_none(item.get("caudalM3s")) for item in samples]
    nivel_values = [value for value in nivel_values if value is not None]
    caudal_values = [value for value in caudal_values if value is not None]

    if not nivel_values and not caudal_values:
        return None

    now = _iso_now()
    return {
        "siteId": site_id,
        "site": site_name or site_id,
        "actualDate": actual_date.isoformat(),
        "nivelM": round(sum(nivel_values) / len(nivel_values), 3) if nivel_values else None,
        "caudalM3s": round(sum(caudal_values) / len(caudal_values), 3) if caudal_values else None,
        "sampleCount": max(len(nivel_values), len(caudal_values)),
        "updatedAt": now,
    }


def _public_latest_actual(record: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not record:
        return None

    return {
        "site_id": record.get("siteId"),
        "site": record.get("site"),
        "observed_at": record.get("observedAt"),
        "sample_hour": record.get("sampleHour"),
        "nivel_m": record.get("nivelM"),
        "caudal_m3s": record.get("caudalM3s"),
    }


def _combined_latest_actual(records: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    items = [record for record in records if isinstance(record, dict)]
    if not items:
        return None

    items.sort(
        key=lambda item: str(item.get("observedAt") or item.get("sampleHour") or ""),
        reverse=True,
    )
    latest = items[0]
    nivel_record = next(
        (item for item in items if _float_or_none(item.get("nivelM")) is not None),
        None,
    )
    caudal_record = next(
        (item for item in items if _float_or_none(item.get("caudalM3s")) is not None),
        None,
    )

    return _public_latest_actual({
        "siteId": latest.get("siteId"),
        "site": latest.get("site"),
        "observedAt": latest.get("observedAt"),
        "sampleHour": latest.get("sampleHour"),
        "nivelM": nivel_record.get("nivelM") if nivel_record else None,
        "caudalM3s": caudal_record.get("caudalM3s") if caudal_record else None,
    })


class JsonPredictionStore:
    storage_backend = "json"

    def __init__(self, path: Path = PREDICTIONS_FILE):
        self.path = Path(path).resolve()
        self.lock = RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _empty_data(self) -> dict[str, Any]:
        return {
            "version": 1,
            "predictions": [],
            "forecasts": [],
            "actualSamples": [],
            "dailyActuals": [],
        }

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            data = self._empty_data()
            self._save_unlocked(data)
            return data

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PredictionStoreError("predictions_file_invalid_json") from exc

        if not isinstance(data, dict):
            data = self._empty_data()

        data.setdefault("version", 1)
        data.setdefault("predictions", [])
        data.setdefault("forecasts", [])
        data.setdefault("actualSamples", [])
        data.setdefault("dailyActuals", [])
        return data

    def _save_unlocked(self, data: dict[str, Any]) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, self.path)

    def store_prediction(self, site: dict[str, Any], predictions: list[dict[str, Any]], issued_at: Optional[str] = None) -> int:
        rows = _prediction_rows(site, predictions, issued_at)
        if not rows:
            return 0

        with self.lock:
            data = self._load_unlocked()
            existing = {item.get("id"): item for item in data.get("predictions", []) if isinstance(item, dict)}

            for row in rows:
                previous = existing.get(row["id"], {})
                row["nivelReal"] = previous.get("nivelReal")
                row["caudalReal"] = previous.get("caudalReal")
                row["realObservedAt"] = previous.get("realObservedAt")
                row["evaluatedAt"] = previous.get("evaluatedAt")
                existing[row["id"]] = row

            predictions_list = sorted(
                existing.values(),
                key=lambda item: (item.get("siteId", ""), item.get("issuedDate", ""), item.get("horizonDay", 0)),
                reverse=True,
            )[:5000]
            data["predictions"] = predictions_list
            self._save_unlocked(data)
            return len(rows)

    def store_forecast(self, site: dict[str, Any], predictions: list[dict[str, Any]], issued_at: Optional[str] = None) -> int:
        record = _forecast_record(site, predictions, issued_at)
        if not record:
            return 0

        with self.lock:
            data = self._load_unlocked()
            existing = {
                item.get("siteId"): item
                for item in data.get("forecasts", [])
                if isinstance(item, dict) and item.get("siteId")
            }
            existing[record["siteId"]] = record
            data["forecasts"] = sorted(
                existing.values(),
                key=lambda item: (item.get("siteId", ""), item.get("updatedAt", "")),
            )
            self._save_unlocked(data)

        return len(record["predictions"])

    def latest_forecast(self, site_id: str) -> Optional[dict[str, Any]]:
        with self.lock:
            data = self._load_unlocked()
            forecasts = [
                item
                for item in data.get("forecasts", [])
                if isinstance(item, dict) and item.get("siteId") == site_id
            ]

        if not forecasts:
            return None

        forecasts.sort(key=lambda item: item.get("updatedAt", ""), reverse=True)
        return _public_forecast(forecasts[0])

    def store_actual_sample(self, site: dict[str, Any], nivel_m: Any, caudal_m3s: Any, observed_at: Any) -> int:
        record = _sample_record(site, nivel_m, caudal_m3s, observed_at)
        if not record:
            return 0

        with self.lock:
            data = self._load_unlocked()
            existing = {
                item.get("id"): item
                for item in data.get("actualSamples", [])
                if isinstance(item, dict) and item.get("id")
            }
            existing[record["id"]] = record
            data["actualSamples"] = sorted(
                existing.values(),
                key=lambda item: (item.get("siteId", ""), item.get("sampleHour", "")),
                reverse=True,
            )[:10000]
            self._save_unlocked(data)

        return 1

    def latest_actual(self, site_id: str) -> Optional[dict[str, Any]]:
        with self.lock:
            data = self._load_unlocked()
            samples = [
                item
                for item in data.get("actualSamples", [])
                if isinstance(item, dict) and item.get("siteId") == site_id
            ]

        if samples:
            return _combined_latest_actual(samples)

        daily = [
            {
                "siteId": item.get("siteId"),
                "site": item.get("site"),
                "sampleHour": item.get("actualDate"),
                "observedAt": item.get("actualDate"),
                "nivelM": item.get("nivelM"),
                "caudalM3s": item.get("caudalM3s"),
            }
            for item in data.get("dailyActuals", [])
            if isinstance(item, dict) and item.get("siteId") == site_id
        ]
        return _combined_latest_actual(daily)

    def aggregate_daily_actuals(self, up_to_date: Optional[date] = None, retention_days: int = 2) -> int:
        up_to_date = up_to_date or (_today() - timedelta(days=1))
        retention_cutoff = up_to_date - timedelta(days=max(0, int(retention_days or 0)))

        with self.lock:
            data = self._load_unlocked()
            grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}

            for sample in data.get("actualSamples", []):
                if not isinstance(sample, dict):
                    continue
                sample_hour = _parse_datetime(sample.get("sampleHour"))
                if not sample_hour:
                    continue
                sample_date = sample_hour.date()
                if sample_date > up_to_date:
                    continue
                grouped.setdefault(
                    (str(sample.get("siteId")), sample_date.isoformat()),
                    [],
                ).append(sample)

            existing = {
                (item.get("siteId"), item.get("actualDate")): item
                for item in data.get("dailyActuals", [])
                if isinstance(item, dict) and item.get("siteId") and item.get("actualDate")
            }

            updated = 0
            for (site_id, day), samples in grouped.items():
                site_name = samples[0].get("site") or site_id
                record = _daily_record_from_samples(site_id, site_name, date.fromisoformat(day), samples)
                if record:
                    existing[(site_id, day)] = record
                    updated += 1

            data["dailyActuals"] = sorted(
                existing.values(),
                key=lambda item: (item.get("siteId", ""), item.get("actualDate", "")),
                reverse=True,
            )[:5000]

            data["actualSamples"] = [
                sample
                for sample in data.get("actualSamples", [])
                if isinstance(sample, dict)
                and (_parse_datetime(sample.get("sampleHour")) or datetime.max).date() >= retention_cutoff
            ]

            if updated:
                self._save_unlocked(data)

        return updated

    def daily_actuals_by_date(self, site_id: str, dates: list[date]) -> dict[str, dict[str, Any]]:
        wanted = {day.isoformat() for day in dates}
        if not wanted:
            return {}

        with self.lock:
            data = self._load_unlocked()
            records = [
                item
                for item in data.get("dailyActuals", [])
                if isinstance(item, dict)
                and item.get("siteId") == site_id
                and item.get("actualDate") in wanted
            ]

        return {
            str(record.get("actualDate")): {
                "nivel_m": record.get("nivelM"),
                "caudal_m3s": record.get("caudalM3s"),
                "observed_at": record.get("actualDate"),
                "sample_count": record.get("sampleCount"),
            }
            for record in records
        }

    def pending_actual_dates(
        self,
        site_id: str,
        max_date: date,
        limit: int = 14,
        refresh_date: Optional[date] = None,
    ) -> list[date]:
        with self.lock:
            data = self._load_unlocked()
            dates = []

            for record in data.get("predictions", []):
                if record.get("siteId") != site_id:
                    continue
                if int(record.get("horizonDay") or 0) != 1:
                    continue
                try:
                    target = date.fromisoformat(str(record.get("targetDate")))
                except ValueError:
                    continue
                if target > max_date:
                    continue
                if refresh_date and target == refresh_date:
                    if target not in dates:
                        dates.append(target)
                    continue
                needs_nivel = record.get("nivelPred") is not None and record.get("nivelReal") is None
                needs_caudal = record.get("caudalPred") is not None and record.get("caudalReal") is None
                if not (needs_nivel or needs_caudal):
                    continue
                if target not in dates:
                    dates.append(target)

            return sorted(dates)[: max(1, min(int(limit or 14), 60))]

    def update_actuals(self, site_id: str, actuals_by_date: dict[str, dict[str, Any]]) -> int:
        if not actuals_by_date:
            return 0

        with self.lock:
            data = self._load_unlocked()
            updated = 0
            now = _iso_now()

            for record in data.get("predictions", []):
                if record.get("siteId") != site_id:
                    continue
                if int(record.get("horizonDay") or 0) != 1:
                    continue
                actual = actuals_by_date.get(str(record.get("targetDate")))
                if not actual:
                    continue

                nivel_real = _float_or_none(actual.get("nivel_m"))
                caudal_real = _float_or_none(actual.get("caudal_m3s"))

                if nivel_real is None and caudal_real is None:
                    continue

                if nivel_real is not None:
                    record["nivelReal"] = nivel_real
                if caudal_real is not None:
                    record["caudalReal"] = caudal_real
                record["realObservedAt"] = actual.get("observed_at") or now
                record["evaluatedAt"] = now
                record["updatedAt"] = now
                updated += 1

            if updated:
                self._save_unlocked(data)

            return updated

    def evaluation(self, site_id: str, limit: int = 30) -> dict[str, Any]:
        with self.lock:
            data = self._load_unlocked()
            records = [
                record
                for record in data.get("predictions", [])
                if record.get("siteId") == site_id and int(record.get("horizonDay") or 0) == 1
            ]

        pending = sum(
            1
            for record in records
            if (
                record.get("nivelPred") is not None and record.get("nivelReal") is None
            ) or (
                record.get("caudalPred") is not None and record.get("caudalReal") is None
            )
        )
        records.sort(key=lambda item: (item.get("targetDate", ""), item.get("issuedDate", ""), item.get("horizonDay", 0)))
        records = records[-max(1, min(int(limit or 30), 90)):]
        points = [_public_point(record) for record in records]

        return {
            "points": points,
            "metrics": _metrics(points),
            "pending": pending,
            "total": len(records),
            "mode": "persisted_predictions",
        }


class PostgresPredictionStore:
    storage_backend = "postgres"

    def __init__(self, database_url: str, minconn: int = 1, maxconn: int = 3):
        if pool is None or Json is None or RealDictCursor is None:
            raise PredictionStoreError("psycopg2_not_installed")

        self.database_url = database_url
        self.path = _safe_database_label(database_url)
        minconn = int(os.getenv("POSTGRES_POOL_MIN", str(minconn)))
        maxconn = int(os.getenv("POSTGRES_POOL_MAX", str(maxconn)))
        self.pool = pool.ThreadedConnectionPool(minconn, maxconn, dsn=database_url)
        self._ensure_schema()

    @contextmanager
    def _connection(self):
        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.pool.putconn(conn)

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prediction_points (
                        id TEXT PRIMARY KEY,
                        site_id TEXT NOT NULL,
                        site_name TEXT NOT NULL,
                        issued_date DATE NOT NULL,
                        issued_at TEXT NOT NULL,
                        target_date DATE NOT NULL,
                        horizon_day INTEGER NOT NULL,
                        nivel_pred DOUBLE PRECISION,
                        caudal_pred DOUBLE PRECISION,
                        nivel_real DOUBLE PRECISION,
                        caudal_real DOUBLE PRECISION,
                        real_observed_at TEXT,
                        evaluated_at TEXT,
                        updated_at TEXT NOT NULL,
                        source TEXT NOT NULL DEFAULT 'model',
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        UNIQUE(site_id, issued_date, target_date)
                    );

                    CREATE INDEX IF NOT EXISTS idx_prediction_points_site_target
                        ON prediction_points(site_id, target_date);
                    CREATE INDEX IF NOT EXISTS idx_prediction_points_pending
                        ON prediction_points(site_id, target_date)
                        WHERE (nivel_pred IS NOT NULL AND nivel_real IS NULL)
                           OR (caudal_pred IS NOT NULL AND caudal_real IS NULL);

                    CREATE TABLE IF NOT EXISTS prediction_forecasts (
                        site_id TEXT PRIMARY KEY,
                        site_name TEXT NOT NULL,
                        issued_at TEXT NOT NULL,
                        predictions JSONB NOT NULL,
                        updated_at TEXT NOT NULL,
                        source TEXT NOT NULL DEFAULT 'model'
                    );

                    CREATE TABLE IF NOT EXISTS hydrology_actual_samples (
                        id TEXT PRIMARY KEY,
                        site_id TEXT NOT NULL,
                        site_name TEXT NOT NULL,
                        sample_hour TIMESTAMP NOT NULL,
                        observed_at TEXT NOT NULL,
                        nivel_m DOUBLE PRECISION,
                        caudal_m3s DOUBLE PRECISION,
                        updated_at TEXT NOT NULL,
                        UNIQUE(site_id, sample_hour)
                    );

                    CREATE INDEX IF NOT EXISTS idx_hydrology_samples_site_hour
                        ON hydrology_actual_samples(site_id, sample_hour);

                    CREATE TABLE IF NOT EXISTS hydrology_daily_actuals (
                        site_id TEXT NOT NULL,
                        site_name TEXT NOT NULL,
                        actual_date DATE NOT NULL,
                        nivel_m DOUBLE PRECISION,
                        caudal_m3s DOUBLE PRECISION,
                        sample_count INTEGER NOT NULL DEFAULT 0,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(site_id, actual_date)
                    );
                    """
                )

    def store_prediction(self, site: dict[str, Any], predictions: list[dict[str, Any]], issued_at: Optional[str] = None) -> int:
        rows = _prediction_rows(site, predictions, issued_at)
        if not rows:
            return 0

        with self._connection() as conn:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute(
                        """
                        INSERT INTO prediction_points (
                            id, site_id, site_name, issued_date, issued_at, target_date,
                            horizon_day, nivel_pred, caudal_pred, updated_at, source, metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (site_id, issued_date, target_date) DO UPDATE SET
                            site_name = EXCLUDED.site_name,
                            issued_at = EXCLUDED.issued_at,
                            horizon_day = EXCLUDED.horizon_day,
                            nivel_pred = EXCLUDED.nivel_pred,
                            caudal_pred = EXCLUDED.caudal_pred,
                            updated_at = EXCLUDED.updated_at,
                            source = EXCLUDED.source
                        """,
                        (
                            row["id"],
                            row["siteId"],
                            row["site"],
                            row["issuedDate"],
                            row["issuedAt"],
                            row["targetDate"],
                            row["horizonDay"],
                            row["nivelPred"],
                            row["caudalPred"],
                            row["updatedAt"],
                            row["source"],
                            Json({}),
                        ),
                    )

        return len(rows)

    def store_forecast(self, site: dict[str, Any], predictions: list[dict[str, Any]], issued_at: Optional[str] = None) -> int:
        record = _forecast_record(site, predictions, issued_at)
        if not record:
            return 0

        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prediction_forecasts (
                        site_id, site_name, issued_at, predictions, updated_at, source
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (site_id) DO UPDATE SET
                        site_name = EXCLUDED.site_name,
                        issued_at = EXCLUDED.issued_at,
                        predictions = EXCLUDED.predictions,
                        updated_at = EXCLUDED.updated_at,
                        source = EXCLUDED.source
                    """,
                    (
                        record["siteId"],
                        record["site"],
                        record["issuedAt"],
                        Json(record["predictions"]),
                        record["updatedAt"],
                        record["source"],
                    ),
                )

        return len(record["predictions"])

    def latest_forecast(self, site_id: str) -> Optional[dict[str, Any]]:
        with self._connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT site_id, site_name, issued_at, predictions, updated_at, source
                    FROM prediction_forecasts
                    WHERE site_id = %s
                    """,
                    (site_id,),
                )
                row = cur.fetchone()

        if not row:
            return None

        return _public_forecast({
            "siteId": row["site_id"],
            "site": row["site_name"],
            "issuedAt": row["issued_at"],
            "predictions": row["predictions"],
            "updatedAt": row["updated_at"],
            "source": row["source"],
        })

    def store_actual_sample(self, site: dict[str, Any], nivel_m: Any, caudal_m3s: Any, observed_at: Any) -> int:
        record = _sample_record(site, nivel_m, caudal_m3s, observed_at)
        if not record:
            return 0

        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO hydrology_actual_samples (
                        id, site_id, site_name, sample_hour, observed_at,
                        nivel_m, caudal_m3s, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (site_id, sample_hour) DO UPDATE SET
                        site_name = EXCLUDED.site_name,
                        observed_at = EXCLUDED.observed_at,
                        nivel_m = COALESCE(EXCLUDED.nivel_m, hydrology_actual_samples.nivel_m),
                        caudal_m3s = COALESCE(EXCLUDED.caudal_m3s, hydrology_actual_samples.caudal_m3s),
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        record["id"],
                        record["siteId"],
                        record["site"],
                        record["sampleHour"],
                        record["observedAt"],
                        record["nivelM"],
                        record["caudalM3s"],
                        record["updatedAt"],
                    ),
                )

        return 1

    def latest_actual(self, site_id: str) -> Optional[dict[str, Any]]:
        with self._connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT site_id, site_name, sample_hour, observed_at, nivel_m, caudal_m3s
                    FROM hydrology_actual_samples
                    WHERE site_id = %s
                    ORDER BY sample_hour DESC
                    LIMIT 200
                    """,
                    (site_id,),
                )
                rows = [dict(row) for row in cur.fetchall()]

        if rows:
            return _combined_latest_actual([
                {
                    "siteId": row["site_id"],
                    "site": row["site_name"],
                    "sampleHour": row["sample_hour"].isoformat(timespec="seconds")
                    if hasattr(row["sample_hour"], "isoformat")
                    else str(row["sample_hour"]),
                    "observedAt": row["observed_at"],
                    "nivelM": row["nivel_m"],
                    "caudalM3s": row["caudal_m3s"],
                }
                for row in rows
            ])

        with self._connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT site_id, site_name, actual_date, nivel_m, caudal_m3s
                    FROM hydrology_daily_actuals
                    WHERE site_id = %s
                    ORDER BY actual_date DESC
                    LIMIT 30
                    """,
                    (site_id,),
                )
                rows = [dict(row) for row in cur.fetchall()]

        return _combined_latest_actual([
            {
                "siteId": row["site_id"],
                "site": row["site_name"],
                "sampleHour": row["actual_date"].isoformat()
                if hasattr(row["actual_date"], "isoformat")
                else str(row["actual_date"]),
                "observedAt": row["actual_date"].isoformat()
                if hasattr(row["actual_date"], "isoformat")
                else str(row["actual_date"]),
                "nivelM": row["nivel_m"],
                "caudalM3s": row["caudal_m3s"],
            }
            for row in rows
        ])

    def aggregate_daily_actuals(self, up_to_date: Optional[date] = None, retention_days: int = 2) -> int:
        up_to_date = up_to_date or (_today() - timedelta(days=1))
        retention_cutoff = up_to_date - timedelta(days=max(0, int(retention_days or 0)))
        now = _iso_now()

        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO hydrology_daily_actuals (
                        site_id, site_name, actual_date, nivel_m, caudal_m3s, sample_count, updated_at
                    )
                    SELECT site_id,
                           MAX(site_name) AS site_name,
                           sample_hour::date AS actual_date,
                           AVG(nivel_m) FILTER (WHERE nivel_m IS NOT NULL) AS nivel_m,
                           AVG(caudal_m3s) FILTER (WHERE caudal_m3s IS NOT NULL) AS caudal_m3s,
                           GREATEST(COUNT(nivel_m), COUNT(caudal_m3s))::integer AS sample_count,
                           %s AS updated_at
                    FROM hydrology_actual_samples
                    WHERE sample_hour::date <= %s
                    GROUP BY site_id, sample_hour::date
                    HAVING COUNT(nivel_m) > 0 OR COUNT(caudal_m3s) > 0
                    ON CONFLICT (site_id, actual_date) DO UPDATE SET
                        site_name = EXCLUDED.site_name,
                        nivel_m = ROUND(EXCLUDED.nivel_m::numeric, 3)::double precision,
                        caudal_m3s = ROUND(EXCLUDED.caudal_m3s::numeric, 3)::double precision,
                        sample_count = EXCLUDED.sample_count,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (now, up_to_date),
                )
                updated = cur.rowcount
                cur.execute(
                    """
                    DELETE FROM hydrology_actual_samples
                    WHERE sample_hour::date < %s
                    """,
                    (retention_cutoff,),
                )

        return updated

    def daily_actuals_by_date(self, site_id: str, dates: list[date]) -> dict[str, dict[str, Any]]:
        if not dates:
            return {}

        with self._connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT actual_date, nivel_m, caudal_m3s, sample_count
                    FROM hydrology_daily_actuals
                    WHERE site_id = %s
                      AND actual_date = ANY(%s)
                    """,
                    (site_id, dates),
                )
                rows = [dict(row) for row in cur.fetchall()]

        return {
            row["actual_date"].isoformat()
            if hasattr(row["actual_date"], "isoformat")
            else str(row["actual_date"]): {
                "nivel_m": row["nivel_m"],
                "caudal_m3s": row["caudal_m3s"],
                "observed_at": row["actual_date"].isoformat()
                if hasattr(row["actual_date"], "isoformat")
                else str(row["actual_date"]),
                "sample_count": row["sample_count"],
            }
            for row in rows
        }

    def pending_actual_dates(
        self,
        site_id: str,
        max_date: date,
        limit: int = 14,
        refresh_date: Optional[date] = None,
    ) -> list[date]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT target_date
                    FROM prediction_points
                    WHERE site_id = %s
                      AND horizon_day = 1
                      AND target_date <= %s
                      AND (
                          (nivel_pred IS NOT NULL AND nivel_real IS NULL)
                          OR (caudal_pred IS NOT NULL AND caudal_real IS NULL)
                          OR target_date = %s
                      )
                    ORDER BY target_date ASC
                    LIMIT %s
                    """,
                    (site_id, max_date, refresh_date, max(1, min(int(limit or 14), 60))),
                )
                return [row[0] for row in cur.fetchall()]

    def update_actuals(self, site_id: str, actuals_by_date: dict[str, dict[str, Any]]) -> int:
        if not actuals_by_date:
            return 0

        updated = 0
        now = _iso_now()

        with self._connection() as conn:
            with conn.cursor() as cur:
                for target_date, actual in actuals_by_date.items():
                    nivel_real = _float_or_none(actual.get("nivel_m"))
                    caudal_real = _float_or_none(actual.get("caudal_m3s"))

                    if nivel_real is None and caudal_real is None:
                        continue

                    cur.execute(
                        """
                        UPDATE prediction_points
                        SET nivel_real = COALESCE(%s, nivel_real),
                            caudal_real = COALESCE(%s, caudal_real),
                            real_observed_at = %s,
                            evaluated_at = %s,
                            updated_at = %s
                        WHERE site_id = %s
                          AND target_date = %s
                          AND horizon_day = 1
                        """,
                        (
                            nivel_real,
                            caudal_real,
                            actual.get("observed_at") or now,
                            now,
                            now,
                            site_id,
                            target_date,
                        ),
                    )
                    updated += cur.rowcount

        return updated

    def evaluation(self, site_id: str, limit: int = 30) -> dict[str, Any]:
        limit = max(1, min(int(limit or 30), 90))

        with self._connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) AS total,
                           COUNT(*) FILTER (
                               WHERE (nivel_pred IS NOT NULL AND nivel_real IS NULL)
                                  OR (caudal_pred IS NOT NULL AND caudal_real IS NULL)
                           ) AS pending
                    FROM prediction_points
                    WHERE site_id = %s
                      AND horizon_day = 1
                    """,
                    (site_id,),
                )
                counts = dict(cur.fetchone() or {})

                cur.execute(
                    """
                    SELECT site_id, site_name, issued_date, issued_at, target_date, horizon_day,
                           nivel_pred, caudal_pred, nivel_real, caudal_real
                    FROM prediction_points
                    WHERE site_id = %s
                      AND horizon_day = 1
                    ORDER BY target_date DESC, issued_date DESC, horizon_day DESC
                    LIMIT %s
                    """,
                    (site_id, limit),
                )
                rows = [dict(row) for row in cur.fetchall()]

        rows.reverse()
        points = [
            {
                "date": row["target_date"].isoformat() if hasattr(row["target_date"], "isoformat") else str(row["target_date"]),
                "target_date": row["target_date"].isoformat() if hasattr(row["target_date"], "isoformat") else str(row["target_date"]),
                "issued_date": row["issued_date"].isoformat() if hasattr(row["issued_date"], "isoformat") else str(row["issued_date"]),
                "horizon_day": row["horizon_day"],
                "nivel_real": row["nivel_real"],
                "nivel_pred": row["nivel_pred"],
                "caudal_real": row["caudal_real"],
                "caudal_pred": row["caudal_pred"],
            }
            for row in rows
        ]

        return {
            "points": points,
            "metrics": _metrics(points),
            "pending": int(counts.get("pending") or 0),
            "total": int(counts.get("total") or 0),
            "mode": "persisted_predictions",
        }


def create_prediction_store():
    backend = os.getenv("PREDICTION_STORE_BACKEND", os.getenv("USER_STORE_BACKEND", "auto")).strip().lower()
    database_url = (
        os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or os.getenv("POSTGRES_DATABASE_URL")
        or _database_url_from_parts()
    )

    wants_postgres = backend in {"postgres", "postgresql", "pg"}
    auto_postgres = backend == "auto" and bool(database_url)

    if wants_postgres or auto_postgres:
        if not database_url:
            raise PredictionStoreError("database_url_missing")
        return PostgresPredictionStore(database_url=database_url)

    return JsonPredictionStore()

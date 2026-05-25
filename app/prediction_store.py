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
    issued_date = _today()
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

    return {
        "nivel_mae": round(sum(nivel_errors) / len(nivel_errors), 3) if nivel_errors else None,
        "caudal_mae": round(sum(caudal_errors) / len(caudal_errors), 3) if caudal_errors else None,
        "samples": len(points),
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


class JsonPredictionStore:
    storage_backend = "json"

    def __init__(self, path: Path = PREDICTIONS_FILE):
        self.path = Path(path).resolve()
        self.lock = RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _empty_data(self) -> dict[str, Any]:
        return {"version": 1, "predictions": []}

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

        evaluated = [
            record
            for record in records
            if record.get("nivelReal") is not None or record.get("caudalReal") is not None
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
        evaluated.sort(key=lambda item: (item.get("targetDate", ""), item.get("issuedDate", ""), item.get("horizonDay", 0)))
        evaluated = evaluated[-max(1, min(int(limit or 30), 90)):]
        points = [_public_point(record) for record in evaluated]

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
                      AND (nivel_real IS NOT NULL OR caudal_real IS NOT NULL)
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

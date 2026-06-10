import json
import os
import secrets
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

import psycopg2
from psycopg2 import pool
from psycopg2.extras import Json, RealDictCursor

try:
    from app.env_utils import env_int, env_value
except ImportError:
    from env_utils import env_int, env_value

from app.user_store import (
    DEFAULT_PREFERENCES,
    SESSION_DAYS,
    UserStoreError,
    _hash_password,
    _iso_now,
    _json_object,
    _minutes_from_alert_time,
    _normalize_download_record,
    _normalize_email,
    _normalize_name,
    _normalize_preferences,
    _utc_now,
    _valid_sites,
    _verify_password,
)


def _safe_database_label(database_url: str) -> str:
    try:
        parts = urlsplit(database_url)
        host = parts.hostname or "postgres"
        port = f":{parts.port}" if parts.port else ""
        database = parts.path or ""
        return urlunsplit((parts.scheme, f"{host}{port}", database, "", ""))
    except Exception:
        return "postgres"


class PostgresUserStore:
    storage_backend = "postgres"

    def __init__(
        self,
        database_url: str,
        sites_by_id: Optional[dict[str, Any]] = None,
        minconn: int = 1,
        maxconn: int = 5,
    ):
        self.database_url = database_url
        self.path = _safe_database_label(database_url)
        self.sites_by_id = sites_by_id or {}
        self.lock = RLock()
        minconn = env_int("POSTGRES_POOL_MIN", minconn)
        maxconn = env_int("POSTGRES_POOL_MAX", maxconn)
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
                    CREATE TABLE IF NOT EXISTS app_users (
                        id TEXT PRIMARY KEY,
                        email TEXT NOT NULL UNIQUE,
                        name TEXT NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        preferences JSONB NOT NULL DEFAULT '{}'::jsonb,
                        alert_state JSONB NOT NULL DEFAULT '{}'::jsonb
                    );

                    CREATE TABLE IF NOT EXISTS user_sessions (
                        token TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS push_devices (
                        token TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
                        created_at TEXT NOT NULL,
                        last_seen_at TEXT NOT NULL,
                        user_agent TEXT NOT NULL DEFAULT '',
                        platform TEXT NOT NULL DEFAULT ''
                    );

                    CREATE TABLE IF NOT EXISTS download_records (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
                        filename TEXT NOT NULL,
                        site_id TEXT NOT NULL DEFAULT '',
                        site_name TEXT NOT NULL DEFAULT '',
                        start_date TEXT NOT NULL DEFAULT '',
                        end_date TEXT NOT NULL DEFAULT '',
                        variable TEXT NOT NULL DEFAULT 'both',
                        granularity TEXT NOT NULL DEFAULT 'hourly',
                        file_format TEXT NOT NULL DEFAULT 'xlsx',
                        bytes BIGINT NOT NULL DEFAULT 0,
                        downloaded_at TEXT NOT NULL,
                        has_local_handle BOOLEAN NOT NULL DEFAULT FALSE,
                        saved_with_picker BOOLEAN NOT NULL DEFAULT FALSE,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                    );

                    CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
                    CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);
                    CREATE INDEX IF NOT EXISTS idx_push_devices_user_id ON push_devices(user_id);
                    CREATE INDEX IF NOT EXISTS idx_download_records_user_id ON download_records(user_id);
                    CREATE INDEX IF NOT EXISTS idx_download_records_downloaded_at ON download_records(downloaded_at);
                    """
                )

    def _valid_sites(self, sites: Any) -> list[str]:
        return _valid_sites(sites, self.sites_by_id)

    def _public_user(self, user: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not user:
            return None

        return {
            "id": user.get("id"),
            "name": user.get("name"),
            "email": user.get("email"),
            "created_at": user.get("created_at"),
            "preferences": deepcopy(user.get("preferences") or DEFAULT_PREFERENCES),
            "has_push_devices": any((device.get("token") or "").strip() for device in user.get("devices", [])),
        }

    def _devices_for_user(self, conn, user_id: str) -> list[dict[str, Any]]:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT token, created_at, last_seen_at, user_agent, platform
                FROM push_devices
                WHERE user_id = %s
                ORDER BY last_seen_at DESC
                """,
                (user_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def _row_to_user(self, conn, row: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not row:
            return None

        user = {
            "id": row["id"],
            "name": row["name"],
            "email": row["email"],
            "password_hash": row["password_hash"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "preferences": _normalize_preferences(row.get("preferences"), self.sites_by_id),
            "devices": self._devices_for_user(conn, row["id"]),
            "alert_state": _json_object(row.get("alert_state")),
        }
        return user

    def _get_user_row(self, conn, user_id: str) -> Optional[dict[str, Any]]:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM app_users WHERE id = %s", (user_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def _cleanup_expired_sessions(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM user_sessions WHERE expires_at < %s", (_iso_now(),))

    def create_user(self, name: str, email: str, password: str) -> dict[str, Any]:
        with self.lock, self._connection() as conn:
            email = _normalize_email(email)
            user_id = secrets.token_urlsafe(12)
            now = _iso_now()
            preferences = deepcopy(DEFAULT_PREFERENCES)

            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO app_users
                            (id, email, name, password_hash, created_at, updated_at, preferences, alert_state)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            user_id,
                            email,
                            _normalize_name(name, email),
                            _hash_password(password),
                            now,
                            now,
                            Json(preferences),
                            Json({}),
                        ),
                    )
            except psycopg2.IntegrityError as exc:
                if getattr(exc, "pgcode", "") == "23505":
                    raise UserStoreError("email_already_registered") from exc
                raise

            return self._public_user(self._row_to_user(conn, self._get_user_row(conn, user_id)))

    def authenticate(self, email: str, password: str) -> Optional[dict[str, Any]]:
        with self.lock, self._connection() as conn:
            email = _normalize_email(email)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM app_users WHERE email = %s", (email,))
                row = cur.fetchone()

            if row and _verify_password(password, row.get("password_hash", "")):
                return self._public_user(self._row_to_user(conn, dict(row)))

        return None

    def create_session(self, user_id: str) -> str:
        with self.lock, self._connection() as conn:
            if not self._get_user_row(conn, user_id):
                raise UserStoreError("user_not_found")

            self._cleanup_expired_sessions(conn)
            token = secrets.token_urlsafe(32)
            expires_at = (_utc_now() + timedelta(days=SESSION_DAYS)).isoformat() + "Z"

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_sessions (token, user_id, created_at, expires_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (token, user_id, _iso_now(), expires_at),
                )

            return token

    def delete_session(self, session_token: str) -> None:
        if not session_token:
            return

        with self.lock, self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM user_sessions WHERE token = %s", (session_token,))

    def get_user_by_session(self, session_token: str) -> Optional[dict[str, Any]]:
        if not session_token:
            return None

        with self.lock, self._connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM user_sessions WHERE token = %s", (session_token,))
                session = cur.fetchone()

            if not session or self._session_is_expired(dict(session)):
                if session:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM user_sessions WHERE token = %s", (session_token,))
                return None

            return self._row_to_user(conn, self._get_user_row(conn, session["user_id"]))

    def get_public_user_by_session(self, session_token: str) -> Optional[dict[str, Any]]:
        return self._public_user(self.get_user_by_session(session_token))

    def get_user(self, user_id: str) -> Optional[dict[str, Any]]:
        with self.lock, self._connection() as conn:
            return self._row_to_user(conn, self._get_user_row(conn, user_id))

    def list_users(self) -> list[dict[str, Any]]:
        with self.lock, self._connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM app_users ORDER BY created_at ASC")
                rows = cur.fetchall()

            return [
                user
                for user in (self._row_to_user(conn, dict(row)) for row in rows)
                if user
            ]

    def update_preferences(self, user_id: str, preferences: dict[str, Any]) -> dict[str, Any]:
        with self.lock, self._connection() as conn:
            row = self._get_user_row(conn, user_id)
            if not row:
                raise UserStoreError("user_not_found")

            current = _normalize_preferences(row.get("preferences"), self.sites_by_id)

            if "notification_channel" in preferences:
                channel = str(preferences["notification_channel"]).strip().lower()
                if channel not in {"push", "email"}:
                    raise UserStoreError("notification_channel_invalid")
                current["notification_channel"] = channel

            if "alert_time" in preferences:
                current["alert_time"] = _normalize_preferences(
                    {"alert_time": str(preferences["alert_time"])},
                    self.sites_by_id,
                )["alert_time"]

            if "theme" in preferences:
                theme = str(preferences["theme"]).strip().lower()
                if theme not in {"dark", "light"}:
                    raise UserStoreError("theme_invalid")
                current["theme"] = theme

            if "sites" in preferences:
                current["sites"] = self._valid_sites(preferences["sites"])

            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE app_users SET preferences = %s, updated_at = %s WHERE id = %s",
                    (Json(current), _iso_now(), user_id),
                )

            return self._public_user(self._row_to_user(conn, self._get_user_row(conn, user_id)))

    def update_profile(
        self,
        user_id: str,
        name: Optional[str] = None,
        preferences: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        with self.lock, self._connection() as conn:
            row = self._get_user_row(conn, user_id)
            if not row:
                raise UserStoreError("user_not_found")

            new_name = row["name"] if name is None else _normalize_name(str(name), row.get("email", ""))
            current = _normalize_preferences(row.get("preferences"), self.sites_by_id)

            if preferences is not None:
                current = self._merge_preferences(current, preferences)

            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE app_users
                    SET name = %s, preferences = %s, updated_at = %s
                    WHERE id = %s
                    """,
                    (new_name, Json(current), _iso_now(), user_id),
                )

            return self._public_user(self._row_to_user(conn, self._get_user_row(conn, user_id)))

    def _merge_preferences(self, current: dict[str, Any], preferences: dict[str, Any]) -> dict[str, Any]:
        merged = deepcopy(current)

        if "notification_channel" in preferences:
            channel = str(preferences["notification_channel"]).strip().lower()
            if channel not in {"push", "email"}:
                raise UserStoreError("notification_channel_invalid")
            merged["notification_channel"] = channel

        if "alert_time" in preferences:
            merged["alert_time"] = _normalize_preferences(
                {"alert_time": str(preferences["alert_time"])},
                self.sites_by_id,
            )["alert_time"]

        if "theme" in preferences:
            theme = str(preferences["theme"]).strip().lower()
            if theme not in {"dark", "light"}:
                raise UserStoreError("theme_invalid")
            merged["theme"] = theme

        if "sites" in preferences:
            merged["sites"] = self._valid_sites(preferences["sites"])

        return merged

    def save_push_subscription(
        self,
        user_id: str,
        token: str,
        sites: list[str],
        user_agent: str = "",
        platform: str = "",
    ) -> dict[str, Any]:
        token = (token or "").strip()
        if not token:
            raise UserStoreError("token_empty")

        with self.lock, self._connection() as conn:
            row = self._get_user_row(conn, user_id)
            if not row:
                raise UserStoreError("user_not_found")

            valid_sites = self._valid_sites(sites)
            now = _iso_now()

            with conn.cursor() as cur:
                cur.execute("DELETE FROM push_devices WHERE token = %s", (token,))

                if valid_sites:
                    cur.execute(
                        """
                        INSERT INTO push_devices
                            (token, user_id, created_at, last_seen_at, user_agent, platform)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            token,
                            user_id,
                            now,
                            now,
                            (user_agent or "")[:180],
                            (platform or "")[:80],
                        ),
                    )

                preferences = {
                    **_normalize_preferences(row.get("preferences"), self.sites_by_id),
                    "sites": valid_sites,
                }
                cur.execute(
                    "UPDATE app_users SET preferences = %s, updated_at = %s WHERE id = %s",
                    (Json(preferences), now, user_id),
                )

            return self._public_user(self._row_to_user(conn, self._get_user_row(conn, user_id)))

    def remove_push_subscription(self, user_id: str, token: str) -> dict[str, Any]:
        token = (token or "").strip()

        with self.lock, self._connection() as conn:
            if not self._get_user_row(conn, user_id):
                raise UserStoreError("user_not_found")

            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM push_devices WHERE user_id = %s AND token = %s",
                    (user_id, token),
                )
                cur.execute(
                    "UPDATE app_users SET updated_at = %s WHERE id = %s",
                    (_iso_now(), user_id),
                )

            return self._public_user(self._row_to_user(conn, self._get_user_row(conn, user_id)))

    def remove_invalid_tokens(self, invalid_tokens: set[str]) -> int:
        invalid_tokens = [token for token in (invalid_tokens or set()) if token]
        if not invalid_tokens:
            return 0

        with self.lock, self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM push_devices WHERE token = ANY(%s) RETURNING token",
                    (invalid_tokens,),
                )
                return len(cur.fetchall())

    def users_due_for_alert(self, force: bool = False, user_id: Optional[str] = None) -> list[dict[str, Any]]:
        now = self._alert_now()
        today = now.strftime("%Y-%m-%d")
        current_minutes = now.hour * 60 + now.minute
        due = []

        for user in self.list_users():
            if user_id and user.get("id") != user_id:
                continue

            preferences = {
                **DEFAULT_PREFERENCES,
                **(user.get("preferences") or {}),
            }

            if not preferences.get("sites"):
                continue

            if preferences.get("notification_channel") == "push":
                has_push_token = any(
                    (device.get("token") or "").strip()
                    for device in user.get("devices", [])
                )
                if not has_push_token:
                    continue

            if not force:
                try:
                    alert_minutes = _minutes_from_alert_time(preferences.get("alert_time", "08:00"))
                except UserStoreError:
                    alert_minutes = _minutes_from_alert_time(DEFAULT_PREFERENCES["alert_time"])

                if current_minutes < alert_minutes:
                    continue

                alert_state = user.get("alert_state") or {}
                last_result = alert_state.get("last_result") or {}
                try:
                    last_sent = int(last_result.get("sent", 0) or 0)
                except (TypeError, ValueError):
                    last_sent = 0

                last_channel = alert_state.get("last_channel")
                current_channel = preferences.get("notification_channel")

                if (
                    alert_state.get("last_sent_date") == today
                    and last_sent > 0
                    and (not last_channel or last_channel == current_channel)
                ):
                    continue

            due.append(deepcopy(user))

        return due

    def mark_alert_result(self, user_id: str, result: dict[str, Any]) -> None:
        with self.lock, self._connection() as conn:
            if not self._get_user_row(conn, user_id):
                return

            now = self._alert_now()
            user = self._row_to_user(conn, self._get_user_row(conn, user_id)) or {}
            alert_state = {
                "last_sent_date": now.strftime("%Y-%m-%d"),
                "last_sent_at": _iso_now(),
                "last_channel": (user.get("preferences") or {}).get("notification_channel"),
                "last_result": {
                    "sent": int(result.get("sent", 0)),
                    "processed_sites": int(result.get("processed_sites", 0)),
                    "errors": list(result.get("errors", []))[:5],
                },
            }

            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE app_users SET alert_state = %s, updated_at = %s WHERE id = %s",
                    (Json(alert_state), _iso_now(), user_id),
                )

    def token_site_map(self) -> dict[str, set[str]]:
        mapping: dict[str, set[str]] = {}

        for user in self.list_users():
            preferences = {
                **DEFAULT_PREFERENCES,
                **(user.get("preferences") or {}),
            }

            if preferences.get("notification_channel") != "push":
                continue

            tokens = [
                device.get("token")
                for device in user.get("devices", [])
                if device.get("token")
            ]

            for site_id in preferences.get("sites", []):
                mapping.setdefault(site_id, set()).update(tokens)

        return mapping

    def record_download(self, user_id: str, download: dict[str, Any]) -> dict[str, Any]:
        with self.lock, self._connection() as conn:
            if not self._get_user_row(conn, user_id):
                raise UserStoreError("user_not_found")

            record = _normalize_download_record(download, self.sites_by_id)

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO download_records (
                        id, user_id, filename, site_id, site_name, start_date, end_date,
                        variable, granularity, file_format, bytes, downloaded_at,
                        has_local_handle, saved_with_picker, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        site_id = EXCLUDED.site_id,
                        site_name = EXCLUDED.site_name,
                        start_date = EXCLUDED.start_date,
                        end_date = EXCLUDED.end_date,
                        variable = EXCLUDED.variable,
                        granularity = EXCLUDED.granularity,
                        file_format = EXCLUDED.file_format,
                        bytes = EXCLUDED.bytes,
                        downloaded_at = EXCLUDED.downloaded_at,
                        has_local_handle = EXCLUDED.has_local_handle,
                        saved_with_picker = EXCLUDED.saved_with_picker,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        record["id"],
                        user_id,
                        record["filename"],
                        record["siteId"],
                        record["site"],
                        record["startDate"],
                        record["endDate"],
                        record["variable"],
                        record["granularity"],
                        record["format"],
                        record["bytes"],
                        record["downloadedAt"],
                        record["hasLocalHandle"],
                        record["savedWithPicker"],
                        Json(record),
                    ),
                )

            return record

    def list_downloads(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 100))

        with self.lock, self._connection() as conn:
            if not self._get_user_row(conn, user_id):
                raise UserStoreError("user_not_found")

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM download_records
                    WHERE user_id = %s
                    ORDER BY downloaded_at DESC
                    LIMIT %s
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()

        return [self._download_row_to_record(dict(row)) for row in rows]

    def _download_row_to_record(self, row: dict[str, Any]) -> dict[str, Any]:
        metadata = _json_object(row.get("metadata"))
        return {
            **metadata,
            "id": row["id"],
            "filename": row["filename"],
            "siteId": row["site_id"],
            "site": row["site_name"],
            "startDate": row["start_date"],
            "endDate": row["end_date"],
            "variable": row["variable"],
            "granularity": row["granularity"],
            "format": row["file_format"],
            "bytes": int(row["bytes"] or 0),
            "downloadedAt": row["downloaded_at"],
            "hasLocalHandle": bool(row["has_local_handle"]),
            "savedWithPicker": bool(row["saved_with_picker"]),
        }

    def import_json_file(self, path: Path) -> int:
        path = Path(path)
        if not path.exists():
            return 0

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 0

        users = data.get("users") if isinstance(data, dict) else {}
        if not isinstance(users, dict) or not users:
            return 0

        with self.lock, self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM app_users")
                if int(cur.fetchone()[0] or 0) > 0:
                    return 0

                imported = 0
                for user_id, user in users.items():
                    if not isinstance(user, dict):
                        continue

                    user_id = str(user.get("id") or user_id)
                    email = str(user.get("email") or "").strip().lower()
                    if not user_id or not email:
                        continue

                    preferences = _normalize_preferences(user.get("preferences"), self.sites_by_id)
                    now = _iso_now()
                    cur.execute(
                        """
                        INSERT INTO app_users (
                            id, email, name, password_hash, created_at, updated_at,
                            preferences, alert_state
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (
                            user_id,
                            email,
                            str(user.get("name") or email.split("@", 1)[0]),
                            str(user.get("password_hash") or ""),
                            str(user.get("created_at") or now),
                            str(user.get("updated_at") or now),
                            Json(preferences),
                            Json(_json_object(user.get("alert_state"))),
                        ),
                    )
                    imported += cur.rowcount

                    for device in user.get("devices", []) or []:
                        token = str((device or {}).get("token") or "").strip()
                        if not token:
                            continue
                        cur.execute(
                            """
                            INSERT INTO push_devices (
                                token, user_id, created_at, last_seen_at, user_agent, platform
                            )
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (token) DO UPDATE SET
                                user_id = EXCLUDED.user_id,
                                last_seen_at = EXCLUDED.last_seen_at,
                                user_agent = EXCLUDED.user_agent,
                                platform = EXCLUDED.platform
                            """,
                            (
                                token,
                                user_id,
                                str(device.get("created_at") or now),
                                str(device.get("last_seen_at") or now),
                                str(device.get("user_agent") or "")[:180],
                                str(device.get("platform") or "")[:80],
                            ),
                        )

                for token, session in (data.get("sessions") or {}).items():
                    if not isinstance(session, dict):
                        continue
                    user_id = str(session.get("user_id") or "")
                    if not user_id:
                        continue
                    cur.execute(
                        """
                        INSERT INTO user_sessions (token, user_id, created_at, expires_at)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (token) DO NOTHING
                        """,
                        (
                            str(token),
                            user_id,
                            str(session.get("created_at") or _iso_now()),
                            str(session.get("expires_at") or _iso_now()),
                        ),
                    )

                for user_id, downloads in (data.get("downloads") or {}).items():
                    if not isinstance(downloads, list):
                        continue
                    for download in downloads:
                        if not isinstance(download, dict):
                            continue
                        record = _normalize_download_record(download, self.sites_by_id)
                        cur.execute(
                            """
                            INSERT INTO download_records (
                                id, user_id, filename, site_id, site_name, start_date, end_date,
                                variable, granularity, file_format, bytes, downloaded_at,
                                has_local_handle, saved_with_picker, metadata
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING
                            """,
                            (
                                record["id"],
                                str(user_id),
                                record["filename"],
                                record["siteId"],
                                record["site"],
                                record["startDate"],
                                record["endDate"],
                                record["variable"],
                                record["granularity"],
                                record["format"],
                                record["bytes"],
                                record["downloadedAt"],
                                record["hasLocalHandle"],
                                record["savedWithPicker"],
                                Json(record),
                            ),
                        )

                return imported

    def _alert_now(self) -> datetime:
        try:
            from zoneinfo import ZoneInfo

            return datetime.now(ZoneInfo(env_value("ALERT_TIMEZONE", "Europe/Madrid")))
        except Exception:
            return datetime.now()

    def _session_is_expired(self, session: dict[str, Any]) -> bool:
        try:
            expires = datetime.fromisoformat(session["expires_at"].replace("Z", ""))
            return _utc_now() > expires
        except Exception:
            return True

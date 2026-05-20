import hashlib
import hmac
import json
import os
import re
import secrets
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Optional
from zoneinfo import ZoneInfo


BASE_DIR = Path(__file__).resolve().parent
USERS_FILE = Path(os.getenv("USERS_FILE", BASE_DIR / "users.json")).resolve()
SESSION_DAYS = int(os.getenv("SESSION_DAYS", "30"))
ALERT_TIMEZONE = os.getenv("ALERT_TIMEZONE", "Europe/Madrid")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
VALID_CHANNELS = {"push", "email"}
VALID_THEMES = {"dark", "light"}

DEFAULT_PREFERENCES = {
    "notification_channel": "push",
    "alert_time": "08:00",
    "theme": "dark",
    "sites": [],
}


class UserStoreError(ValueError):
    pass


def _utc_now() -> datetime:
    return datetime.utcnow().replace(microsecond=0)


def _iso_now() -> str:
    return _utc_now().isoformat() + "Z"


def _normalize_email(email: str) -> str:
    normalized = (email or "").strip().lower()
    if not EMAIL_RE.match(normalized):
        raise UserStoreError("email_invalid")
    return normalized


def _normalize_name(name: str, email: str) -> str:
    clean_name = " ".join((name or "").strip().split())
    return clean_name[:80] or email.split("@", 1)[0]


def _normalize_alert_time(value: str) -> str:
    raw = (value or "").strip()
    parts = raw.split(":")

    if len(parts) < 2:
        raise UserStoreError("alert_time_invalid")

    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError as exc:
        raise UserStoreError("alert_time_invalid") from exc

    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise UserStoreError("alert_time_invalid")

    return f"{hour:02d}:{minute:02d}"


def _hash_password(password: str) -> str:
    if len(password or "") < 8:
        raise UserStoreError("password_too_short")

    iterations = 200_000
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("ascii"),
        iterations,
    ).hex()

    return f"pbkdf2_sha256${iterations}${salt}${digest}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt, expected = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False

        digest = hashlib.pbkdf2_hmac(
            "sha256",
            (password or "").encode("utf-8"),
            salt.encode("ascii"),
            int(iterations),
        ).hex()

        return hmac.compare_digest(digest, expected)

    except Exception:
        return False


class UserStore:
    def __init__(self, path: Path = USERS_FILE, sites_by_id: Optional[dict[str, Any]] = None):
        self.path = Path(path).resolve()
        self.sites_by_id = sites_by_id or {}
        self.lock = RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _empty_data(self) -> dict[str, Any]:
        return {
            "version": 1,
            "users": {},
            "sessions": {},
        }

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            data = self._empty_data()
            self._save_unlocked(data)
            return data

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise UserStoreError("users_file_invalid_json") from exc

        if not isinstance(data, dict):
            data = self._empty_data()

        data.setdefault("version", 1)
        data.setdefault("users", {})
        data.setdefault("sessions", {})

        for user in data["users"].values():
            user.setdefault("id", secrets.token_urlsafe(10))
            user.setdefault("created_at", _iso_now())
            user.setdefault("updated_at", user["created_at"])
            user.setdefault("devices", [])
            user.setdefault("alert_state", {})

            preferences = {
                **DEFAULT_PREFERENCES,
                **(user.get("preferences") or {}),
            }
            preferences["notification_channel"] = (
                preferences["notification_channel"]
                if preferences["notification_channel"] in VALID_CHANNELS
                else DEFAULT_PREFERENCES["notification_channel"]
            )
            preferences["theme"] = (
                preferences["theme"]
                if preferences["theme"] in VALID_THEMES
                else DEFAULT_PREFERENCES["theme"]
            )
            preferences["alert_time"] = _normalize_alert_time(preferences["alert_time"])
            preferences["sites"] = self._valid_sites(preferences.get("sites", []))
            user["preferences"] = preferences

        return data

    def _save_unlocked(self, data: dict[str, Any]) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_path, self.path)

    def _valid_sites(self, sites: Any) -> list[str]:
        if not isinstance(sites, list):
            return []

        valid = []
        for site_id in sites:
            site_id = str(site_id).strip()
            if not site_id:
                continue
            if self.sites_by_id and site_id not in self.sites_by_id:
                continue
            if site_id not in valid:
                valid.append(site_id)

        return valid

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

    def create_user(self, name: str, email: str, password: str) -> dict[str, Any]:
        with self.lock:
            data = self._load_unlocked()
            email = _normalize_email(email)

            for existing in data["users"].values():
                if existing.get("email") == email:
                    raise UserStoreError("email_already_registered")

            user_id = secrets.token_urlsafe(12)
            now = _iso_now()
            user = {
                "id": user_id,
                "name": _normalize_name(name, email),
                "email": email,
                "password_hash": _hash_password(password),
                "created_at": now,
                "updated_at": now,
                "preferences": deepcopy(DEFAULT_PREFERENCES),
                "devices": [],
                "alert_state": {},
            }

            data["users"][user_id] = user
            self._save_unlocked(data)
            return self._public_user(user)

    def authenticate(self, email: str, password: str) -> Optional[dict[str, Any]]:
        with self.lock:
            data = self._load_unlocked()
            email = _normalize_email(email)

            for user in data["users"].values():
                if user.get("email") == email and _verify_password(password, user.get("password_hash", "")):
                    return self._public_user(user)

        return None

    def create_session(self, user_id: str) -> str:
        with self.lock:
            data = self._load_unlocked()
            if user_id not in data["users"]:
                raise UserStoreError("user_not_found")

            self._cleanup_expired_sessions_unlocked(data)

            token = secrets.token_urlsafe(32)
            expires_at = (_utc_now() + timedelta(days=SESSION_DAYS)).isoformat() + "Z"
            data["sessions"][token] = {
                "user_id": user_id,
                "created_at": _iso_now(),
                "expires_at": expires_at,
            }
            self._save_unlocked(data)
            return token

    def delete_session(self, session_token: str) -> None:
        if not session_token:
            return

        with self.lock:
            data = self._load_unlocked()
            data["sessions"].pop(session_token, None)
            self._save_unlocked(data)

    def get_user_by_session(self, session_token: str) -> Optional[dict[str, Any]]:
        if not session_token:
            return None

        with self.lock:
            data = self._load_unlocked()
            session = data["sessions"].get(session_token)

            if not session or self._session_is_expired(session):
                if session:
                    data["sessions"].pop(session_token, None)
                    self._save_unlocked(data)
                return None

            user = data["users"].get(session.get("user_id"))
            return deepcopy(user) if user else None

    def get_public_user_by_session(self, session_token: str) -> Optional[dict[str, Any]]:
        return self._public_user(self.get_user_by_session(session_token))

    def get_user(self, user_id: str) -> Optional[dict[str, Any]]:
        with self.lock:
            data = self._load_unlocked()
            user = data["users"].get(user_id)
            return deepcopy(user) if user else None

    def list_users(self) -> list[dict[str, Any]]:
        with self.lock:
            data = self._load_unlocked()
            return [deepcopy(user) for user in data["users"].values()]

    def update_preferences(self, user_id: str, preferences: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            data = self._load_unlocked()
            user = data["users"].get(user_id)

            if not user:
                raise UserStoreError("user_not_found")

            current = {
                **DEFAULT_PREFERENCES,
                **(user.get("preferences") or {}),
            }

            if "notification_channel" in preferences:
                channel = str(preferences["notification_channel"]).strip().lower()
                if channel not in VALID_CHANNELS:
                    raise UserStoreError("notification_channel_invalid")
                current["notification_channel"] = channel

            if "alert_time" in preferences:
                current["alert_time"] = _normalize_alert_time(str(preferences["alert_time"]))

            if "theme" in preferences:
                theme = str(preferences["theme"]).strip().lower()
                if theme not in VALID_THEMES:
                    raise UserStoreError("theme_invalid")
                current["theme"] = theme

            if "sites" in preferences:
                current["sites"] = self._valid_sites(preferences["sites"])

            user["preferences"] = current
            user["updated_at"] = _iso_now()
            self._save_unlocked(data)

            return self._public_user(user)

    def update_profile(
        self,
        user_id: str,
        name: Optional[str] = None,
        preferences: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        with self.lock:
            data = self._load_unlocked()
            user = data["users"].get(user_id)

            if not user:
                raise UserStoreError("user_not_found")

            if name is not None:
                user["name"] = _normalize_name(str(name), user.get("email", ""))

            if preferences is not None:
                current = {
                    **DEFAULT_PREFERENCES,
                    **(user.get("preferences") or {}),
                }

                if "notification_channel" in preferences:
                    channel = str(preferences["notification_channel"]).strip().lower()
                    if channel not in VALID_CHANNELS:
                        raise UserStoreError("notification_channel_invalid")
                    current["notification_channel"] = channel

                if "alert_time" in preferences:
                    current["alert_time"] = _normalize_alert_time(str(preferences["alert_time"]))

                if "theme" in preferences:
                    theme = str(preferences["theme"]).strip().lower()
                    if theme not in VALID_THEMES:
                        raise UserStoreError("theme_invalid")
                    current["theme"] = theme

                if "sites" in preferences:
                    current["sites"] = self._valid_sites(preferences["sites"])

                user["preferences"] = current

            user["updated_at"] = _iso_now()
            self._save_unlocked(data)

            return self._public_user(user)

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

        with self.lock:
            data = self._load_unlocked()
            user = data["users"].get(user_id)
            if not user:
                raise UserStoreError("user_not_found")

            valid_sites = self._valid_sites(sites)

            for existing_user in data["users"].values():
                existing_user["devices"] = [
                    device
                    for device in existing_user.get("devices", [])
                    if device.get("token") != token
                ]

            if valid_sites:
                now = _iso_now()
                user.setdefault("devices", []).append({
                    "token": token,
                    "created_at": now,
                    "last_seen_at": now,
                    "user_agent": (user_agent or "")[:180],
                    "platform": (platform or "")[:80],
                })

            preferences = {
                **DEFAULT_PREFERENCES,
                **(user.get("preferences") or {}),
                "sites": valid_sites,
            }
            user["preferences"] = preferences
            user["updated_at"] = _iso_now()
            self._save_unlocked(data)

            return self._public_user(user)

    def remove_push_subscription(self, user_id: str, token: str) -> dict[str, Any]:
        token = (token or "").strip()

        with self.lock:
            data = self._load_unlocked()
            user = data["users"].get(user_id)
            if not user:
                raise UserStoreError("user_not_found")

            user["devices"] = [
                device
                for device in user.get("devices", [])
                if device.get("token") != token
            ]
            user["updated_at"] = _iso_now()
            self._save_unlocked(data)

            return self._public_user(user)

    def remove_invalid_tokens(self, invalid_tokens: set[str]) -> int:
        invalid_tokens = set(t for t in (invalid_tokens or set()) if t)
        if not invalid_tokens:
            return 0

        with self.lock:
            data = self._load_unlocked()
            removed = 0

            for user in data["users"].values():
                before = len(user.get("devices", []))
                user["devices"] = [
                    device
                    for device in user.get("devices", [])
                    if device.get("token") not in invalid_tokens
                ]
                removed += before - len(user["devices"])

            if removed:
                self._save_unlocked(data)

            return removed

    def users_due_for_alert(self, force: bool = False, user_id: Optional[str] = None) -> list[dict[str, Any]]:
        with self.lock:
            data = self._load_unlocked()
            now = self._alert_now()
            today = now.strftime("%Y-%m-%d")
            current_time = now.strftime("%H:%M")
            due = []

            for user in data["users"].values():
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
                    if preferences.get("alert_time") != current_time:
                        continue

                    alert_state = user.get("alert_state") or {}
                    last_result = alert_state.get("last_result") or {}
                    try:
                        last_sent = int(last_result.get("sent", 0) or 0)
                    except (TypeError, ValueError):
                        last_sent = 0

                    if alert_state.get("last_sent_date") == today and last_sent > 0:
                        continue

                due.append(deepcopy(user))

            return due

    def mark_alert_result(self, user_id: str, result: dict[str, Any]) -> None:
        with self.lock:
            data = self._load_unlocked()
            user = data["users"].get(user_id)
            if not user:
                return

            now = self._alert_now()
            user["alert_state"] = {
                "last_sent_date": now.strftime("%Y-%m-%d"),
                "last_sent_at": _iso_now(),
                "last_channel": (user.get("preferences") or {}).get("notification_channel"),
                "last_result": {
                    "sent": int(result.get("sent", 0)),
                    "processed_sites": int(result.get("processed_sites", 0)),
                    "errors": list(result.get("errors", []))[:5],
                },
            }
            user["updated_at"] = _iso_now()
            self._save_unlocked(data)

    def token_site_map(self) -> dict[str, set[str]]:
        mapping: dict[str, set[str]] = {}

        with self.lock:
            data = self._load_unlocked()

            for user in data["users"].values():
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

    def _alert_now(self) -> datetime:
        try:
            return datetime.now(ZoneInfo(ALERT_TIMEZONE))
        except Exception:
            return datetime.now()

    def _session_is_expired(self, session: dict[str, Any]) -> bool:
        try:
            expires = datetime.fromisoformat(session["expires_at"].replace("Z", ""))
            return _utc_now() > expires
        except Exception:
            return True

    def _cleanup_expired_sessions_unlocked(self, data: dict[str, Any]) -> None:
        expired_tokens = [
            token
            for token, session in data.get("sessions", {}).items()
            if self._session_is_expired(session)
        ]

        for token in expired_tokens:
            data["sessions"].pop(token, None)

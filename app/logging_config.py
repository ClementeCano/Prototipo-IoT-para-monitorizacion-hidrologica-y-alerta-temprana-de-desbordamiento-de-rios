import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any


SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|apikey|password|passwd|token|secret|credential|authorization)(=|:)\s*([^&\s,;}]+)"),
    re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]{12,}"),
]


def redact(value: Any) -> Any:
    if isinstance(value, str):
        text = value
        for pattern in SECRET_PATTERNS:
            if pattern.groups >= 3:
                text = pattern.sub(r"\1\2***", text)
            else:
                text = pattern.sub(r"\1***", text)
        return text

    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if any(word in str(key).lower() for word in ("password", "token", "secret", "credential", "apikey", "api_key")):
                redacted[key] = "***"
            else:
                redacted[key] = redact(item)
        return redacted

    if isinstance(value, (list, tuple, set)):
        return [redact(item) for item in value]

    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }

        for key in (
            "event",
            "site_id",
            "user_id",
            "channel",
            "status",
            "duration_ms",
            "count",
        ):
            if hasattr(record, key):
                payload[key] = redact(getattr(record, key))

        if record.exc_info:
            payload["exc_info"] = redact(self.formatException(record.exc_info))

        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    root = logging.getLogger()
    if getattr(root, "_dashboard_ebro_configured", False):
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root._dashboard_ebro_configured = True

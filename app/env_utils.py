import logging
import os
from typing import Any


logger = logging.getLogger(__name__)


def env_value(name: str, default: Any = None) -> Any:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = str(raw).strip()

    # Fly secrets must store only the value, but a common mistake is pasting
    # the full .env line as the secret value: KEY=VALUE.
    if "=" in value:
        left, right = value.split("=", 1)
        if left.strip().upper() == name.upper():
            logger.warning("ENV %s contiene formato KEY=VALUE; se usara solo VALUE", name)
            value = right.strip()

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()

    return value


def env_int(name: str, default: int) -> int:
    value = env_value(name, None)
    if value in (None, ""):
        return int(default)

    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        logger.warning("ENV %s=%r no es un entero valido; usando %r", name, value, default)
        return int(default)


def env_float(name: str, default: float) -> float:
    value = env_value(name, None)
    if value in (None, ""):
        return float(default)

    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        logger.warning("ENV %s=%r no es un numero valido; usando %r", name, value, default)
        return float(default)


def env_bool(name: str, default: bool = False) -> bool:
    value = env_value(name, None)
    if value in (None, ""):
        return bool(default)

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    logger.warning("ENV %s=%r no es booleano valido; usando %r", name, value, default)
    return bool(default)

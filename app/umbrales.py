import logging
from pathlib import Path

try:
    from app.logging_config import configure_logging
except ImportError:
    from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

def cargar_umbrales(path=None):
    umbrales = {}

    if path is None:
        path = BASE_DIR / "umbrales.txt"

    path = Path(path)

    logger.debug("Ruta umbrales: %s", path)
    logger.debug("Existe umbrales: %s", path.exists())

    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()

            if not linea:
                continue

            if "->" not in linea:
                logger.warning("Linea de umbrales invalida: %s", linea)
                continue

            municipio, valores = linea.split("->", 1)

            municipio = municipio.strip().lower()

            niveles = [v.strip() for v in valores.split(",")]

            umbrales[municipio] = {
                "amarillo": float(niveles[0]) if niveles[0] != "-" else None,
                "naranja": float(niveles[1]) if niveles[1] != "-" else None,
                "rojo": float(niveles[2]) if niveles[2] != "-" else None,
            }

    return umbrales

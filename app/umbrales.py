from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def cargar_umbrales(path=None):
    umbrales = {}

    if path is None:
        path = BASE_DIR / "umbrales.txt"

    path = Path(path)

    print("📁 Ruta umbrales:", path)
    print("✅ Existe:", path.exists())

    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()

            if not linea or linea.startswith("#"):
                continue

            municipio, valor = linea.split("=")
            umbrales[municipio.strip().lower()] = float(valor.strip())

    return umbrales
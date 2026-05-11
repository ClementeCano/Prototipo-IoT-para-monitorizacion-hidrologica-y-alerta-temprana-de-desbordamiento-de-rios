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

            if not linea:
                continue

            if "->" not in linea:
                print(f"⚠️ Línea inválida: {linea}")
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
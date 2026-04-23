import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
SITES = json.loads((BASE_DIR / "sites.json").read_text(encoding="utf-8"))

import os

# print("📂 CONTENIDO /app:")
# for root, dirs, files in os.walk("/app"):
#     print(root, files)

def collect_all_tags():
    tags = []

    for s in SITES:
        nivel = (s.get("saih") or {}).get("nivel", "") or ""
        caudal = (s.get("saih") or {}).get("caudal", "") or ""

        if nivel:
            tags.append(nivel)
        if caudal:
            tags.append(caudal)

    return list(set(tags))
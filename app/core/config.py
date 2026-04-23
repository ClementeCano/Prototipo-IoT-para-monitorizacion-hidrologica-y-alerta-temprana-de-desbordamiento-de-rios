import json
from pathlib import Path

SITES = json.loads(Path("sites.json").read_text(encoding="utf-8"))

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
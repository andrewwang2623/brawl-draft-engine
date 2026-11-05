"""Fetch Brawlify metadata (via the api.brawlapi.com mirror) and write compact
meta/brawlers.json and meta/maps.json used by the model and UI.

    .venv/bin/python build_meta.py            # re-download + rebuild
    .venv/bin/python build_meta.py --offline  # rebuild from meta/*_raw.json
"""
import json
import sys
from pathlib import Path

import requests

META = Path(__file__).resolve().parent / "meta"
BASE = "https://api.brawlapi.com/v1/"


def fetch(name):
    r = requests.get(BASE + name, headers={"User-Agent": "brawl-draft-engine/0.1"}, timeout=30)
    r.raise_for_status()
    (META / f"{name}_raw.json").write_text(r.text)


def main():
    META.mkdir(exist_ok=True)
    if "--offline" not in sys.argv:
        for n in ("brawlers", "maps", "gamemodes"):
            fetch(n)
    raw_b = json.loads((META / "brawlers_raw.json").read_text())["list"]
    brawlers = {}
    for b in raw_b:
        if not b.get("released", True):
            continue
        brawlers[str(b["id"])] = {
            "id": b["id"],
            "name": b["name"],
            "class": (b.get("class") or {}).get("name") or "Unknown",
            "rarity": (b.get("rarity") or {}).get("name"),
            "image": b.get("imageUrl2") or b.get("imageUrl"),
        }
    raw_m = json.loads((META / "maps_raw.json").read_text())["list"]
    maps = {}
    for m in raw_m:
        gm = m.get("gameMode") or {}
        env = m.get("environment") or {}
        maps[str(m["id"])] = {
            "id": m["id"], "name": m["name"], "mode": gm.get("name"),
            "environment": env.get("name"), "disabled": m.get("disabled"),
            "image": m.get("imageUrl"),
        }
    (META / "brawlers.json").write_text(json.dumps(brawlers, indent=1, ensure_ascii=False))
    (META / "maps.json").write_text(json.dumps(maps, indent=1, ensure_ascii=False))
    print(f"{len(brawlers)} brawlers, {len(maps)} maps")


if __name__ == "__main__":
    main()

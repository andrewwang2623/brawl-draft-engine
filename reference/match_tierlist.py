"""Read AshBS's tier-list image: identify each icon and its arrow marker.

Each tier row is template-matched against official Brawlify brawler icons ("emoji" pin
style, which is what the image uses; "borderless" as a fallback for brawlers without one).
Icons are assigned greedily by score without overlap, then sorted left to right. An arrow
marker sits over the top-right of the icon it belongs to, and is classified by color:
red = nerf, green = buff, rainbow (multi-hue) = new buffie / special.

    .venv/bin/python reference/match_tierlist.py     # writes reference/tierlist_detected.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMG = HERE / "ashbs_tierlist_nov2026.webp"
ICON_DIR = HERE / ".icons"
OUT = HERE / "tierlist_detected.json"

# Tier rows as y-bands in the 1650x2000 image (measured from the image).
ROWS = [("S", 290, 440), ("A", 520, 660), ("A", 655, 800), ("B", 825, 965), ("B", 960, 1095),
        ("C", 1110, 1250), ("C", 1250, 1390), ("D", 1470, 1610), ("F", 1750, 1890)]
X0 = 300                       # icons start right of the tier badge
SCALES = (76, 82, 88, 94)      # icon size in px
MIN_SCORE = 0.45
MIN_SEP = 48                   # px between icon centers in one row


def fetch_icons(brawlers):
    ICON_DIR.mkdir(exist_ok=True)
    for b in brawlers:
        for style in ("emoji", "borderless"):
            p = ICON_DIR / f"{style}_{b['id']}.png"
            if p.exists():
                continue
            r = requests.get(f"https://cdn.brawlify.com/brawlers/{style}/{b['id']}.png",
                             headers={"User-Agent": "brawl-draft-engine/0.1"}, timeout=20)
            if r.status_code == 200:
                p.write_bytes(r.content)


def load_template(bid):
    """BGR template + alpha mask, cropped to the central face region."""
    for style in ("emoji", "borderless"):
        p = ICON_DIR / f"{style}_{bid}.png"
        if p.exists():
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if im.shape[2] == 3:
                im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
            if style == "borderless":       # square portrait: crop toward the face
                h, w = im.shape[:2]
                im = im[int(h * .05):int(h * .85), int(w * .1):int(w * .9)]
            return im, style
    return None, None


def arrow_color(img, x, y, size):
    """Classify the arrow marker over the top-right of an icon centered at (x, y)."""
    x1, x2 = int(x + size * 0.05), int(x + size * 0.75)
    y1, y2 = int(y - size * 1.05), int(y - size * 0.25)
    patch = img[max(0, y1):max(0, y2), max(0, x1):x2]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
    vivid = (s > 150) & (v > 170)
    n = patch.shape[0] * patch.shape[1]
    red = (vivid & ((h < 6) | (h > 174))).sum() / n
    green = (vivid & (h > 50) & (h < 75)).sum() / n
    pastel = (s > 60) & (s < 170) & (v > 200)
    hues = h[pastel]
    rainbow = 0.0
    if hues.size > 0.04 * n:
        hist = np.histogram(hues, bins=6, range=(0, 180))[0]
        rainbow = (hist > 0.08 * hues.size).sum()
    if red > 0.06:
        return "red"
    if green > 0.06:
        return "green"
    if rainbow >= 4:
        return "rainbow"
    return None


def main():
    brawlers = json.loads((ROOT / "meta" / "official_brawlers.json").read_text())
    fetch_icons(brawlers)
    img = cv2.cvtColor(np.array(__import__("PIL.Image", fromlist=["Image"]).open(IMG).convert("RGB")),
                       cv2.COLOR_RGB2BGR)
    templates = {}
    for b in brawlers:
        t, style = load_template(b["id"])
        if t is not None:
            templates[b["name"]] = (t, style)

    candidates = []   # (score, row_i, x_center, y_center, size, name)
    for ri, (tier, y1, y2) in enumerate(ROWS):
        band = img[y1:y2, X0:]
        for name, (t, style) in templates.items():
            for s in SCALES:
                tt = cv2.resize(t, (s, s), interpolation=cv2.INTER_AREA)
                bgr, alpha = tt[..., :3], tt[..., 3]
                # match only the central 70% (neighbors overlap the edges)
                c0, c1 = int(s * .15), int(s * .85)
                bgr_c, mask_c = bgr[c0:c1, c0:c1], alpha[c0:c1, c0:c1]
                if band.shape[0] < bgr_c.shape[0]:
                    continue
                res = cv2.matchTemplate(band, bgr_c, cv2.TM_CCOEFF_NORMED,
                                        mask=(mask_c > 128).astype(np.uint8) * 255)
                res = np.nan_to_num(res, nan=-1, posinf=-1, neginf=-1)
                # keep the top few peaks per (template, scale, row)
                for _ in range(2):
                    _, mx, _, loc = cv2.minMaxLoc(res)
                    if mx < MIN_SCORE:
                        break
                    cx = X0 + loc[0] + (c1 - c0) / 2
                    cy = y1 + loc[1] + (c1 - c0) / 2
                    candidates.append((float(mx), ri, cx, cy, s, name, style))
                    xa, xb = max(0, loc[0] - 30), loc[0] + 30
                    res[:, xa:xb] = -1

    candidates.sort(reverse=True)
    used_names, placed = set(), []
    for sc, ri, cx, cy, s, name, style in candidates:
        if name in used_names:
            continue
        if any(p["row"] == ri and abs(p["x"] - cx) < MIN_SEP for p in placed):
            continue
        used_names.add(name)
        placed.append({"row": ri, "tier": ROWS[ri][0], "x": cx, "y": cy, "size": s,
                       "name": name, "score": round(sc, 3), "template": style})

    # second-best alternative per placed icon (for confidence)
    for p in placed:
        alts = [c for c in candidates if c[1] == p["row"] and abs(c[2] - p["x"]) < MIN_SEP / 2
                and c[5] != p["name"]]
        p["runner_up"] = alts[0][5] if alts else None
        p["runner_up_score"] = round(alts[0][0], 3) if alts else None
        p["arrow"] = arrow_color(img, p["x"], p["y"], p["size"])
    placed.sort(key=lambda p: (p["row"], p["x"]))
    OUT.write_text(json.dumps(placed, indent=1))
    missing = sorted(set(templates) - used_names)
    print(f"placed {len(placed)} icons; not found: {missing}")
    for ri, (tier, *_r) in enumerate(ROWS):
        row = [p for p in placed if p["row"] == ri]
        print(tier, " ".join(f"{p['name']}({p['score']:.2f}{',' + p['arrow'] if p['arrow'] else ''})"
                             for p in row))


if __name__ == "__main__":
    sys.exit(main())

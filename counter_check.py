"""Are learned pair counters over-shrunk? For known counters (checks/matchups.yaml) and the most
played cross-team pairs, compare what the model predicts in the games where a faced b with what
actually happened. A consistently positive residual (a's team wins more than predicted) means
the pair term C(a, b) is shrunk too far.

    .venv/bin/python counter_check.py [--top 50] [--model models/model_legendary.pkl]
"""
import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from data import load_matches
from model import DraftModel, model_path, sigmoid

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--model", default=str(model_path("legendary")))
    a = ap.parse_args()
    md = DraftModel.load(Path(a.model))
    m = load_matches()
    m = m.subset(np.flatnonzero(md.in_target(m)))
    p = md.predict_proba(m, rank_zero=False)          # full model incl. skill controls
    name = {int(b): v["name"] for b, v in md.meta.items()}
    nid = {v.upper(): k for k, v in name.items()}
    # per ordered pair (a beats b?): games, wins of a's side, predicted wins of a's side
    st = defaultdict(lambda: [0, 0.0, 0.0])
    for i in range(len(m)):
        A, B, y, pi = m.team_a[i], m.team_b[i], m.y[i], p[i]
        for x, z in itertools.product(A, B):
            s = st[(int(x), int(z))]; s[0] += 1; s[1] += y; s[2] += pi
            s = st[(int(z), int(x))]; s[0] += 1; s[1] += 1 - y; s[2] += 1 - pi
    checks = yaml.safe_load((ROOT / "checks" / "matchups.yaml").read_text()).get("checks", [])
    known = []
    for c in checks:
        if "counter" in c:
            vs = c["victim"] if isinstance(c["victim"], list) else [c["victim"]]
            known += [(nid[c["counter"].upper()], nid[v.upper()]) for v in vs]
    played = sorted({k for k in st if k[0] < k[1]}, key=lambda k: -st[k][0])[: a.top]
    mm_c = {}
    maps = md.known_maps()
    tot = sum(n for _, _, n in maps)
    for mode, mp, n in maps:                      # games-weighted average learned C over maps
        mm = md.map_model(mp, mode)
        for k in known + played:
            i, j = mm.id_to_idx[k[0]], mm.id_to_idx[k[1]]
            mm_c[k] = mm_c.get(k, 0.0) + n / tot * float(mm.C[i, j])

    def row(k):
        n, w, pw = st[k]
        obs, pred = w / n, pw / n
        se = np.sqrt(max(pred * (1 - pred), 1e-6) / n)
        return n, obs, pred, (obs - pred), (obs - pred) / se

    print(f"{'pair':<26}{'games':>7}{'learned C':>11}{'obs':>7}{'pred':>7}{'resid':>8}{'z':>6}")
    out = {"known": [], "played": []}
    for label, keys in (("known counters (checks/matchups.yaml)", known), (f"top {a.top} most-played pairs", played)):
        print(f"-- {label}")
        zs = []
        for k in keys:
            n, obs, pred, res, z = row(k)
            zs.append(z)
            print(f"{name[k[0]] + ' > ' + name[k[1]]:<26}{n:>7}{25 * mm_c[k]:>+10.1f}%{100 * obs:>6.1f}%{100 * pred:>6.1f}%"
                  f"{100 * res:>+7.1f}%{z:>6.1f}")
            out["known" if keys is known else "played"].append(
                {"pair": f"{name[k[0]]} > {name[k[1]]}", "games": n, "learned_c_pct": round(25 * mm_c[k], 2),
                 "obs": round(100 * obs, 1), "pred": round(100 * pred, 1), "resid_pct": round(100 * res, 2), "z": round(z, 2)})
        zs = np.array(zs)
        print(f"   mean z {zs.mean():+.2f}  (|z|>2: {(np.abs(zs) > 2).sum()}/{len(zs)})")
    return out


if __name__ == "__main__":
    main()

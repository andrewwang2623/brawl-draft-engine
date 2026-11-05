"""Bracket model vs all-ranks model on a bracket-only holdout.

    .venv/bin/python compare_brackets.py --tier 16    # Legendary+
    .venv/bin/python compare_brackets.py --tier 19    # Masters+
"""
import argparse
import json
from pathlib import Path

from data import load_matches, tier_name
from evaluate import run

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=16)
    args = ap.parse_args()
    base = json.loads((ROOT / "params.json").read_text()) if (ROOT / "params.json").exists() else {}
    m = load_matches()
    t = args.tier
    rows = []
    cfgs = [("all-ranks model (all games equal weight)", dict(base, bracket_min_tier=0))]
    for s in (1.0, 3.0, 10.0):
        cfgs.append((f"{tier_name(t)}+ model, low tiers as weak prior (w=0.25), dev penalty x{s:g}",
                     dict(base, bracket_min_tier=t, low_tier_weight=0.25, lam_dev_scale=s)))
    cfgs.append((f"{tier_name(t)}+ games only", dict(base, bracket_min_tier=t, low_tier_weight=0.0,
                                                      lam_dev_scale=3.0)))
    for name, p in cfgs:
        r = run(p, matches=m, verbose=False, holdout_min_tier=t)
        d = r["model_draft_only"]; f = r["model_with_rank"]
        rows.append((name, r, d, f))
        print(f"{name:<70} draft-only ll={d['log_loss']:.4f} acc={d['accuracy']:.3f} | "
              f"full ll={f['log_loss']:.4f} acc={f['accuracy']:.3f}")
    r0 = rows[0][1]
    n_in = int((m.tier_min >= t).sum())
    print(f"games={len(m)} in-bracket={n_in} holdout={r0['n_test']} "
          f"baselines: coin={r0['coin_flip']['log_loss']:.4f} map_wr={r0['map_winrate']['log_loss']:.4f}")
    return rows, len(m), n_in


if __name__ == "__main__":
    main()

"""Run ONE experiment: current params.json vs. a candidate change, same data snapshot.

    .venv/bin/python experiment.py --desc "lam_cnt 300 -> 100" --set lam_cnt=100
    .venv/bin/python experiment.py --desc "..." --set k=v --apply   # keep if it wins

Appends a row to EXPERIMENTS.md. With --apply, writes the change to params.json if the
holdout draft-only log loss improves. Also reports a paired-bootstrap standard error of
the log-loss difference, so tiny "wins" can be read as noise.
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from data import load_matches, time_split
from evaluate import baseline_map_wr, metrics, parse_sets
from model import DraftModel

ROOT = Path(__file__).resolve().parent


def holdout_eval(params, train, test):
    md = DraftModel(params).fit(train)
    p = np.clip(md.predict_proba(test, rank_zero=True), 1e-6, 1 - 1e-6)
    pf = md.predict_proba(test, rank_zero=False)
    ll_i = -(test.y * np.log(p) + (1 - test.y) * np.log(1 - p))
    return metrics(p, test.y), metrics(pf, test.y), ll_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", required=True)
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    pf = ROOT / "params.json"
    base = json.loads(pf.read_text()) if pf.exists() else {}
    cand = dict(base, **parse_sets(args.set))

    m = load_matches()
    train, test = time_split(m)
    bt = DraftModel(base).params["bracket_min_tier"]
    if bt:
        test = test.subset(np.flatnonzero(test.tier_min >= bt))
    base_train = train.subset(np.flatnonzero(train.tier_min >= bt)) if bt else train
    wr = metrics(baseline_map_wr(base_train, test), test.y)

    mb, fb, llb = holdout_eval(base, train, test)
    mc, fc, llc = holdout_eval(cand, train, test)
    diff = llc - llb
    rng = np.random.default_rng(0)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(500)]
    se = float(np.std(boots))
    improved = mc["log_loss"] < mb["log_loss"]
    kept = improved and args.apply
    print(f"games={len(m)} holdout={len(test)}")
    print(f"baseline : ll={mb['log_loss']:.4f} brier={mb['brier']:.4f} acc={mb['accuracy']:.3f}")
    print(f"candidate: ll={mc['log_loss']:.4f} brier={mc['brier']:.4f} acc={mc['accuracy']:.3f}")
    print(f"delta ll={diff.mean():+.5f} (bootstrap SE {se:.5f}); map-WR baseline {wr['log_loss']:.4f}")
    if kept:
        pf.write_text(json.dumps(cand, indent=1) + "\n")
        print("KEPT -> params.json updated")
    else:
        print("reverted" if args.apply else "(dry run)")
    now = datetime.datetime.now().strftime("%m-%d %H:%M")
    row = (f"| {now} | {len(m):,} | {args.desc} (Δll {diff.mean():+.4f} ± {se:.4f}; "
           f"prev {mb['log_loss']:.4f}) | {mc['log_loss']:.4f} / {mc['brier']:.4f} / {mc['accuracy']:.3f} "
           f"| {fc['log_loss']:.4f} | {wr['log_loss']:.4f} | {'**kept**' if kept else 'reverted'} |\n")
    if args.apply:
        with open(ROOT / "EXPERIMENTS.md", "a") as f:
            f.write(row)


if __name__ == "__main__":
    main()

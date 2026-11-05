"""Run ONE experiment: current params.json vs. a candidate change, same data snapshot.

    .venv/bin/python experiment.py --desc "lam_cnt 300 -> 100" --set lam_cnt=100
    .venv/bin/python experiment.py --desc "..." --set k=v --apply   # keep if it wins
    .venv/bin/python experiment.py --desc "..." --set k=v --snapshot 1793845455   # frozen data

Both models are trained on the same time split of the same snapshot (rows the crawler had
inserted by --snapshot; default: now). Reports holdout draft-only log loss with a
paired-bootstrap SE, AND runs audit.py on both models. The candidate is kept (with --apply)
only if log loss improves AND the audit doesn't regress (no more flagged maps / flags, no
lower matchup-check pass rate). An audit regression fails the experiment even if log loss
improves. Appends a row to EXPERIMENTS.md (with --apply).
"""
import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np

import audit
from data import load_matches, time_split
from evaluate import baseline_map_wr, metrics, parse_sets
from model import LEGACY_PARAMS, DraftModel, load_patch_config

ROOT = Path(__file__).resolve().parent


def holdout_eval(params, train, test):
    md = DraftModel(params).fit(train)
    p = np.clip(md.predict_proba(test, rank_zero=True), 1e-6, 1 - 1e-6)
    pf = md.predict_proba(test, rank_zero=False)
    ll_i = -(test.y * np.log(p) + (1 - test.y) * np.log(1 - p))
    return md, metrics(p, test.y), metrics(pf, test.y), ll_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--base-set", action="append", default=[],
                    help="override the baseline (e.g. legacy=true for the pre-fix model)")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--snapshot", type=int, default=None, help="max inserted_at (unix s)")
    args = ap.parse_args()
    pf = ROOT / "params.json"
    current = json.loads(pf.read_text()) if pf.exists() else {}
    # baseline = params.json (optionally overridden by --base-set); candidate = params.json + --set
    bo = parse_sets(args.base_set)
    base = dict(current, **LEGACY_PARAMS) if bo.pop("legacy", False) else dict(current)
    base.update(bo)
    co = parse_sets(args.set)
    cand = dict(current, **({**LEGACY_PARAMS, **co} if co.pop("legacy", False) else co))
    snap = args.snapshot or int(time.time())

    m = load_matches(max_inserted=snap)
    train, test = time_split(m)
    bt = DraftModel(base).params["bracket_min_tier"]
    if bt:
        test = test.subset(np.flatnonzero(test.tier_min >= bt))
    test = test.subset(np.flatnonzero(test.ts >= load_patch_config()["patch_ts"]))   # post-patch holdout
    base_train = train.subset(np.flatnonzero(train.tier_min >= bt)) if bt else train
    wr = metrics(baseline_map_wr(base_train, test), test.y)

    mdb, mb, fb, llb = holdout_eval(base, train, test)
    mdc, mc, fc, llc = holdout_eval(cand, train, test)
    diff = llc - llb
    rng = np.random.default_rng(0)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(500)]
    se = float(np.std(boots))
    ab, ac = audit.audit_model(mdb, train), audit.audit_model(mdc, train)
    improved = mc["log_loss"] < mb["log_loss"]
    audit_bad = audit.regressed(ab, ac)
    kept = improved and not audit_bad and args.apply
    print(f"games={len(m)} holdout={len(test)} snapshot={snap}")
    print(f"baseline : ll={mb['log_loss']:.4f} brier={mb['brier']:.4f} acc={mb['accuracy']:.3f}")
    print(f"candidate: ll={mc['log_loss']:.4f} brier={mc['brier']:.4f} acc={mc['accuracy']:.3f}")
    print(f"delta ll={diff.mean():+.5f} (bootstrap SE {se:.5f}); map-WR baseline {wr['log_loss']:.4f}")
    print(f"audit baseline : {audit.summary_line(ab)}")
    print(f"audit candidate: {audit.summary_line(ac)}")
    if audit_bad:
        print("AUDIT REGRESSION -> experiment fails")
    if kept:
        pf.write_text(json.dumps(cand, indent=1) + "\n")
        print("KEPT -> params.json updated")
    else:
        print("reverted" if args.apply else "(dry run)")
    now = datetime.datetime.now().strftime("%m-%d %H:%M")
    mp = lambda a: f"{a['matchups']['n_pass']}/{a['matchups']['n']}" if a.get("matchups") else "-"
    row = (f"| {now} | {len(m):,} | {args.desc} (Δll {diff.mean():+.4f} ± {se:.4f}; "
           f"prev {mb['log_loss']:.4f}; audit flagged maps {ab['n_flagged']}→{ac['n_flagged']}, "
           f"matchups {mp(ab)}→{mp(ac)}) | {mc['log_loss']:.4f} / {mc['brier']:.4f} / {mc['accuracy']:.3f} "
           f"| {fc['log_loss']:.4f} | {wr['log_loss']:.4f} | "
           f"{'**kept**' if kept else ('reverted (audit)' if audit_bad else 'reverted')} |\n")
    if args.apply:
        with open(ROOT / "EXPERIMENTS.md", "a") as f:
            f.write(row)


if __name__ == "__main__":
    main()

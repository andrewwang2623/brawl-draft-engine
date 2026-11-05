"""Evaluation harness: time-based holdout, baselines vs. the main model.

    .venv/bin/python evaluate.py                       # default params
    .venv/bin/python evaluate.py --set lam_map=40      # override params
    .venv/bin/python evaluate.py --json                # machine-readable output

The primary metric is the holdout log loss of the *draft-only* model (rank-difference
feature zeroed at prediction time), because that's what the draft engine sees.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from data import Matches, load_matches, time_split
from model import DraftModel, sigmoid

ROOT = Path(__file__).resolve().parent
EPS = 1e-6


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, EPS, 1 - EPS)
    return {
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p > 0.5) == (y == 1))),
    }


def baseline_coin(train: Matches, test: Matches) -> np.ndarray:
    return np.full(len(test), 0.5)


def baseline_map_wr(train: Matches, test: Matches, k: float | None = None) -> np.ndarray:
    """Per-map brawler win rates alone (smoothed toward 50% with k pseudo-games).

    logit P(A) = sum_{a in A} logit(wr_map(a)) - sum_{b in B} logit(wr_map(b))

    If k is None it's tuned on an inner time split of the training data, so the
    baseline is as strong as it can fairly be.
    """
    if k is None:
        inner_tr, inner_va = time_split(train, 0.15)
        grid = [5, 10, 20, 50, 100, 200, 500, 1000]
        k = min(grid, key=lambda kk: metrics(baseline_map_wr(inner_tr, inner_va, kk),
                                             inner_va.y)["log_loss"])
    wins, games = {}, {}
    for i in range(len(train)):
        mk = (train.mode[i], train.map[i])
        for team, won in ((train.team_a[i], train.y[i]), (train.team_b[i], 1 - train.y[i])):
            for x in team:
                key = (mk, int(x))
                wins[key] = wins.get(key, 0.0) + won
                games[key] = games.get(key, 0.0) + 1

    def lg(mk, x):
        key = (mk, int(x))
        wr = (wins.get(key, 0.0) + 0.5 * k) / (games.get(key, 0.0) + k)
        return np.log(wr / (1 - wr))

    z = np.array([
        sum(lg((test.mode[i], test.map[i]), x) for x in test.team_a[i])
        - sum(lg((test.mode[i], test.map[i]), x) for x in test.team_b[i])
        for i in range(len(test))])
    return sigmoid(z)


def run(params: dict | None = None, matches: Matches | None = None, test_frac: float = 0.15,
        verbose: bool = True, holdout_min_tier: int | None = None) -> dict:
    """Time split over ALL games; the test set is then restricted to the holdout bracket
    (lowest tier >= holdout_min_tier; defaults to the model's bracket)."""
    m = matches if matches is not None else load_matches()
    train, test = time_split(m, test_frac)
    model = DraftModel(params)
    hmt = model.params.get("bracket_min_tier", 0) if holdout_min_tier is None else holdout_min_tier
    if hmt and test.tier_min is not None:
        test = test.subset(np.flatnonzero(test.tier_min >= hmt))
    t0 = time.time()
    model.fit(train)
    fit_s = time.time() - t0
    # Baselines see only in-bracket training games (their best shot at the bracket).
    base_train = train.subset(np.flatnonzero(train.tier_min >= hmt)) if hmt else train
    res = {
        "n_matches": len(m), "n_train": len(train), "n_test": len(test),
        "holdout_min_tier": hmt,
        "n_maps": len(set(zip(m.mode, m.map))),
        "test_start": int(test.ts.min()) if len(test) else None,
        "params": model.params, "fit_seconds": round(fit_s, 1),
        "coin_flip": metrics(baseline_coin(base_train, test), test.y),
        "map_winrate": metrics(baseline_map_wr(base_train, test), test.y),
        "model_draft_only": metrics(model.predict_proba(test, rank_zero=True), test.y),
        "model_with_rank": metrics(model.predict_proba(test, rank_zero=False), test.y),
    }
    if verbose:
        print(f"matches={res['n_matches']} train={res['n_train']} test={res['n_test']} "
              f"(holdout tier>={hmt}) maps={res['n_maps']} fit={fit_s:.1f}s")
        print(f"{'model':<20}{'log_loss':>10}{'brier':>10}{'acc':>8}")
        for k in ("coin_flip", "map_winrate", "model_draft_only", "model_with_rank"):
            r = res[k]
            print(f"{k:<20}{r['log_loss']:>10.4f}{r['brier']:>10.4f}{r['accuracy']:>8.3f}")
    return res


def parse_sets(sets: list[str]) -> dict:
    out = {}
    for s in sets or []:
        k, v = s.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", help="param override key=value")
    ap.add_argument("--params-file", default=str(ROOT / "params.json"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--holdout-tier", type=int, default=None,
                    help="restrict holdout to games with lowest tier >= this (default: model bracket)")
    args = ap.parse_args()
    params = {}
    pf = Path(args.params_file)
    if pf.exists():
        params.update(json.loads(pf.read_text()))
    params.update(parse_sets(args.set))
    res = run(params, verbose=not args.json, holdout_min_tier=args.holdout_tier)
    if args.json:
        print(json.dumps(res, indent=1))
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / "last_eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

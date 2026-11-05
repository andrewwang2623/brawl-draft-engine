"""Train comp models on ALL collected ranked games and save one per rank bracket.

    .venv/bin/python train.py                         # Legendary+ (default bracket)
    .venv/bin/python train.py --brackets legendary,masters,all

Writes models/model_<bracket>.pkl. Brackets are defined in data.BRACKETS by the lowest
tier of the six players; out-of-bracket games act only as a weak prior (see model.py).
"""
import argparse
import json
import time
from pathlib import Path

from data import BRACKETS, load_matches
from model import DraftModel, model_path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brackets", default="legendary")
    args = ap.parse_args()
    params = {}
    pf = ROOT / "params.json"
    if pf.exists():
        params = json.loads(pf.read_text())
    m = load_matches()
    for b in args.brackets.split(","):
        t0 = time.time()
        p = dict(params, bracket_min_tier=BRACKETS[b])
        model = DraftModel(p).fit(m, verbose=True)
        model.bracket = b
        model.save(model_path(b))
        n_in = int(model.in_bracket(m).sum())
        print(f"[{b}] trained on {len(m)} games ({n_in} in bracket) in {time.time() - t0:.1f}s "
              f"-> {model_path(b).relative_to(ROOT)}")


if __name__ == "__main__":
    main()

# Status

_Updated: 2026-11-04 17:20 PST (in progress; refreshed at each iteration)_

## What works
- Crawler running in the background (`./run_crawler.sh`), ~40k ranked games so far, 99.9% Legendary+.
- Comp model + eval harness; the model beats both baselines on the Legendary+ holdout.
- Draft engine: pick + ban recommendations with explanations, ~0.1–0.2 s per request.
- FastAPI server + JSON API, personal mode backend.
- 18 tests passing.

## In progress
- Web UI (`web/index.html`).
- Iteration loop (see EXPERIMENTS.md).

# Brawl Stars ranked draft engine

Given map, mode, which side picks first, bans, and picks so far (ranked 1-2-2-1 order),
recommends the best next pick and the best bans, with an explanation for each. Focus:
**Legendary+** games.

```
crawler.py  ──► data/matches.db ──► model.py (comp strength) ──► draft.py (expectimax) ──► server.py + web/
 (official API)     (SQLite)          train.py / evaluate.py                                (local UI)
```

## Setup

```bash
cd ~/Documents/Claude/brawl-draft-engine
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env     # then set BRAWL_API_KEY=... (IP-locked key from developer.brawlstars.com)
                         # optional: BRAWL_PLAYER_TAG=#YOURTAG for personal mode
.venv/bin/python build_meta.py   # brawler/map metadata (already committed in meta/)
```

## Crawler

```bash
./run_crawler.sh     # starts detached with an auto-restart supervisor; no-op if already running
./stop_crawler.sh    # clean stop (commits, exits); safe to restart any time, all state is in SQLite
tail -f data/crawl.log
```

- Seeds from global + 37 country top-200 rankings (re-seeded every 6h), then snowballs through
  players found in ranked games. Legendary+ players are queued first.
- Keeps only `soloRanked` (the ranked draft queue; `ranked` in the API is the trophy ladder).
- Dedupes on battle time + sorted 6 player tags. 4 req/s with backoff on 429/5xx.
- Re-polls active high-rank players every ~2h, since battlelogs only hold 25 games.
- If `BRAWL_PLAYER_TAG` is set, also saves *all* of your games to `my_games` every 20 min.
- On `accessDenied` / `invalidIp` (key not whitelisted for your current IP), it writes a note to
  STATUS.md and stops cleanly.

Tables: `matches` (map, mode, time, winner, `tier_min`/`tier_avg`), `match_players` (brawler,
power, `rank` = ranked tier, tag), `players` (crawl queue), `my_games`, `type_counts`.

## Train / evaluate

```bash
.venv/bin/python train.py                     # -> models/model_legendary.pkl  (~seconds)
.venv/bin/python train.py --brackets legendary,masters,all
.venv/bin/python evaluate.py                  # time holdout: model vs coin flip vs map win rates
.venv/bin/python compare_brackets.py --tier 16
.venv/bin/python experiment.py --desc "..." --set lam_map=100 --apply   # one tuning experiment
.venv/bin/python -m pytest -q                 # tests (model + draft search)
```

Hyperparameter overrides live in `params.json` (defaults in `model.DEFAULT_PARAMS`).
**New balance patch?** Edit `patch_time_utc` (and `changed_brawlers`) in `patch.json` and retrain.
Games before that time become a weak prior, and changed brawlers lean on it even less.
Experiment history: `EXPERIMENTS.md`. Design decisions: `DECISIONS.md`.

## Web UI

```bash
.venv/bin/python server.py          # http://127.0.0.1:8000   (--host 0.0.0.0 for another device)
```

Pick the map, side, and rank bracket, then click brawlers to ban/pick. Recommendations update on
every click. The server hot-reloads the model whenever `train.py` writes a new one.
**Personal mode** (toggle) restricts the recommended pick to brawlers you own, penalizes power < 11,
and adds your per-brawler "me factor" + comfort bonus as a separate "you" line.

Personal-mode inputs: `BRAWL_PLAYER_TAG` (owned brawlers, power, buffie flags, and your battle
history from the crawler), `reference/my_buffies.txt` (brawlers you're missing buffies on; also
editable in the UI's "My stats" panel), and `my_brawlers.yaml` (penalty size).

API: `GET /api/meta`, `POST /api/recommend`, `GET /api/my_stats`, `GET/PUT /api/my_buffies` (see `server.py`).

## Reference data

`reference/` holds AshBS's Nov 2026 tier list (image + `ashbs_tiers.csv`, normalized and
cross-checked against the image by `match_tierlist.py`). `sanity_check.py` compares it with the
model and writes `SANITY_CHECK.md`. It's a sanity check only, never training data.

## How it works

**Comp model** (`model.py`): regularized logistic regression, antisymmetric in the two teams:

    logit P(A wins) = Σ_A S(a) − Σ_B S(b) + Σ_pairs∈A syn − Σ_pairs∈B syn + Σ_{a∈A,b∈B} C(a,b) [+ rank control]
    S(x) = global(x) + mode(x) + map(x) + class×mode + class×map

Separate L2 penalties per layer amount to Bayesian shrinkage map → mode → global. Class terms give
unseen brawler/map combinations a sensible prior. Games are recency-weighted. For the
Legendary+ bracket, lower-tier games inform only shared terms, at reduced weight (a weak prior).
Every logit is a sum of per-brawler and per-pair terms, so each pick's contribution decomposes exactly.

**Draft engine** (`draft.py`): expectimax over the remaining picks. Our picks are max nodes
(beam of top candidates). Opponent picks are chance nodes drawn from a realistic distribution,
∝ map pick rate × exp(β·gain), rather than perfect play. Because of this, first picks favor
brawlers that can't easily be countered, and last picks are pure counters. Each recommendation reports
expected win%, delta vs a typical pick, and a breakdown: map strength / synergy /
counters vs current enemies / vs likely future enemy picks (counter risk). Bans are scored by how
much removing a brawler raises our expected draft value, from the actual pick side.

Win% numbers assume equal skill and an opponent who drafts "realistically". Expect real
prediction accuracy to top out around 55–60%, because player skill dominates.

# Status

_Updated: 2026-11-04 21:05 PST. The tuning loop has finished (stopped after 3 consecutive non-improving
rounds). The crawler and profile fetch are still running. Next up: counterpick-risk work in the draft
engine._

## For you to check
- `roles.yaml`: all tags are high confidence after your 24 corrections. One open question: **Lumi** is
  set to `controller` (you weren't sure about that line).
- `checks/matchups.yaml`: add any matchup you're sure about. It's checked after every retrain
  (`reference/MATCHUP_CHECKS.md`) and never used as training data.

## Data
- **456,287 decisive ranked games** (414,644 Legendary+, all post-Nov 3 patch), 26 maps, 174k players
  known. Crawler adds ~100–130k games/hour.
- Player profiles (Elo + per-brawler trophies, for skill control): 10,101 in `data/profiles.db`. Fetch
  continues at ≤1 req/s.
- Draft log (`data/drafts.db`): 6 drafts. "We won / We lost" results feed a future fit of the
  ban-intent strength.
- Your history (`my_games`): 25 games, polled every 20 min.

## Best metrics (post-patch Legendary+ holdout = newest 15%, paired on frozen snapshots)
| | value |
|---|---|
| draft-only holdout log loss | **0.6717** (445k-game snapshot); coin flip 0.6931; per-map win-rate baseline 0.6753 |
| Brier / accuracy | 0.2395 / 58.4% |
| calibration | predicted vs. observed within ~2 pts in every bin |
| audit (production model) | **1/26 maps flagged**: Gem Fort low-overlap with its most-picked. Its top 10 are 7/10 picked-and-winning, all high-confidence with 845+ games, so it's data-driven. Mean overlap 6.7/10 most-picked, 8.4/10 picked-and-winning |
| matchup checks | **19/19** |
| typical-pick baseline | 52.9% (was 63% before the fix) |
| AshBS tier correlation (sanity only) | Spearman 0.32 (was 0.00 before the fix) |

## Tuning loop summary
| round | change | Δ holdout ll | audit / matchups | result |
|---|---|---|---|---|
| 1 (pre-fix) | counter L2 300 → 100 | +0.0009 | n/a | reverted |
| 2 (pre-fix) | recency half-life 14d → 2d | +0.0001 | n/a | reverted |
| model fix | skill controls, pick-rate prior, map similarity, p25 ranking, roles | −0.0013 | 26 → 0 flagged | **kept** |
| 3 | latent role dims 4 → 6 | +0.0001 | same | reverted |
| 4a | your role-tag corrections | +0.0000 | same | kept (domain knowledge) |
| 4 | map-deviation L2 200 → 100 | +0.0000 | same | reverted |
| 5 | player-effect L2 40 → 20 | +0.0003 | same | reverted → **loop stopped** |

Regularization is at a plateau. Further gains need new information or better draft-time decisions, not tuning.

## What works
- **Crawler:** official API → SQLite, Legendary+-focused, resume-safe.
- **Model:** Legendary+ bracket model with patch-aware priors (`patch.json`). Skill controls (player
  effect, Elo, mastery), pick-rate prior, map-similarity priors, range × openness, latent-role +
  role-tag matchups, conservative (p25) ranking.
- **Draft engine:**
  - picks for any slot, with missing earlier picks treated as unknown;
  - ban recs and enemy-pick prediction;
  - ban-intent reading (off/low/medium);
  - comp notes ("Their 3 divers hard-counter your 3 throwers (−27% for you)").
- **Web UI** (http://127.0.0.1:8000):
  - slot input in any order and per-team bans;
  - "me" slot + Suggest for me;
  - personal mode: owned brawlers, buffies from `reference/my_buffies.txt`, me factor;
  - Appearance settings: detail level, themes, colorblind colors;
  - We won / We lost draft log.
- **Reference:** AshBS tier list (normalized + sanity check), audit, matchup checks.
- **Tests:** 37 pass.

## How to run / stop
- Crawler: `./run_crawler.sh` (detached, auto-restart) / `./stop_crawler.sh` (clean stop, resume any time).
- Profiles: `.venv/bin/python profiles.py --max N` (resumable; stop with `pkill -f profiles.py`).
- Retrain: `.venv/bin/python train.py` (~3 min). The server hot-reloads the model.
- Server: `.venv/bin/python server.py` (port 8000).
- One experiment: `.venv/bin/python experiment.py --desc "..." --set key=value --snapshot $(date +%s) --apply`.

## Known limitations / next ideas
- **In progress:** counterpick risk (position-aware enemy greed, worst-case exposure, risk slider, pair
  counter magnitudes, draft replay).
- **Fit the ban-intent strength (gamma)** from logged drafts once ~100 drafts with bans exist.
- **Retrain periodically** as profile coverage grows.
- **Some learned matchup magnitudes are large** (e.g. Edgar vs Sprout +12%). Directionally right.
- **No map geometry from the API**, so there's no wall-break feature.
- **The bracket mix is drifting** (~91% Legendary+ in recent crawls). Out-of-bracket games only act as a weak prior.

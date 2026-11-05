# Status

_Updated: 2026-11-04 19:10 PST, tuning loop round 3 (first round after the model fix)_

## ⚠️ For you to check: low-confidence role tags in `roles.yaml` (24)
These tags drive role-level counters and synergy, comp notes, and the role matchup checks.
Correct any that are wrong in `roles.yaml` (edit `roles:`, then set `confidence: high`) and retrain.

Gigi: assassin · Nori: assassin · Starr Nova: assassin · Chuck: assassin, controller ·
Janet: sniper, assassin · Bolt: tank · Damian: tank · Ollie: tank · Trunk: tank ·
Doug: tank, support · Cosmo: controller · Finx: controller · Meeple: controller ·
Sirius: controller · Ziggy: controller · Glowy: support · Wendy: support · Eve: mid_dps ·
Lumi: mid_dps · Mina: mid_dps · Moe: mid_dps · Najia: mid_dps · R-T: mid_dps · Vince: mid_dps

## ⚠️ Web server needs a restart
The server on :8000 is still running pre-fix code and returns HTTP 500 since the new model
format landed. Restart it: `pkill -f "server.py --port 8000"; nohup .venv/bin/python server.py --port 8000 > data/server.stdout.log 2>&1 &`

## Data
- **280,036 decisive ranked games** (269,623 Legendary+, all post-patch), 26 maps, 109,712 players
  known. The crawler runs continuously (~130k new games/hour), resume-safe (`./run_crawler.sh`,
  `./stop_crawler.sh`).
- Player profiles (Elo + per-brawler trophies, skill control): 3,701 fetched into
  `data/profiles.db`. Fetch continues at ≤1 req/s (`.venv/bin/python profiles.py --max N`, resumable).
- Your history (`my_games`): 25 games so far, polled every 20 min.

## Best metrics (post-patch Legendary+ holdout, newest 15%)
| | value |
|---|---|
| draft-only log loss | **0.6709** (coin flip 0.6931; per-map win-rate baseline ≈0.679) |
| Brier / accuracy | 0.2391 / 58.6% |
| calibration | predicted vs. observed within ~2 pts in every bin |
| audit | **0/26 maps flagged**; top-10 overlap with 15 most-picked 6.5/10, with picked-and-winning 8.2/10 |
| matchup checks | **17/17** (`reference/MATCHUP_CHECKS.md`) |
| typical-pick baseline | 53.0% (was 63% before the fix) |
| AshBS tier correlation (sanity only) | Spearman 0.32 (was 0.00) |

## What works
- Crawler → SQLite; Legendary+ bracket model with patch-aware priors (`patch.json`).
- Model: skill controls (player effect, Elo, mastery), pick-rate prior, map-similarity priors,
  range × openness, latent-role + role-tag matchups, conservative (p25) ranking.
- Draft engine: picks for any slot (missing earlier picks treated as unknown), ban recs, enemy-pick
  prediction, ban-intent reading (off/low/medium), comp notes ("Their 3 divers hard-counter…").
- Web UI: slot-based input in any order, per-team bans, "me" slot + Suggest for me, personal mode
  (owned brawlers, buffies from `reference/my_buffies.txt`, me factor), Appearance settings
  (detail level, themes, colorblind colors), draft log with We won / We lost (`data/drafts.db`).
- Reference: AshBS tier list normalized + sanity check (`reference/SANITY_CHECK.md`), audit
  (`reference/AUDIT.md`).
- 37 tests.

## Iteration loop
See EXPERIMENTS.md. Every experiment is a paired comparison on a frozen snapshot. It's kept only if
holdout log loss improves AND audit flags / matchup pass rate don't regress.

## Known limitations / next
- Fit the ban-intent strength (gamma) from logged drafts once ~100 drafts with bans exist.
- Profile coverage is partial (~15–20% of player slots). Retrain as it grows.
- Some learned matchup magnitudes are large (e.g. Edgar vs Sprout +12%). Directionally right; watch them.
- The API has no map geometry, so there's no wall-break feature.

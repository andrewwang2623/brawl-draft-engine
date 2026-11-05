# Status

_Updated: 2026-11-04 20:10 PST, tuning loop round 4 done (2 consecutive non-improving rounds since the model rewrite; the loop stops at 3)_

## For you to check
- `roles.yaml`: your 24 corrections are applied. One open question: **Lumi** is set to `controller`
  (you weren't sure about that line). Everything is now high-confidence.

## Data
- **~359,000 decisive ranked games** (~94% Legendary+, all post-patch), 26 maps, 110k+ players
  known. The crawler runs continuously (~130k new games/hour), resume-safe (`./run_crawler.sh`,
  `./stop_crawler.sh`).
- Player profiles (Elo + per-brawler trophies, skill control): 3,701 fetched into
  `data/profiles.db`. Fetch continues at ≤1 req/s (`.venv/bin/python profiles.py --max N`, resumable).
- Your history (`my_games`): 25 games so far, polled every 20 min.

## Best metrics (post-patch Legendary+ holdout, newest 15%)
| | value |
|---|---|
| draft-only log loss | **0.6722** on the current 359k-game snapshot (0.6709 on the 214k snapshot at fix time; coin flip 0.6931; per-map win-rate baseline 0.6762) |
| Brier / accuracy | 0.2391 / 58.6% |
| calibration | predicted vs. observed within ~2 pts in every bin |
| audit | **0/26 maps flagged**; top-10 overlap with 15 most-picked 6.9/10, with picked-and-winning 8.2/10 |
| matchup checks | **19/19** (`reference/MATCHUP_CHECKS.md`; includes your Bull vs Bibi/Kenji) |
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

## Tuning rounds since the model rewrite
| round | change | Δ holdout ll | audit / matchups | result |
|---|---|---|---|---|
| 3 | latent role dims 4 → 6 | +0.0001 | 0/26, 17/17 → same | reverted |
| 4a | your role-tag corrections | +0.0000 | 0/26, 17/17 → same | kept (domain knowledge) |
| 4 | map-deviation L2 200 → 100 | +0.0000 | 0/26, 19/19 → same | reverted |

## Iteration loop
See EXPERIMENTS.md. Every experiment is a paired comparison on a frozen snapshot. It's kept only if
holdout log loss improves AND audit flags / matchup pass rate don't regress.

## Known limitations / next
- Fit the ban-intent strength (gamma) from logged drafts once ~100 drafts with bans exist.
- Profile coverage is partial (~15–20% of player slots). Retrain as it grows.
- Some learned matchup magnitudes are large (e.g. Edgar vs Sprout +12%). Directionally right; watch them.
- The API has no map geometry, so there's no wall-break feature.

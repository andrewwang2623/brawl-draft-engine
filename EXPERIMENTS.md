# Experiments

Holdout = newest 15% of games by time; the model trains on everything older. Primary metric:
**holdout log loss** of the draft-only model (rank feature zeroed, i.e. what the draft engine
sees). Since the bracket change, the holdout is restricted to **Legendary+ games** (lowest
of the 6 tiers ≥ 16). Since the patch change (11-04 17:40), it is further restricted to
**post-patch games** (≥ `patch.json` time, 2026-11-03 09:00 UTC). Today, that's every game. "full" = model including the rank-difference skill control.
Baselines on the same holdout are listed in each row (coin flip = 0.6931).

| time (PST) | #matches | change tried | draft-only ll / brier / acc | full ll | map-WR baseline ll | kept? |
|---|---|---|---|---|---|---|
| 11-04 17:05 | 11,993 | setup: default L2 ×10 stronger (first fit overfit: ll 0.7033 > coin) | 0.6852 / 0.2460 / 0.566 | 0.6753 | 0.6885 | kept (setup) |
| 11-04 17:13 | 31,890 | bracket cmp on **Legendary+ holdout** (n=4781): all-ranks model (all games equal weight) | 0.6785 / 0.2428 / 0.578 | 0.6668 | 0.6835 | best |
| 11-04 17:13 | 31,890 | bracket cmp on **Legendary+ holdout** (n=4781): Legendary+ model, lower tiers weak prior (w=0.25), dev penalty ×1 | 0.6801 / 0.2435 / 0.571 | 0.6684 | 0.6835 | - |
| 11-04 17:13 | 31,890 | bracket cmp on **Legendary+ holdout** (n=4781): Legendary+ model, lower tiers weak prior (w=0.25), dev penalty ×3 | 0.6790 / 0.2430 / 0.575 | 0.6673 | 0.6835 | - |
| 11-04 17:13 | 31,890 | bracket cmp on **Legendary+ holdout** (n=4781): Legendary+ model, lower tiers weak prior (w=0.25), dev penalty ×10 | 0.6786 / 0.2429 / 0.576 | 0.6669 | 0.6835 | - |
| 11-04 17:13 | 31,890 | bracket cmp on **Legendary+ holdout** (n=4781): Legendary+ games only | 0.6790 / 0.2430 / 0.574 | 0.6673 | 0.6835 | - |
| 11-04 17:13 | 31,890 | bracket cmp on **Masters+ holdout** (n=1739): all-ranks model (all games equal weight) | 0.6843 / 0.2456 / 0.561 | 0.6722 | 0.6909 | best |
| 11-04 17:13 | 31,890 | bracket cmp on **Masters+ holdout** (n=1739): Masters+ model, lower tiers weak prior (w=0.25), dev penalty ×1 | 0.6894 / 0.2479 / 0.541 | 0.6772 | 0.6909 | - |
| 11-04 17:13 | 31,890 | bracket cmp on **Masters+ holdout** (n=1739): Masters+ model, lower tiers weak prior (w=0.25), dev penalty ×3 | 0.6874 / 0.2470 / 0.549 | 0.6752 | 0.6909 | - |
| 11-04 17:13 | 31,890 | bracket cmp on **Masters+ holdout** (n=1739): Masters+ model, lower tiers weak prior (w=0.25), dev penalty ×10 | 0.6864 / 0.2466 / 0.547 | 0.6742 | 0.6909 | - |
| 11-04 17:13 | 31,890 | bracket cmp on **Masters+ holdout** (n=1739): Masters+ games only | 0.6883 / 0.2475 / 0.545 | 0.6760 | 0.6909 | - |
| 11-04 17:17 | 39,546 | iter 1: counter-pair L2 lam_cnt 300 -> 100 (Δll +0.0009 ± 0.0004; prev 0.6783) | 0.6792 / 0.2431 / 0.571 | 0.6685 | 0.6815 | reverted |
| 11-04 17:35 | 52,977 | patch-aware v1: post-patch target, pre-patch weight 0.3, changed brawlers' deviation penalty ×0.3. **No pre-patch data exists**, so this only loosened regularization on 34 changed brawlers | 0.6769 / 0.2420 / 0.568 | — | 0.6800 | replaced (worse than no-patch 0.6745 by 0.0024 ± 0.0007) |
| 11-04 17:40 | 52,977 | patch-aware v2: same total prior variance for changed brawlers, 90% of it on the post-patch deviation (`changed_prior_share` 0.1). Identical to the no-patch model while there's no pre-patch data (Δ 0.0000) | 0.6757 / 0.2414 / 0.573 | — | 0.6806 | **kept** (required; activates when pre-patch data exists) |
| 11-04 17:44 | 98,613 | iter 2: recency half-life 14d -> 2d (meta shifting right after patch) (Δll +0.0001 ± 0.0001; prev 0.6739) | 0.6740 / 0.2406 / 0.579 | 0.6632 | 0.6786 | reverted |
| 11-04 19:10 | 176,726 | **v2 ablations** (one frozen snapshot, same split; audit = flagged maps / mean top-10 overlap with 15 most picked; all use the new engine). Pre-fix model (legacy params) + new engine | 0.6729 / 0.2401 / 0.581 | 0.6635 | — | baseline (20/26 flagged, overlap 3.3, matchups 12/17) |
| 11-04 19:10 | 176,726 | v2 first cut (k=4, lam_role 5, lam_player 10, sim_rank 10, rare 3) | 0.6727 / 0.2399 / 0.584 | 0.6685 | — | - (0/26, 7.5) |
| 11-04 19:10 | 176,726 | matchups: **in-game classes** (class-vs-class + class+class, no roles, no latent) | 0.6750 / 0.2411 / 0.578 | 0.6708 | — | rejected (worse than embeddings) |
| 11-04 19:10 | 176,726 | matchups: **embeddings only** (latent k=4, no roles) | 0.6721 / 0.2396 / 0.586 | 0.6683 | — | - (0/26, 6.7) |
| 11-04 19:10 | 176,726 | matchups: **roles only** (roles.yaml, k=0) | 0.6739 / 0.2405 / 0.578 | 0.6697 | — | - (0/26, 7.4) |
| 11-04 19:10 | 176,726 | matchups: **embeddings + roles** (first cut above, lam_role 5) | 0.6727 / 0.2399 / 0.584 | 0.6685 | — | - |
| 11-04 19:10 | 176,726 | matchups: neither (per-pair terms only) | 0.6738 / 0.2405 / 0.580 | 0.6697 | — | - (1/26 flagged) |
| 11-04 19:10 | 176,726 | latent k: 2 / 4 / 8 | 0.6726 / 0.6727 / 0.6731 | — | — | k=4 (k=2 ties; 8 overfits) |
| 11-04 19:10 | 176,726 | lam_role 5 → 30 → 100 (role terms lightly vs moderately shrunk) | 0.6727 → 0.6724 → 0.6722 | — | — | 100 |
| 11-04 19:10 | 176,726 | skill controls off (no player effect / profiles / main) | 0.6715 / 0.2394 / 0.582 | 0.6622 | — | not taken: better ll but overlap 7.5 → 6.0 (selection bias is what (a) removes) |
| 11-04 19:10 | 176,726 | lam_player 10 → 40; 40 without "main" | 0.6721; 0.6713 | 0.6626; 0.6593 | — | lam_player 40, main kept (overlap 7.4 vs 6.3) |
| 11-04 19:10 | 176,726 | no pick-rate prior / no map-similarity / no rare-cell shrinkage | 0.6726 / 0.6728 / 0.6730 | — | — | all kept (each helps a little) |
| 11-04 19:10 | 176,726 | lam_lat 50 → 20 / 100 / 300 | 0.6743 / 0.6717 / 0.6730 | — | — | 100 |
| 11-04 19:35 | 176,726 | v2 tuned (lam_player 40, lam_role 100/300, sim_rank 5, lam_lat 100; recentered pair terms, role/latent on target games only) | 0.6717 / 0.2395 / 0.584 | 0.6615 | — | - (0/26, 6.35, matchups 16/17) |
| 11-04 19:35 | 176,726 | + lam_player 100 / + rare_shrink 6 | 0.6716 / 0.6717 | 0.6609 / 0.6613 | — | rare_shrink 6 (overlap 6.46, picked-and-winning 7.92) |
| 11-04 18:37 | 214,091 | **v2 recommendation fix** vs pre-fix model (legacy params, same engine): skill controls, pick-rate prior, map-similarity + range×openness priors, rare-cell shrinkage, role + latent matchups (k=4), p25 ranking (Δll -0.0012 ± 0.0005; prev 0.6722; audit flagged maps 19→0, matchups 11/17→16/17) | 0.6709 / 0.2391 / 0.586 | 0.6623 | 0.6767 | **kept** |
| 11-04 18:41 | 214,091 | calibration of the v2 model (draft-only, both orientations), predicted/observed per bin: <.30 0.26/0.28; .30–.40 0.36/0.38; .40–.45 0.43/0.43; .45–.50 0.48/0.47; .50–.55 0.53/0.53; .55–.60 0.57/0.57; .60–.70 0.64/0.63; >.70 0.74/0.72 | 0.6709 / 0.2392 / 0.586 | 0.6621 | 0.6767 | (check only) |
| 11-04 18:41 | 214,091 | audit before → after (trained on all data): flagged maps 26 → 0, flags 75 → 0, top-10 overlap with 15 most picked 2.5 → 6.6, with picked-and-winning 3.4 → 8.0, typical-pick baseline 63.1% → 53.0%, matchup checks 17/17, AshBS Spearman 0.00 → 0.32 | — | — | — | (check only) |
| 11-04 19:05 | 274,819 | **Loop resumed after the model rewrite: consecutive-failure counter reset to 0.** Retrained on all data: audit 0/26 flagged, overlap 6.54 / 8.15, matchups 17/17 | — | — | — | — |
| 11-04 19:10 | 279,197 | round 3 (post-fix #1): latent role dims lat_k 4 -> 6 (Δll +0.0001 ± 0.0001; prev 0.6718; audit flagged maps 0→0, matchups 17/17→17/17) | 0.6719 / 0.2396 / 0.584 | 0.6637 | 0.6769 | reverted |
| 11-04 19:55 | 346,798 | round 4a: user role-tag corrections (24 brawlers) - old roles.yaml vs corrected (Δll +0.0000 ± 0.0001; prev 0.6723; audit flagged maps 0→0, matchups 17/17→17/17) | 0.6723 / 0.2398 / 0.584 | 0.6666 | 0.6763 | **kept**: user domain correction; metric-neutral (not a tuning attempt, doesn't count toward the failure streak) |
| 11-04 20:04 | 359,132 | round 4 (post-fix #2): map-deviation L2 lam_map 200 -> 100 (now ~350k games) (Δll +0.0000 ± 0.0001; prev 0.6722; audit flagged maps 0→0, matchups 19/19→19/19) | 0.6722 / 0.2397 / 0.584 | 0.6661 | 0.6762 | reverted |

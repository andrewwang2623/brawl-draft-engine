# Experiments

Holdout = newest 15% of games by time; the model trains on everything older. Primary metric:
**holdout log loss** of the draft-only model (rank feature zeroed, i.e. what the draft engine
sees). Since the bracket change, the holdout is restricted to **Legendary+ games** (lowest
of the 6 tiers ≥ 16). "full" = model including the rank-difference skill control.
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

# Decisions log

Judgment calls made while building autonomously. Newest at the bottom of each section.

## Data / API

- **Ranked battle type = `soloRanked`** (verified, not guessed). In battlelogs from the top 15 global
  players (samples in `samples/`):
  - `soloRanked`: 3v3, no `trophyChange`. Here `brawler.trophies` holds a small integer (seen 12–20),
    which is the player's **ranked tier** (1 = Bronze I … ~19+ = Masters/Pro). All powers were 11.
  - `ranked`: the **trophy ladder**. It has `trophyChange` and includes 5v5 and showdown events.
    Despite the name, this is NOT the draft mode, so it's excluded.
  - `teamRanked` is also accepted in case a team queue reports under that name (none seen so far).
    `battle_type` is stored per match, so it can be filtered later.
- **No battle id in the API.** Dedupe key = `battleTime + "|" + sorted(6 player tags)`, as specified.
- **Canonical orientation**: team A is the team containing the lexicographically smallest player tag.
  `winner` is 0 (A), 1 (B) or NULL (draw). Draws are stored but excluded from training.
- **Ranked is best-of-3 sets**, and each game appears as a separate battlelog entry, so games in a set
  share comps and players and are correlated. They're kept as separate rows (they are real outcomes).
  The time-based holdout keeps whole sets mostly on one side of the split.
- Some top players' battlelogs return **404** (likely private or empty). Those players are marked dead
  after 3 consecutive 404s.
- **Power level carries ~no signal in ranked**: every observed ranked brawler is power 11. It's stored
  but not used as a model feature.
- **Crawl prioritization**: never-polled players by highest known ranked tier first. Re-polls are
  scheduled by activity (2h for very active high-rank players, 6h/12h/48h otherwise). The mix is
  ~65% fresh / ~35% re-poll. Seeds come from the global ranking + 37 country rankings (top 200 each),
  refreshed every 6h. Trophy-leaderboard players aren't necessarily ranked players, but snowballing
  through `soloRanked` opponents quickly moves the crawl into the ranked population.
- **Rate**: 4 req/s with exponential backoff on 429/5xx.
- **Personal history** (user add-on): the crawler also polls `BRAWL_PLAYER_TAG`'s battlelog every
  20 min and stores *all* of the owner's games in `my_games` (raw JSON kept, so nothing is lost).
  The owner's ranked games also go into `matches`.

## Metadata

- `api.brawlify.com` returns a 403 "Request Blocked" page to scripts. The legacy mirror
  `api.brawlapi.com/v1/*` serves the same JSON, so that's what `build_meta.py` uses.
- Brawlify map entries have mode and environment but **no geometry** (wall density, lane width, ...).
  Cold-start "map features" are therefore the **mode**, plus **class × map** terms that are themselves
  shrunk toward **class × mode**. A brawler unseen on a map still gets a sensible prior from its
  class (Brawlify classes: Damage Dealer, Tank, Assassin, Controller, Support, Marksman, Artillery).

## Rank tiers / brackets (user request: focus on Legendary+)

- **Tier mapping verified, not guessed.** In `soloRanked` battles, `brawler.trophies` is the player's
  ranked tier. Checked against `/players/{tag}` → `rankedRank` / `rankedRankName` for 48 players
  sampled across every value seen (14…22). The battle value equals the profile `rankedRank` for 42/48;
  the other 6 differ by exactly ±1 (they played games after the stored battle). Names observed:
  14 MYTHIC II, 15 MYTHIC III, 16 LEGENDARY I, 17 LEGENDARY II, 18 LEGENDARY III, 19 MASTERS I,
  20 MASTERS II, 21 MASTERS III, 22 PRO. So the ladder is 1–3 Bronze … 13–15 Mythic, 16–18 Legendary,
  19–21 Masters, 22 Pro. Within a player's games the value only ever moves ±1 (rarely ±2).
  Sample profile: `samples/player_profile_tier22.json`. Profiles also expose `rankedElo`, which could
  serve as a finer skill control later.
- **Match tier** is stored as `tier_min` (lowest of the 6) and `tier_avg`. A match is "Legendary+"
  if `tier_min >= 16`, i.e. every player is Legendary or higher. Lowest-of-6 is the strict definition.
- At the time of the switch, **16,563 of 16,583 games (99.9%) were already Legendary+**, because
  seeding from top trophy players lands the crawl in high tiers. So restricting to Legendary+ costs
  almost no data.
- **Crawler focus**: unpolled players who are Legendary+ themselves, or who appeared in a Legendary+
  game (`players.seen_tier`), are queued first (higher tiers first), then unpolled seeds. Players
  known to be below Legendary are polled only if nothing else is queued. Everything is still stored.
- **Bracket model** (`bracket_min_tier`): shared terms are fit on all games, and bracket games
  additionally fit deviation terms (same feature groups, penalty × `lam_dev_scale`). Out-of-bracket
  games are down-weighted (`low_tier_weight` = 0.25), so they act only as a weak prior that bracket
  estimates shrink toward. With 99.9% Legendary+ data this is ≈ the all-ranks model. A Masters+ test of
  the same machinery found bracket-specific estimates *worse* than pooling (see EXPERIMENTS.md), so
  the deviation penalty defaults to ×10 (mostly shared).
- Pick-rate priors (the opponent model) use only in-bracket games.

## Balance patch (user request)

- **Patch time = 2026-11-03 09:00 UTC** (`patch.json`; change `patch_time_utc` for future patches,
  it's one line). Evidence:
  - The official [Release Notes November 2026](https://supercell.com/en/games/brawlstars/blog/game-updates/release-notes-november-2026/)
    are dated 3 Nov 2026 (no go-live hour given). The user confirmed the update has been live since Nov 3.
  - Battlelogs were evidently **reset at the update**. Across ~40k games and 25k players, nothing is
    older than 2026-11-03 09:36 UTC, and many low-activity players' logs hold fewer than 25 games,
    none earlier than 09:55. So maintenance ended at ~09:30 UTC. 09:00 is a safe boundary just before it.
  - A Nov 2 date (the user's first estimate) couldn't be confirmed; there are no Nov 2 notes.
  - Caveat: our map pool is the same 26 maps throughout, and none of the notes' 19 new ranked maps or 3
    new brawlers appear yet (probably arriving with the new season; a leak says Nov 6).
- **Every game we have is post-patch (52,326 Legendary+ games at the time), and pre-patch data
  can't be obtained from the API at all** (logs were reset; they hold only 25 games). So the
  pre-patch prior machinery exists, but has nothing to shrink toward this patch. It becomes active for
  the *next* patch: games already in the DB will then be "pre-patch".
- **Changed brawlers**: all 34 brawlers with global balance changes, reworks or new
  Buffies/Hypercharges in the notes (`patch.json`). Brawl Arena-only changes are excluded.
  For terms involving them, the total prior variance stays the same, but only 10% of it is on the
  shared (pre-patch-informed) term (`changed_prior_share`). Their stale pre-patch stats therefore
  carry little weight. A first version just made their deviation penalty cheaper; with no pre-patch
  data that loosened regularization and overfit (EXPERIMENTS.md), so it was replaced.
- Pre-patch games get weight 0.3 (`pre_patch_weight`), on top of recency weighting.
- Holdout = post-patch (and Legendary+) games only.

## Personal mode: buffies (user request)

- **The API exposes buffies**: `/players/{tag}` → `brawlers[].buffies = {gadget, starPower, hyperCharge}`
  (booleans). Battlelogs do **not** include buffies or gears for anyone, so the penalty can't be
  estimated from match data. It's a configured, moderate value (`buffie_penalty_pct: 3.0` in
  `my_brawlers.yaml`, i.e. −3 win-% for missing all buffies), scaled by the fraction missing.
- **"All false" usually means the brawler has no buffies in the game**, not that they're missing. Even a
  Pro player's profile shows exactly 33 fully-buffied brawlers and 75 with none. Which brawlers (and
  which buffie types) exist is inferred from the union over 10 Masters III+/Pro profiles + the owner
  (`meta/buffies_available.json`). A brawler counts as missing buffies only if it has them in the game.
- Auto-detection from the owner's profile reproduces all 6 brawlers the user named (Gale, Mr. P, Lou,
  Berry, Mandy, Chester; all reworked or given new Buffies in the Nov update). It also flags 6 more
  partial gaps: El Primo, Poco, Amber, Gus, Chuck, Shade. Per the instruction to read buffies
  automatically when the API has them, these are penalized too. The user's 6 are also forced in
  `my_brawlers.yaml`, and anything can be exempted via `assume_full`.
- Applied only to the user's own recommended pick (personal mode on). Teammates' and enemies' picks,
  the opponent model, and personal-mode-off recommendations all assume fully built brawlers.
- Ranked requires power 11 (confirmed by the Nov 2026 release notes; every observed ranked brawler is
  power 11). The owner's two power-9 brawlers (Ziggy, Vince) get −3%/level in personal mode. In
  practice they're probably not even eligible.
- **Update (user):** `reference/my_buffies.txt` is now the authoritative list of brawlers the user is
  missing buffies on. Every other brawler that has buffies counts as fully built. Which buffie types
  are missing (and so the penalty fraction) comes from the API flags when available, else all three. The
  server reloads the file when it changes, and the UI's "My stats" panel can edit it (`PUT /api/my_buffies`).
  The 12 names in the file match exactly the 12 the API auto-detected.

## Reference tier list (AshBS, Nov 2026)

- Used **only** for a sanity check (`reference/SANITY_CHECK.md`), never for training.
- Name normalization: "new brawler frog" → Thunder Frogs (unreleased per the user, excluded),
  "dick" → Tick (icon match 0.79 at that slot), "allie" → Alli, "primo" → El Primo, "ruffs" → Ruffs
  (official API name), "larry lawrie" → Larry & Lawrie, "8bit" → 8-Bit, "mr p" → Mr. P.
- Cross-checked by template-matching every icon against official Brawlify "emoji" icons
  (`reference/match_tierlist.py`) plus visual checks. All transcribed tiers agree with the image. The image
  also contains **Janet** and **R-T** (both B tier, green arrows), which were missing from the transcription
  and are added. Rey and Ruby are on neither.
- Arrow reading rule: each arrow marker sits over the top-right of its icon. This was calibrated on the
  six brawlers the patch notes give new Buffies (all rainbow). Arrows correlate with the patch but also
  mark unchanged brawlers, so they look like tier movement rather than a buff/nerf legend. `patch.json`
  keeps the official notes as the source of truth.

## Recommendation fix (Nov 4 2026): inflated "+10%" picks, winner's curse, roles

### Diagnosis: where the +10% came from (measured on the pre-fix model)

Bridge Too Far top 4 were Clancy (7 games on the map), Bolt (8), Bull (24), Ollie (0), each at about
+10–12% "map strength". `audit.py` on the pre-fix model flagged **26/26 maps**, so this happened everywhere.
Causes, biggest first (Bridge Too Far numbers):

1. **Global terms for rare brawlers were barely shrunk, and absorbed specialist bias.** Clancy's +12.7% is
   +9.0% `global`, from 158 Legendary+ games at a 58.9% raw win rate. `lam_g = 20` is a prior sd of 0.22 logit
   (±5.6 win-% points) per brawler, wider than the real spread of brawler strengths. With no skill control
   beyond the tier difference, the "mains" who play niche brawlers made those brawlers look strong.
2. **Class priors spread that bias to brawlers with no data.** `class×mode + class×map` for Tank on Bridge Too
   Far was +4.6%. It was fit on the few tanks actually played in Heist (specialists who win), then handed to
   every tank: Ollie (0 games) got +6.3% global + 4.6% class = +10.6%, and Hank (0) and Trunk (0) got the
   same class boost. Damage Dealer got +2.7%.
3. **Wrong zero point.** "Map strength" was raw S, centered on an average over all 108 brawlers, including
   never-picked ones. The pick-weighted typical pick on Bridge Too Far was already +1.9%, so every number read
   ~2 points high against what people actually pick. Nothing used pick rate as evidence, so a brawler nobody
   picks started neutral.
4. **Ranking by the posterior mean.** Taking the top of 108 noisy estimates selects the noisiest ones
   (optimizer's curse). Nothing in the engine knew a 7-game estimate is less certain than a 3,000-game one.
5. **Why the "typical pick" baseline was ~63% (mean over maps 63.1%).** If both sides draft realistically
   (Monte Carlo, both drawn from the realistic pick distribution), the expected win% is **50.0%** on every
   map. The engine's baseline put a realistic pick in our slot, then made our *later* picks argmax max
   nodes, while opponents stayed realistic chance nodes. Restricting our later picks to brawlers with ≥ 30
   games on the map drops it to 60.2%. So ~3 points came from inflated low-data picks, and ~10 points were
   structural: perfect teammates vs a merely realistic opponent, plus the winner's curse at every max node,
   plus the model believing popular picks are weak (no pick-rate prior). In solo ranked the user controls
   one slot, so "perfect teammates" was simply the wrong model.

### What changed (model-wide, no map-specific code)

- **(a) Skill controls, zeroed at draft time:** rank-tier difference (as before), a per-player skill effect
  (player tag, L2 `lam_player`), `rankedElo` and the player's `highestTrophies` on the brawler they played
  (profiles from `/players/{tag}`, cached by `profiles.py` in `data/profiles.db`, with a coverage
  indicator), and a "main" feature = leave-one-out log share of the player's games on that brawler. All are
  zeroed in the draft-only metric and in the engine. Low-pick-rate cells get extra L2:
  penalty × (1 + `rare_shrink` · n0 / (n + n0)), with n0 = 2% pick rate × games in scope. This applies to
  global, mode, map and similar-map terms, and to pairs (`pair_rare_shrink`, n0 = 0.5% of games).
- **(b) Pick-rate prior:** covariate lpr = log(smoothed Legendary+ map pick rate + 1%), centered per map,
  with a learned slope (global + per mode). It came out clearly positive: at Legendary+, being picked is
  evidence. A brawler almost nobody picks on a map starts several points down instead of neutral. The 1%
  floor caps how negative it can get.
- **(c) Map similarity replaces class priors.** Each map is represented by its Legendary+ log pick-rate
  vector. Kernel K = exp(3·(cos − 1)) between maps, factorized as E·Eᵀ (top 5 components), and per-brawler
  coefficients on E: a GP prior, so sparse cells shrink toward how the brawler does on *similar* maps
  (Bridge Too Far's nearest maps are Kaboom Canyon 0.89 and Safe Zone 0.87). **Range × openness:** Brawlify's
  API has no range stats (checked `meta/brawlers_raw.json` and the brawlapi mirror), so both are learned
  from pick patterns. They're the leading component of the double-centered brawler × map log pick-rate
  matrix. The learned "range" ranks Piper, Brock, Angelo, Belle, Byron, Nani, Mandy, Pierce on top and
  Buzz, Shade, El Primo, Ash, Stu, Emz at the bottom; "openness" ranks Knockout/Bounty maps open and Brawl
  Ball/Gem Grab closed. Class × mode / class × map terms are off (`use_class: false`, kept only for the
  legacy comparison).
- **(d) Conservative ranking.** A block-diagonal Laplace approximation gives a posterior sd for S per
  brawler and map: per brawler, all its strength columns (global, mode, map, similar-map and their
  bracket deviations) are handled jointly, using curvature aggregated per (brawler, map). Pairs get their
  own sd. Each pick rec gets `win_prob_p25`: win% with the candidate's own contribution at its 25th
  percentile (S sd plus its current pair terms). Recs are **ranked by `win_prob_p25`**.
  `confidence` = high / medium / low when that sd is ≤ 1.5 / ≤ 3 / > 3 win-% points (also
  `uncertainty_pct`). Our later picks are chosen on the conservative gain too.
- **(e) Pairs.** Per-pair counter/synergy terms keep their L2 plus the rare-pair multiplier, so sparse pairs
  shrink to ~0. They now shrink toward role-level and latent matchups (next section), not toward 0.
- **S is re-centered on the typical pick** (pick-weighted mean on the map). Average pair effects are folded
  into S as well: C' = C − g(x) + g(y), Syn' = Syn − h(x) − h(y), S' = S + 3g + 2h, where g = expected edge
  vs a typical opponent and h = expected synergy with a typical teammate. Both are exact for 3v3, so no
  prediction changes. "Map strength" now means "vs typical opponents with typical teammates on this map",
  and counters/synergy mean "what this specific enemy/teammate changes".

### Winner's curse, teammates and calibration (addendum 1 and 3)

- **Teammates are realistic chance nodes** (`SearchConfig.our_policy = "realistic"`, default):
  P ∝ prior^alpha_us · exp(beta_us · conservative gain), with their own pick-rate weight `alpha_us`. Only
  the pick being recommended is a max node. This removes the asymmetry at the source: the typical-pick
  baseline is ~50% + the current comp's edge, and a recommended pick shows a few points above it.
  `our_policy = "max"` is still available for a coordinated team. It picks our later picks by the
  conservative value plus a pick-rate prior (`prior_us` logit per nat) and propagates that lower-bound
  value at interior max nodes.
- `delta_vs_typical` keeps its meaning (vs a realistic pick in this slot, same search). It's no longer
  inflated, because neither side is assumed to draft perfectly after this pick.

### Matchups: roles + latent embeddings instead of class-vs-class (addenda 2 and 3)

- In-game classes are unreliable for matchups and are now **UI-only**. Class-vs-class counters/synergy
  were built and tested, and were worse on the holdout than latent embeddings (EXPERIMENTS.md), so they're
  off (`use_class_pairs: false`). `meta/class_overrides.json` (addendum 2, item 4) was superseded by
  `roles.yaml` and not created; the loader ignores a missing file.
- **`roles.yaml`** (repo root): 8 role tags, several allowed per brawler, each with a confidence
  (low = please check). Pre-filled from Brawlify classes, the learned range/pick patterns, and common game
  knowledge. Main judgment calls: Shelly = tank/frontline + wall-breaker (short-range anti-diver), Willow
  and Squeak = thrower (lobbed attacks), Berry = thrower + support, Bonnie and Janet = sniper + diver, Chuck
  = diver + controller (low confidence). Tags are split evenly (a brawler with 2 tags is 0.5 of each).
- **Model:** C(x, y) = role-vs-role (Tx M Tyᵀ, M antisymmetric, global + per mode)
  + latent (u_x·v_y − u_y·v_x, k = 4) + per-pair deviation. Syn(x, y) = role+role + latent (w_x·w_y) +
  per-pair deviation. Fit jointly by L-BFGS (the bilinear part is non-convex; small random init, since 0 is
  a saddle point). Role and latent terms learn from target games only (post-patch, in bracket), because
  they're not patch-aware per brawler. Everything folds into the per-map Syn/C matrices, so expectimax
  stays additive and fast.
- **Comp-level features:** class/role counts are per-brawler (already in S). "#divers(enemy) ×
  #throwers(us)" and every other count interaction is a sum of cross pairs, so it *is* the role-vs-role term.
  "No frontline" = (1−t1)(1−t2)(1−t3) expands into per-brawler terms + pairwise tank–tank synergy + a
  triple term that never occurs in data (3 tanks), so it's covered by the role-synergy terms without a
  non-additive leaf feature. Wall-break presence on walled maps wasn't added: we have no map geometry. The
  similar-map and openness terms carry the map-style signal instead.
- **Explanations:** `comp_notes` (role-level parts of the comp logit, e.g. "Their 3 divers hard-counter
  your 3 throwers (−40%)") per pick rec, at the response top level, and in `final`. The pick explanation
  text appends the top note. Shown as a win-% change from an even game (sigmoid scale).
- **Matchup sanity checks:** `checks/matchups.yaml` (domain knowledge; never training data). `audit.py`
  reports model effect vs raw head-to-head samples in `reference/MATCHUP_CHECKS.md`. A lower pass rate fails
  an experiment. When ≥ 200 raw games point the other way (z ≤ −2) and the model fails, it's labelled
  "data disagrees" rather than forced.

### Process

- **Snapshots:** `load_matches(max_inserted=...)` freezes the data by crawler insertion time. Filtering on
  game time doesn't freeze anything, because battlelogs keep delivering older games.
- **Draft-only log loss vs debiasing:** the skill controls *cost* a little holdout draft-only log loss
  (≈ +0.001 with `lam_player` 10). In the holdout, rare picks are still played by the mains who win with
  them, so a biased model "predicts" better. They were kept (with `lam_player` 40) because they're the point
  of fix (a), and the full v2 model still beats the pre-fix model on the same snapshot.
- **Profiles fetch:** `profiles.py` at ≤ 1 req/s, separate SQLite file, resumable, prioritized by recent
  Legendary+ games covered, with a 3× boost for players seen on < 2%-pick-rate brawlers (where specialist
  bias lives). Coverage is partial (see EXPERIMENTS.md). Rows without a profile get 0 + a coverage
  indicator. On 403 it stops; after 10 consecutive errors it backs off 5 minutes.
- **Running server:** the pickled model format changed (v2), so a server process started before this change
  can't use the new pickle with its old in-memory code. It was restarted once after retraining.

### Results (same frozen snapshot, 214,091 games; holdout 30,802 post-patch Legendary+ games)

- Holdout draft-only log loss **0.6722 → 0.6709** (Δ −0.00125 ± 0.0005, paired bootstrap), Brier 0.2398 →
  0.2391, accuracy 57.9% → 58.6% (map-WR baseline 0.6767). With rank: 0.6633 → 0.6621.
- Calibration (draft-only, both orientations; predicted / observed): 0.26/0.28, 0.36/0.38, 0.43/0.43,
  0.48/0.47, 0.53/0.53, 0.57/0.57, 0.64/0.63, 0.74/0.72. Very slightly overconfident in the tails (≤ 2
  points). No extra temperature scaling added.
- Audit: **26/26 → 0/26 maps flagged** (75 → 0 flags). Mean top-10 overlap with the 15 most picked went
  2.5 → 6.6, and with picked-and-winning 3.4 → 8.0. Matchup checks 17/17 (the pre-fix model with the new
  engine: 11/17).
- Displayed win%: the typical-pick baseline averages 53.0% (was 63.1%), and the top rec is ~+3.5–4 points
  above it at every stage of the draft. First picks average 57.6% with the current comp at 50%. The
  remaining gap vs the *partial* comp (`current_win_prob`) comes from the value of the picks still to be made
  (e.g. the last-pick counter). It's not inflation, so the API now also returns **`expected_win_prob`**: what
  the draft is worth from here if every remaining pick is realistic. That's the apples-to-apples reference
  for `win_prob`.
- First-pick side: after the opponent's most likely first pick, our typical pick sits at ~46% and the top
  rec at ~50%. The model sees a real first-pick edge, because the best brawler on a map, e.g. 8-Bit on Bridge
  Too Far (59.6% raw over 3,168 games), goes first.
- Spearman with AshBS tiers: **0.00 → 0.32** (p = 0.0008); pick rate alone is 0.79.
- **Bridge Too Far:** before, 8-Bit, Clancy (12 games), Bolt (15), Ollie (0), Bull (64). After: 8-Bit, Colt,
  Nori, Melodie, Colette, all with 1,900–4,800 games, picked in 24–59% of games, and winning (except
  Colette, 50%). The user expected "mostly long-range"; the data disagrees in part. Post-patch, the
  long-range snipers *lose* on this map (Brock 46%, Piper 48%, Pierce 43%, Mandy 42%, Byron 41%, several
  of them nerfed in the patch), while Melodie and Nori win ~56% over thousands of games. The engine follows
  that rather than a hardcoded preference. The test checks "mostly ranged (sniper / thrower / mid-range DPS
  or learned range above median), no tanks, none of Ollie/Bull/Clancy/Bolt".
- Skill-control coefficients: rank tier +1.96 per 3 tiers, Elo +0.15 logit per sd (sd 856), profile coverage
  indicator +0.12, "main" share +0.057 per log-unit, player-effect sd 0.044 (81k players). Brawler
  highestTrophies came out ≈ 0 (+0.005/sd) with only ~2,200 profiles fetched (~15–20% of player slots).
  The fetch keeps running and the feature becomes more useful as coverage grows. Retrain to pick it up.
- Fit time ~65 s on 218k games (was ~5 s). `/api/recommend` first pick ~240–300 ms, mid-draft 3–40 ms.

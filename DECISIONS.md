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

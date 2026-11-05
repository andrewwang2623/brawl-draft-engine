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

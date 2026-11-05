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

# Brawl Stars ranked draft engine

A data-driven draft assistant for Brawl Stars **ranked** (Legendary+). You give it the map, which
side picks first, the bans, and the picks so far (ranked 1-2-2-1 order). It recommends the best
next pick and the best bans, and explains each one: map strength, synergy, counters against current
enemies, and counter-pick risk from enemy picks still to come.

```
crawler.py  ──► data/matches.db ──► model.py (comp strength) ──► draft.py (expectimax) ──► server.py + web/
 (official API)     (SQLite)          train.py / evaluate.py                                (local UI)
```

## Highlights

- **Data pipeline:** a resumable crawler on the official Brawl Stars API. It has collected
  ~456k decisive ranked games (~415k Legendary+, all after the latest patch) from ~174k players.
- **Comp model:** regularized, team-antisymmetric logistic regression with per-brawler map, mode,
  and global terms, pairwise synergy and counters, and controls for player skill. Every
  recommendation breaks down exactly into per-brawler and per-pair contributions.
- **Draft search:** expectimax over the remaining picks against a *realistic* (not perfect)
  opponent. It accounts for counter-pick risk, the intent behind enemy bans, and a
  safe/balanced/greedy risk setting.
- **Web UI:** enter slots in any order, bans per team, a "Suggest for me" button, personal mode
  (owned brawlers, power levels, buffies), themes and colorblind colors, and a We won / We lost
  draft log.
- **Evaluation discipline:** time-based holdout, paired bootstrap experiments, and a
  recommendation audit on every map plus domain matchup checks. A change is reverted if the audit
  regresses, even when log loss improves.

### Results (post-patch Legendary+ holdout, newest 15% of games)

| metric | value |
|---|---|
| holdout log loss | **0.6717** (coin flip 0.6931; per-map win-rate baseline 0.6753) |
| accuracy / Brier | 58.4% / 0.2395 |
| calibration | within ~2 pts in every bin |
| recommendation audit | 0 / 26 maps flagged |
| matchup sanity checks | 19 / 19 |

Player skill dominates ranked outcomes, so draft-only accuracy realistically tops out around 55–60%.

## Repository layout

| path | what it is |
|---|---|
| `crawler.py`, `bsapi.py`, `run_crawler.sh` | API client + crawler → SQLite (`data/`, not committed) |
| `model.py`, `train.py`, `evaluate.py` | comp-strength model, training, holdout evaluation |
| `draft.py` | expectimax draft engine, ban scoring, enemy-pick prediction |
| `server.py`, `web/index.html` | FastAPI backend + single-page UI |
| `personal.py`, `profiles.py`, `my_brawlers.yaml` | personal mode + player-skill profiles |
| `audit.py`, `experiment.py`, `checks/` | recommendation audit, tuning experiments, matchup checks |
| `roles.yaml`, `patch.json`, `params.json` | role tags, patch config, hyperparameters |
| `meta/`, `reference/`, `samples/` | brawler/map metadata, tier-list sanity data, sample API payloads |
| `tests/` | pytest suite (model + draft search) |
| `DECISIONS.md`, `EXPERIMENTS.md`, `STATUS.md` | design log, experiment history, current status |

The match database (~1 GB) and trained models are gitignored. Run the crawler and `train.py` to
rebuild them.

## Setup

Requires Python 3 (developed on 3.13) and a Brawl Stars API key from
[developer.brawlstars.com](https://developer.brawlstars.com) (keys are locked to an IP).

```bash
git clone https://github.com/andrewwang2623/brawl-draft-engine.git
cd brawl-draft-engine
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

### Iteration loop: log loss AND the recommendation audit

```bash
.venv/bin/python audit.py                     # top-10 first picks on EVERY map + flags -> models/audit_latest.json
.venv/bin/python audit.py --map "Bridge Too Far"   # full top 10 with every score component
.venv/bin/python experiment.py --desc "..." --set k=v --apply --snapshot <unix time>
```

`audit.py` flags top-5 picks with < 30 games on the map, 0-game brawlers with a large positive map
term, and maps whose top 10 barely overlaps the 15 most-picked brawlers. It also runs the matchup
sanity checks in `checks/matchups.yaml` (domain knowledge, never training data; results in
`reference/MATCHUP_CHECKS.md`). `experiment.py` trains baseline and candidate on the same frozen
snapshot (`--snapshot` = crawler `inserted_at` cutoff), compares holdout log loss with a paired
bootstrap, and audits both. **An experiment fails (is reverted) if the audit regresses** (more
flagged maps or flags, or a lower matchup pass rate), even when log loss improves. Before/after
history: `reference/AUDIT.md`. Role tags for brawlers (used for role-vs-role matchups and comp
notes) live in `roles.yaml`; edit and retrain. Player profiles for the skill controls are
cached by `profiles.py` in `data/profiles.db` (≤ 1 req/s, resumable).

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

API: `GET /api/meta`, `POST /api/recommend`, `GET /api/my_stats`, `GET/PUT /api/my_buffies`,
`POST /api/drafts`, `GET /api/drafts?limit=` (see `server.py`).
Pick recs carry `win_prob`, `win_prob_p25` (same 0–100 scale; ranking key), `confidence`
(low/medium/high) and `comp_notes`. The response has `comp_notes` for the current comps and
`expected_win_prob` (the draft's value if all remaining picks are realistic, which is the fair
reference for a pick's `win_prob`).

### Reading enemy bans + draft log

Bans hint at a team's plan (banning three divers → likely throwers). Enter bans per team (`our_bans` /
`their_bans`; the old `bans` field still works, without intent). The enemy's pick model gets
`exp(gamma · intent(c))`, where intent = surprise-weighted positive counters of their bans on c,
normalized per map. Meta bans count little. The "read off/low/med" toggle next to "Their bans" (API:
`ban_intent`, default low = 0.10, medium = 0.20) affects next-pick recs, likely enemy picks and ban
recs. When the effect is noticeable, the response has a plain-words `ban_read`, e.g. "They banned
Mortis/Shade/Leon → likely throwers (15% → 22% of their next pick); divers gain ≈+0.8%; Pearl now
+1.4%". Every draft with ≥ 1 pick is logged to `data/drafts.db` (separate from the crawler's DB), with
"We won / We lost" buttons once all picks are in. That log is for fitting gamma later
(`.venv/bin/python drafts.py` lists recent drafts). Details: DECISIONS.md ("Ban intent + draft log").

## Reference data

`reference/` holds AshBS's Nov 2026 tier list (image + `ashbs_tiers.csv`, normalized and
cross-checked against the image by `match_tierlist.py`). `sanity_check.py` compares it with the
model and writes `SANITY_CHECK.md`. It's a sanity check only, never training data.

## How it works

**Comp model** (`model.py`): regularized logistic regression, antisymmetric in the two teams:

    logit P(A wins) = Σ_A S(a) − Σ_B S(b) + Σ_pairs∈A syn − Σ_pairs∈B syn + Σ_{a∈A,b∈B} C(a,b) [+ rank control]
    S(x) = global(x) + mode(x) + map(x) + similar-maps(x) + pick-rate prior + range×openness

Separate L2 penalties per layer amount to Bayesian shrinkage map → mode → global; sparse cells shrink
toward performance on maps with similar Legendary+ pick patterns and start from a pick-rate prior.
Counters/synergy pairs shrink toward role-level (roles.yaml) and latent-embedding matchups. Player
skill (tier, Elo, player effect, "main" signal) is controlled for in training and zeroed at draft time.
See DECISIONS.md ("Recommendation fix"). Games are recency-weighted. For the
Legendary+ bracket, lower-tier games inform only shared terms, at reduced weight (a weak prior).
Every logit is a sum of per-brawler and per-pair terms, so each pick's contribution decomposes exactly.

**Draft engine** (`draft.py`): expectimax over the remaining picks. The pick being recommended is
a max node; every other pick (opponents and, by default, our solo-queue teammates) is a chance node
drawn from a realistic distribution, ∝ map pick rate × exp(β·gain), rather than perfect play. Picks
are ranked by `win_prob_p25` (candidate at the 25th percentile of its posterior), with a
`confidence` label. Because of this, first picks favor
brawlers that can't easily be countered, and last picks are pure counters. Each recommendation reports
expected win%, delta vs a typical pick, and a breakdown: map strength / synergy /
counters vs current enemies / vs likely future enemy picks (counter risk). Bans are scored by how
much removing a brawler raises our expected draft value, from the actual pick side.

Win% numbers assume equal skill and an opponent who drafts "realistically". Expect real
prediction accuracy to top out around 55–60%, because player skill dominates.

## Counter-pick risk

Each pick suggestion shows its **biggest realistic counter** ("Biggest risk: Gale -4.6% · 20%
chance, their last pick"). The **risk** setting (safe / balanced / greedy, under "Your pick")
controls how much that exposure lowers a pick's rank. Teams' last picks are modeled as deliberate
counter-picks. Tools: `replay_drafts.py` (replay logged drafts through the engine, compare
before/after) and `counter_check.py` (learned vs. observed counter sizes).

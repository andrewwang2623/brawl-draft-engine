"""Snowball crawler for Brawl Stars ranked (draft) matches.

Seeds from global + country top-player rankings, then crawls players found in
ranked battles. All state lives in SQLite (data/matches.db) so the crawler can
be killed and restarted at any time.

Usage:
    .venv/bin/python crawler.py               # run forever
    .venv/bin/python crawler.py --max-requests 200   # short run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bsapi import ApiError, Client

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DB_PATH = DATA / "matches.db"
LOG_PATH = DATA / "crawl.log"
PID_PATH = DATA / "crawler.pid"

# Verified in Phase 0 (see samples/ and DECISIONS.md): "soloRanked" is the
# ranked draft queue. "ranked" is the trophy ladder and is NOT kept.
# "teamRanked" is accepted in case the team queue reports separately.
RANKED_TYPES = {"soloRanked", "teamRanked"}

SEED_COUNTRIES = ["global", "US", "KR", "BR", "FR", "DE", "ES", "IT", "PL", "TR",
                  "MX", "JP", "GB", "CA", "RU", "UA", "AR", "CL", "PE", "CO",
                  "SA", "AE", "EG", "IN", "ID", "PH", "VN", "TH", "MY", "NL",
                  "BE", "PT", "SE", "FI", "CN", "TW", "HK", "AU"]
RESEED_SECONDS = 6 * 3600
MY_POLL_SECONDS = 20 * 60
STOP = False


SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    battle_key   TEXT PRIMARY KEY,   -- battleTime + sorted player tags
    battle_time  TEXT NOT NULL,      -- raw API string, e.g. 20261104T113608.000Z
    ts           INTEGER NOT NULL,   -- unix seconds
    event_id     INTEGER,
    map          TEXT,
    mode         TEXT,
    battle_type  TEXT,
    duration     INTEGER,
    winner       INTEGER,            -- 0 = team A won, 1 = team B won, NULL = draw
    star_tag     TEXT,
    source_tag   TEXT,
    inserted_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_matches_ts ON matches(ts);
CREATE INDEX IF NOT EXISTS idx_matches_map ON matches(map);

CREATE TABLE IF NOT EXISTS match_players (
    battle_key   TEXT NOT NULL,
    team         INTEGER NOT NULL,   -- 0 = team A (contains smallest tag), 1 = team B
    slot         INTEGER NOT NULL,
    player_tag   TEXT NOT NULL,
    brawler_id   INTEGER,
    brawler      TEXT,
    power        INTEGER,
    rank         INTEGER,            -- 'trophies' field in ranked = rank tier
    PRIMARY KEY (battle_key, team, slot)
);
CREATE INDEX IF NOT EXISTS idx_mp_tag ON match_players(player_tag);

CREATE TABLE IF NOT EXISTS players (
    tag            TEXT PRIMARY KEY,
    name           TEXT,
    max_rank       INTEGER DEFAULT 0,
    source         TEXT,
    first_seen     INTEGER,
    last_polled    INTEGER,
    times_polled   INTEGER DEFAULT 0,
    last_ranked_n  INTEGER DEFAULT 0,   -- ranked games in last poll
    last_new_n     INTEGER DEFAULT 0,   -- new ranked games inserted in last poll
    fail_count     INTEGER DEFAULT 0,
    dead           INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_players_poll ON players(dead, last_polled);

CREATE TABLE IF NOT EXISTS type_counts (
    battle_type TEXT, mode TEXT, n INTEGER DEFAULT 0,
    PRIMARY KEY (battle_type, mode)
);

CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);

-- Personal history: ALL of the owner's games (any type), since the API keeps only 25.
CREATE TABLE IF NOT EXISTS my_games (
    battle_time  TEXT PRIMARY KEY,
    ts           INTEGER NOT NULL,
    battle_type  TEXT,
    mode         TEXT,
    map          TEXT,
    event_id     INTEGER,
    result       TEXT,               -- victory / defeat / draw / NULL (showdown uses rank)
    my_brawler   TEXT,
    my_power     INTEGER,
    trophy_change INTEGER,
    raw_json     TEXT NOT NULL
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    return con


def parse_time(s: str) -> int:
    return int(datetime.strptime(s, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc).timestamp())


def battle_key(battle_time: str, tags: list[str]) -> str:
    return battle_time + "|" + ",".join(sorted(tags))


def parse_battle(item: dict, owner_tag: str) -> dict | None:
    """Turn one battlelog item into a canonical match dict, or None if unusable."""
    b = item.get("battle", {})
    if b.get("type") not in RANKED_TYPES:
        return None
    teams = b.get("teams")
    if not teams or len(teams) != 2 or any(len(t) != 3 for t in teams):
        return None
    ev = item.get("event", {}) or {}
    tags = [p["tag"] for t in teams for p in t]
    if len(set(tags)) != 6:
        return None
    # Canonical orientation: team A contains the lexicographically smallest tag.
    smallest = min(tags)
    a_idx = 0 if any(p["tag"] == smallest for p in teams[0]) else 1
    team_a, team_b = teams[a_idx], teams[1 - a_idx]
    owner_in_a = any(p["tag"] == owner_tag for p in team_a)
    owner_in_b = any(p["tag"] == owner_tag for p in team_b)
    if not (owner_in_a or owner_in_b):
        return None
    res = b.get("result")
    if res == "victory":
        winner = 0 if owner_in_a else 1
    elif res == "defeat":
        winner = 1 if owner_in_a else 0
    elif res == "draw":
        winner = None
    else:
        return None

    def players(team):
        out = []
        for p in team:
            br = p.get("brawler", {}) or {}
            out.append({
                "tag": p["tag"], "name": p.get("name"),
                "brawler_id": br.get("id"), "brawler": br.get("name"),
                "power": br.get("power"), "rank": br.get("trophies"),
            })
        return out

    bt = item["battleTime"]
    return {
        "battle_key": battle_key(bt, tags),
        "battle_time": bt,
        "ts": parse_time(bt),
        "event_id": ev.get("id"),
        "map": ev.get("map"),
        "mode": ev.get("mode") or b.get("mode"),
        "battle_type": b.get("type"),
        "duration": b.get("duration"),
        "winner": winner,
        "star_tag": (b.get("starPlayer") or {}).get("tag"),
        "teams": [players(team_a), players(team_b)],
    }


def store_match(con: sqlite3.Connection, m: dict, source_tag: str) -> bool:
    cur = con.execute(
        "INSERT OR IGNORE INTO matches (battle_key, battle_time, ts, event_id, map, mode, "
        "battle_type, duration, winner, star_tag, source_tag, inserted_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (m["battle_key"], m["battle_time"], m["ts"], m["event_id"], m["map"], m["mode"],
         m["battle_type"], m["duration"], m["winner"], m["star_tag"], source_tag, int(time.time())))
    if cur.rowcount == 0:
        return False
    rows = []
    for t, team in enumerate(m["teams"]):
        for s, p in enumerate(team):
            rows.append((m["battle_key"], t, s, p["tag"], p["brawler_id"], p["brawler"],
                         p["power"], p["rank"]))
    con.executemany("INSERT OR IGNORE INTO match_players VALUES (?,?,?,?,?,?,?,?)", rows)
    return True


def upsert_player(con, tag, name=None, rank=0, source="snowball"):
    now = int(time.time())
    con.execute(
        "INSERT INTO players (tag, name, max_rank, source, first_seen) VALUES (?,?,?,?,?) "
        "ON CONFLICT(tag) DO UPDATE SET max_rank = MAX(COALESCE(max_rank,0), excluded.max_rank), "
        "name = COALESCE(excluded.name, name)",
        (tag, name, rank or 0, source, now))


def seed(con, client, log):
    added = 0
    for country in SEED_COUNTRIES:
        try:
            r = client.rankings(country, 200)
        except ApiError as e:
            if e.reason in ("accessDenied", "accessDenied.invalidIp"):
                raise
            log.warning("seed %s failed: %s", country, e)
            continue
        for p in r.get("items", []):
            upsert_player(con, p["tag"], p.get("name"), 0, "seed:" + country)
            added += 1
        con.commit()
        if STOP:
            break
    con.execute("INSERT OR REPLACE INTO meta VALUES ('last_seed', ?)", (str(int(time.time())),))
    con.commit()
    log.info("seeded %d ranking entries from %d lists", added, len(SEED_COUNTRIES))


def next_players(con, n=50) -> list[str]:
    """Pick the next batch of players to poll.

    Mix of (a) never-polled players, highest known rank first, and (b) re-polls of
    players due according to how active they were in ranked last time. Battle logs
    only hold the last 25 games, so active high-rank players are re-polled every
    ~2h, moderately active every ~8h, and inactive ones only daily.
    """
    now = int(time.time())
    fresh = [r[0] for r in con.execute(
        "SELECT tag FROM players WHERE dead=0 AND last_polled IS NULL "
        "ORDER BY max_rank DESC, first_seen ASC LIMIT ?", (n,))]
    due = [r[0] for r in con.execute(
        """SELECT tag FROM players WHERE dead=0 AND last_polled IS NOT NULL AND
           last_polled < ? - CASE
               WHEN last_new_n >= 5 AND max_rank >= 13 THEN 2*3600
               WHEN last_new_n >= 2 THEN 6*3600
               WHEN last_ranked_n >= 1 THEN 12*3600
               ELSE 48*3600 END
           ORDER BY (last_new_n * 3 + max_rank) DESC LIMIT ?""", (now, n))]
    # ~65% fresh, ~35% re-poll when both are available
    k_due = min(len(due), max(n - len(fresh), int(n * 0.35)))
    batch = due[:k_due] + fresh[: n - k_due]
    random.shuffle(batch)
    return batch


def poll_player(con, client, tag, log) -> tuple[int, int]:
    """Fetch one battle log; return (ranked_games_seen, new_matches_inserted)."""
    now = int(time.time())
    try:
        bl = client.battlelog(tag)
    except ApiError as e:
        if e.status == 403 or e.reason.startswith("accessDenied"):
            raise
        if e.status == 404:
            con.execute("UPDATE players SET last_polled=?, fail_count=fail_count+1, "
                        "dead = CASE WHEN fail_count >= 2 THEN 1 ELSE 0 END WHERE tag=?", (now, tag))
        else:
            log.warning("battlelog %s failed: %s", tag, e)
            con.execute("UPDATE players SET last_polled=?, fail_count=fail_count+1 WHERE tag=?", (now, tag))
        return 0, 0
    ranked = new = 0
    for item in bl.get("items", []):
        b = item.get("battle", {})
        con.execute("INSERT INTO type_counts VALUES (?,?,1) ON CONFLICT(battle_type, mode) "
                    "DO UPDATE SET n=n+1", (b.get("type"), b.get("mode")))
        m = parse_battle(item, tag)
        if m is None:
            continue
        ranked += 1
        if store_match(con, m, tag):
            new += 1
        for team in m["teams"]:
            for p in team:
                upsert_player(con, p["tag"], p["name"], p["rank"] or 0)
    con.execute("UPDATE players SET last_polled=?, times_polled=times_polled+1, last_ranked_n=?, "
                "last_new_n=?, fail_count=0 WHERE tag=?", (now, ranked, new, tag))
    return ranked, new


def my_tag() -> str | None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    t = os.environ.get("BRAWL_PLAYER_TAG", "").strip().upper()
    if not t:
        return None
    return t if t.startswith("#") else "#" + t


def poll_me(con, client, tag, log) -> int:
    """Save every game in the owner's battle log to my_games. Returns new rows."""
    try:
        bl = client.battlelog(tag)
    except ApiError as e:
        if e.status == 403 or e.reason.startswith("accessDenied"):
            raise
        log.warning("my battlelog failed: %s", e)
        return 0
    new = 0
    for item in bl.get("items", []):
        b = item.get("battle", {}) or {}
        ev = item.get("event", {}) or {}
        me = None
        for p in [p for t in (b.get("teams") or []) for p in t] + list(b.get("players") or []):
            if p.get("tag", "").upper() == tag:
                me = p
        br = (me or {}).get("brawler") or {}
        if not br and me and me.get("brawlers"):
            br = me["brawlers"][0]
        cur = con.execute(
            "INSERT OR IGNORE INTO my_games VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (item["battleTime"], parse_time(item["battleTime"]), b.get("type"),
             ev.get("mode") or b.get("mode"), ev.get("map"), ev.get("id"), b.get("result"),
             br.get("name"), br.get("power"), b.get("trophyChange"), json.dumps(item)))
        new += cur.rowcount
        # The owner's ranked games are also normal training data.
        m = parse_battle(item, tag)
        if m is not None:
            store_match(con, m, tag)
    con.commit()
    log.info("my battlelog: %d new games saved", new)
    return new


def stats(con) -> dict:
    total, = con.execute("SELECT COUNT(*) FROM matches").fetchone()
    maps, = con.execute("SELECT COUNT(DISTINCT map) FROM matches").fetchone()
    hour_ago = int(time.time()) - 3600
    last_hour, = con.execute("SELECT COUNT(*) FROM matches WHERE inserted_at >= ?", (hour_ago,)).fetchone()
    players, = con.execute("SELECT COUNT(*) FROM players").fetchone()
    polled, = con.execute("SELECT COUNT(*) FROM players WHERE last_polled IS NOT NULL").fetchone()
    return dict(total=total, maps=maps, last_hour=last_hour, players=players, polled=polled)


def write_access_problem(reason: str):
    note = (f"\n## Crawler blocked ({datetime.now().isoformat(timespec='minutes')})\n"
            f"API returned `{reason}`. The key is IP-locked; the current network IP is probably "
            f"not whitelisted. Crawler stopped; restart with `./run_crawler.sh` once fixed.\n")
    with open(ROOT / "STATUS.md", "a") as f:
        f.write(note)


def setup_logging() -> logging.Logger:
    DATA.mkdir(exist_ok=True)
    log = logging.getLogger("crawler")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fh = logging.FileHandler(LOG_PATH)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
    return log


def main():
    global STOP
    ap = argparse.ArgumentParser()
    ap.add_argument("--rps", type=float, default=4.0)
    ap.add_argument("--max-requests", type=int, default=0, help="stop after N requests (0 = forever)")
    args = ap.parse_args()

    log = setup_logging()

    def handle(sig, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    PID_PATH.write_text(str(os.getpid()))
    con = connect()
    client = Client(rps=args.rps)
    log.info("crawler start pid=%d rps=%.1f %s", os.getpid(), args.rps, stats(con))

    last_report = time.time()
    me = my_tag()
    last_me = 0.0
    try:
        last_seed = con.execute("SELECT v FROM meta WHERE k='last_seed'").fetchone()
        if not last_seed or time.time() - int(last_seed[0]) > RESEED_SECONDS:
            seed(con, client, log)
        polls = 0
        while not STOP:
            batch = next_players(con)
            if not batch:
                log.info("queue empty; sleeping 5 min")
                time.sleep(300)
                continue
            for tag in batch:
                if STOP:
                    break
                if me and time.time() - last_me > MY_POLL_SECONDS:
                    poll_me(con, client, me, log)
                    last_me = time.time()
                poll_player(con, client, tag, log)
                polls += 1
                if polls % 10 == 0:
                    con.commit()
                if args.max_requests and client.requests_made >= args.max_requests:
                    STOP = True
                if time.time() - last_report > 300:
                    con.commit()
                    s = stats(con)
                    log.info("progress total=%d new_last_hour=%d distinct_maps=%d players=%d "
                             "polled=%d requests=%d", s["total"], s["last_hour"], s["maps"],
                             s["players"], s["polled"], client.requests_made)
                    last_report = time.time()
            con.commit()
            last_seed = con.execute("SELECT v FROM meta WHERE k='last_seed'").fetchone()
            if time.time() - int(last_seed[0]) > RESEED_SECONDS:
                seed(con, client, log)
    except ApiError as e:
        log.error("access problem, stopping: %s %s", e.status, e.reason)
        write_access_problem(e.reason or str(e.status))
    except Exception:
        log.exception("crawler crashed")
        raise
    finally:
        con.commit()
        log.info("crawler stop %s requests=%d", stats(con), client.requests_made)
        con.close()
        try:
            PID_PATH.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()

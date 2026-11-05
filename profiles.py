"""Player-profile cache for the skill / brawler-experience controls.

Battlelogs only carry each player's ranked tier. /players/{tag} adds `rankedElo` (finer
skill) and, per brawler, `trophies` / `highestTrophies` (how much the player has played that
brawler: the "main" signal behind specialist bias). This script fetches profiles for players
seen in our matches and caches them in a SEPARATE SQLite file (data/profiles.db), so it never
contends with the crawler's writes to matches.db.

    .venv/bin/python profiles.py --max 3000          # resumable; <= 1 req/s
    .venv/bin/python profiles.py --status

Priority: players covering the most recent Legendary+ games first, with a boost for players
seen on rarely picked brawlers (where specialist bias matters most). The API key is loaded by
bsapi.Client from .env and is never printed or logged.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROFILES_DB = ROOT / "data" / "profiles.db"
MATCHES_DB = ROOT / "data" / "matches.db"


def connect(path: Path = PROFILES_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.execute("""CREATE TABLE IF NOT EXISTS profiles (
        tag TEXT PRIMARY KEY, fetched_at INTEGER, status INTEGER,
        ranked_elo INTEGER, ranked_rank INTEGER, highest_ranked_elo INTEGER,
        trophies INTEGER, brawlers TEXT)""")
    return con


def load_profiles(path: Path = PROFILES_DB) -> dict:
    """tag -> {'elo': int|None, 'rank': int|None, 'b': {brawler_id: (trophies, highest)}}.
    Read-only; returns {} if the cache doesn't exist yet."""
    if not path.exists():
        return {}
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    out = {}
    for tag, elo, rank, br in con.execute(
            "SELECT tag, ranked_elo, ranked_rank, brawlers FROM profiles WHERE status = 200"):
        b = {int(k): tuple(v) for k, v in json.loads(br or "{}").items()}
        out[tag] = {"elo": elo, "rank": rank, "b": b}
    con.close()
    return out


def priority_tags() -> list[str]:
    """Tags ordered by (recent Legendary+ games covered) with a boost for rare-brawler players."""
    con = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True, timeout=30)
    rows = con.execute("""
        SELECT p.player_tag, p.brawler_id, m.ts FROM match_players p
        JOIN matches m ON m.battle_key = p.battle_key
        WHERE m.battle_type IN ('soloRanked','teamRanked') AND m.tier_min >= 16
          AND m.winner IS NOT NULL""").fetchall()
    con.close()
    if not rows:
        return []
    t_max = max(r[2] for r in rows)
    pick = Counter(r[1] for r in rows)
    total = sum(pick.values())
    score = Counter()
    for tag, bid, ts in rows:
        rec = 0.5 ** ((t_max - ts) / (2 * 86400.0))            # 2-day half-life
        rare = 3.0 if 6 * pick[bid] / total < 0.02 else 1.0     # brawler in < 2% of games
        score[tag] += rec * rare
    return [t for t, _ in score.most_common()]


def fetch(max_n: int, rps: float = 1.0):
    from bsapi import ApiError, Client
    cli = Client(rps=min(rps, 1.0))
    con = connect()
    done = {r[0] for r in con.execute("SELECT tag FROM profiles")}
    todo = [t for t in priority_tags() if t not in done][:max_n]
    print(f"{len(done)} cached, fetching {len(todo)}", flush=True)
    errs = 0
    for k, tag in enumerate(todo):
        try:
            p = cli.player(tag)
            br = {int(b["id"]): [int(b.get("trophies", 0)), int(b.get("highestTrophies", 0))]
                  for b in p.get("brawlers", [])}
            con.execute("INSERT OR REPLACE INTO profiles VALUES (?,?,?,?,?,?,?,?)",
                        (tag, int(time.time()), 200, p.get("rankedElo"), p.get("rankedRank"),
                         p.get("highestAllTimeRankedElo"), p.get("trophies"),
                         json.dumps(br, separators=(",", ":"))))
            errs = 0
        except ApiError as e:
            con.execute("INSERT OR REPLACE INTO profiles (tag, fetched_at, status) VALUES (?,?,?)",
                        (tag, int(time.time()), e.status))
            if e.status == 403:
                print("403 accessDenied/invalidIp: key not valid for this IP; stopping.", flush=True)
                con.commit()
                return
            errs += 1
            if errs >= 10:
                print(f"{errs} consecutive errors (last {e.status}); backing off 5 min", flush=True)
                con.commit()
                time.sleep(300)
                errs = 0
        if k % 50 == 0:
            con.commit()
            print(f"{k}/{len(todo)}", flush=True)
    con.commit()
    con.close()


def status():
    con = connect()
    for st, n in con.execute("SELECT status, COUNT(*) FROM profiles GROUP BY status"):
        print(f"status {st}: {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=3000)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    status() if a.status else fetch(a.max)

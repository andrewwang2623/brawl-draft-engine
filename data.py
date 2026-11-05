"""Load ranked matches from SQLite into compact arrays for modelling."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "matches.db"
META = ROOT / "meta"


def load_brawler_meta() -> dict[int, dict]:
    raw = json.loads((META / "brawlers.json").read_text())
    return {int(k): v for k, v in raw.items()}


def load_map_meta() -> dict[int, dict]:
    raw = json.loads((META / "maps.json").read_text())
    return {int(k): v for k, v in raw.items()}


@dataclass
class Matches:
    ts: np.ndarray            # (N,) unix seconds
    map: list[str]            # (N,) map name
    mode: list[str]           # (N,) mode string as in API (e.g. 'brawlBall')
    team_a: np.ndarray        # (N, 3) brawler ids
    team_b: np.ndarray        # (N, 3) brawler ids
    y: np.ndarray             # (N,) 1 if team A won
    rank_a: np.ndarray        # (N,) mean ranked tier of team A
    rank_b: np.ndarray        # (N,)
    tier_min: np.ndarray = None   # (N,) lowest ranked tier of the 6 players (16 = Legendary I)
    keys: list[str] = field(default_factory=list)
    tags_a: list[tuple] = field(default_factory=list)
    tags_b: list[tuple] = field(default_factory=list)

    def __len__(self):
        return len(self.y)

    def subset(self, idx) -> "Matches":
        idx = np.asarray(idx)
        pick = lambda lst: [lst[i] for i in idx]
        return Matches(self.ts[idx], pick(self.map), pick(self.mode), self.team_a[idx],
                       self.team_b[idx], self.y[idx], self.rank_a[idx], self.rank_b[idx],
                       self.tier_min[idx] if self.tier_min is not None else None,
                       pick(self.keys) if self.keys else [],
                       pick(self.tags_a) if self.tags_a else [],
                       pick(self.tags_b) if self.tags_b else [])

    def flipped(self) -> "Matches":
        """Same games with teams swapped (used for symmetry checks)."""
        return Matches(self.ts, self.map, self.mode, self.team_b, self.team_a, 1 - self.y,
                       self.rank_b, self.rank_a, self.tier_min, self.keys, self.tags_b, self.tags_a)


def load_matches(db_path: Path = DB_PATH, battle_types=("soloRanked", "teamRanked"),
                 min_rank: int = 0, max_ts: int | None = None,
                 max_inserted: int | None = None) -> Matches:
    """Load decisive (non-draw) ranked 3v3 games, sorted by time. max_inserted freezes a data
    snapshot (rows the crawler had inserted by that unix time), so paired experiments see the
    same games while the crawler keeps writing. (max_ts filters on game time, which does NOT
    freeze the set: battlelogs keep delivering older games.)"""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    q = f"""
        SELECT m.battle_key, m.ts, m.map, m.mode, m.winner,
               p.team, p.slot, p.brawler_id, p.rank, p.player_tag
        FROM matches m JOIN match_players p ON p.battle_key = m.battle_key
        WHERE m.winner IS NOT NULL AND m.map IS NOT NULL
          AND m.battle_type IN ({",".join("?" * len(battle_types))})
          {"AND m.ts <= ?" if max_ts is not None else ""}
          {"AND m.inserted_at <= ?" if max_inserted is not None else ""}
        ORDER BY m.ts, m.battle_key, p.team, p.slot
    """
    args = tuple(battle_types) + ((int(max_ts),) if max_ts is not None else ()) \
        + ((int(max_inserted),) if max_inserted is not None else ())
    rows = con.execute(q, args).fetchall()
    con.close()

    games = {}
    for key, ts, mp, mode, winner, team, slot, bid, rank, tag in rows:
        g = games.get(key)
        if g is None:
            g = games[key] = {"ts": ts, "map": mp, "mode": mode, "winner": winner,
                              "b": [[], []], "r": [[], []], "t": [[], []]}
        g["b"][team].append(bid)
        g["r"][team].append(rank or 0)
        g["t"][team].append(tag)

    keep = [(k, g) for k, g in games.items()
            if len(g["b"][0]) == 3 and len(g["b"][1]) == 3]
    if min_rank:
        keep = [(k, g) for k, g in keep
                if min(np.mean(g["r"][0]), np.mean(g["r"][1])) >= min_rank]
    keep.sort(key=lambda kg: (kg[1]["ts"], kg[0]))
    n = len(keep)
    return Matches(
        ts=np.array([g["ts"] for _, g in keep], dtype=np.int64),
        map=[g["map"] for _, g in keep],
        mode=[g["mode"] for _, g in keep],
        team_a=np.array([g["b"][0] for _, g in keep], dtype=np.int64).reshape(n, 3),
        team_b=np.array([g["b"][1] for _, g in keep], dtype=np.int64).reshape(n, 3),
        y=np.array([1 - g["winner"] for _, g in keep], dtype=np.float64),
        rank_a=np.array([np.mean(g["r"][0]) for _, g in keep], dtype=np.float64),
        rank_b=np.array([np.mean(g["r"][1]) for _, g in keep], dtype=np.float64),
        tier_min=np.array([min(g["r"][0] + g["r"][1]) for _, g in keep], dtype=np.int64),
        keys=[k for k, _ in keep],
        tags_a=[tuple(g["t"][0]) for _, g in keep],
        tags_b=[tuple(g["t"][1]) for _, g in keep],
    )


TIERS = {16: "Legendary", 19: "Masters", 22: "Pro"}
BRACKETS = {"legendary": 16, "masters": 19, "all": 0}   # bracket name -> min tier (lowest of 6)


def tier_name(t: int) -> str:
    names = ["Bronze", "Silver", "Gold", "Diamond", "Mythic", "Legendary", "Masters"]
    if t >= 22:
        return "Pro"
    if t < 1:
        return "Unranked"
    return f"{names[(t - 1) // 3]} {'I' * ((t - 1) % 3 + 1)}"


def time_split(m: Matches, test_frac: float = 0.15) -> tuple[Matches, Matches]:
    """Train on the oldest games, test on the newest `test_frac`. Input is time-sorted."""
    n = len(m)
    cut = int(round(n * (1 - test_frac)))
    return m.subset(np.arange(cut)), m.subset(np.arange(cut, n))

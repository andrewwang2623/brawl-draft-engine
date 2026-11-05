"""Personal mode: the owner's brawler pool, a per-brawler "me factor", and comfort picks.

Inputs: BRAWL_PLAYER_TAG (.env), the `my_games` table (filled by the crawler every ~20 min,
ALL modes), and /players/{tag} for owned brawlers + power (cached in data/my_player.json).

me factor (logit units, per brawler): how much better the owner does on a brawler than the
global comp model expected for those exact games (map + both comps):
    me_b = sum_i w_i (y_i - p_i) / (sum_i w_i p_i (1 - p_i) + K)
which is a one-step Newton / Laplace estimate of a logit offset under a N(0, 1/K) prior.
K = ME_PRIOR_GAMES pseudo-games, so 5 games barely move it. Ranked games weigh 1.0, other
3v3 modes OTHER_MODE_WEIGHT; recency half-life ME_HALF_LIFE_DAYS. Clipped to ±ME_CAP.

comfort: small bonus growing with log(#recent games on the brawler), capped.
power: brawlers below power 11 get POWER_PENALTY per missing level; unowned are excluded.
buffies: missing buffies (API + my_brawlers.yaml overrides) cost buffie_penalty_pct x the
fraction of the brawler's buffies you lack. Applied only to YOUR pick; teammates and enemies
are assumed fully built.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

from model import sigmoid

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "matches.db"
PLAYER_CACHE = ROOT / "data" / "my_player.json"
MY_BRAWLERS = ROOT / "my_brawlers.yaml"
MY_BUFFIES = ROOT / "reference" / "my_buffies.txt"   # user-edited list of brawlers MISSING buffies
BUFFIES_AVAILABLE = ROOT / "meta" / "buffies_available.json"
BUFFIE_TYPES = ("gadget", "starPower", "hyperCharge")

ME_PRIOR_GAMES = 25.0
OTHER_MODE_WEIGHT = 0.35
ME_HALF_LIFE_DAYS = 30.0
ME_CAP = 0.3                  # ~ ±7.5 win-% points
COMFORT_PER_LOG_GAME = 0.03   # logit per log(1 + games)
COMFORT_CAP = 0.08            # ~ +2%
POWER_PENALTY = 0.12          # logit per level below 11 (~ -3% each)
PROFILE_MAX_AGE = 6 * 3600
RANKED_TYPES = ("soloRanked", "teamRanked")


def my_tag() -> str | None:
    load_dotenv(ROOT / ".env")
    t = os.environ.get("BRAWL_PLAYER_TAG", "").strip().upper()
    if not t:
        return None
    return t if t.startswith("#") else "#" + t


def load_player_profile(tag: str, refresh: bool = True) -> dict | None:
    """Owned brawlers from the API, cached; falls back to the cache if the API fails."""
    if PLAYER_CACHE.exists():
        cached = json.loads(PLAYER_CACHE.read_text())
        fresh = time.time() - PLAYER_CACHE.stat().st_mtime < PROFILE_MAX_AGE
        if fresh or not refresh:
            return cached
    else:
        cached = None
    try:
        from bsapi import Client
        prof = Client(rps=1).player(tag)
        PLAYER_CACHE.write_text(json.dumps(prof))
        return prof
    except Exception:
        return cached


def load_my_brawlers(path: Path = MY_BRAWLERS) -> dict:
    cfg = {"use_api_buffies": True, "missing_buffies": [], "assume_full": [],
           "buffie_penalty_pct": 3.0}
    if path.exists():
        cfg.update(yaml.safe_load(path.read_text()) or {})
    return cfg


def read_buffies_file(path: Path = MY_BUFFIES) -> list[str] | None:
    """Names from my_buffies.txt (one per line, '#' comments). None if the file doesn't exist."""
    if not path.exists():
        return None
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def resolve_name(n: str, name_to_id: dict) -> int | None:
    """Loose name match: 'mr p' -> Mr. P, 'el primo' / 'primo' -> El Primo, '8bit' -> 8-Bit."""
    import re
    norm = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
    table = {norm(k): v for k, v in name_to_id.items()}
    aliases = {"primo": "elprimo", "colonelruffs": "ruffs", "larrylawrie": "larrylawrie", "allie": "alli"}
    k = norm(n)
    return table.get(k) or table.get(aliases.get(k, ""))


def buffie_gaps(prof: dict | None, cfg: dict, name_to_id: dict,
                missing_file: list[str] | None = None) -> dict:
    """brawler_id -> (fraction missing, [missing buffie types], source).

    If my_buffies.txt exists it is authoritative: listed brawlers are missing buffies (which
    types: from the API when it shows them, else all), every other brawler is fully built.
    Otherwise: API detection + my_brawlers.yaml overrides."""
    api_gaps = _api_gaps(prof, name_to_id) if prof else {}
    if missing_file is not None:
        gaps = {}
        for n in missing_file:
            bid = resolve_name(n, name_to_id)
            if bid is None:
                continue
            if bid in api_gaps:
                frac, types, _ = api_gaps[bid]
                gaps[bid] = (frac, types, "my_buffies.txt (types from API)")
            else:
                gaps[bid] = (1.0, list(BUFFIE_TYPES), "my_buffies.txt")
        return gaps
    gaps = dict(api_gaps) if cfg.get("use_api_buffies", True) else {}
    for n in cfg.get("missing_buffies") or []:
        bid = resolve_name(str(n), name_to_id)
        if bid:
            gaps[bid] = (1.0, list(BUFFIE_TYPES), "my_brawlers.yaml")
    for n in cfg.get("assume_full") or []:
        gaps.pop(resolve_name(str(n), name_to_id), None)
    return gaps


def _api_gaps(prof: dict, name_to_id: dict) -> dict:
    avail = {}
    if BUFFIES_AVAILABLE.exists():
        avail = {k.upper(): set(v) for k, v in json.loads(BUFFIES_AVAILABLE.read_text()).items()}
    gaps = {}
    for b in prof.get("brawlers", []):
        have = {k for k, v in (b.get("buffies") or {}).items() if v}
        exists = avail.get(b["name"].upper(), set()) | have
        if not exists:
            continue            # brawler has no buffies in the game
        miss = sorted(exists - have)
        if miss:
            gaps[int(b["id"])] = (len(miss) / len(exists), miss, "api")
    return gaps


@dataclass
class BrawlerStat:
    games: int = 0
    wins: int = 0
    ranked_games: int = 0
    w_resid: float = 0.0     # sum w (y - p)
    w_info: float = 0.0      # sum w p (1 - p)
    w_games: float = 0.0     # recency-weighted games (comfort)
    exp_wins: float = 0.0    # sum p (expected wins under the model)

    @property
    def me(self) -> float:
        return float(np.clip(self.w_resid / (self.w_info + ME_PRIOR_GAMES), -ME_CAP, ME_CAP))

    @property
    def comfort(self) -> float:
        return float(min(COMFORT_CAP, COMFORT_PER_LOG_GAME * np.log1p(self.w_games)))


@dataclass
class Profile:
    available: bool
    tag: str | None = None
    owned: dict = field(default_factory=dict)       # brawler_id -> power
    stats_by: dict = field(default_factory=dict)    # brawler_id -> BrawlerStat
    names: dict = field(default_factory=dict)
    n_games: int = 0
    n_scored: int = 0
    n_ranked: int = 0
    rank_name: str | None = None
    buffies: dict = field(default_factory=dict)     # brawler_id -> (frac, types, source)
    buffie_penalty: float = 0.12                    # logit for missing all buffies
    built_at: float = field(default_factory=time.time)

    def parts(self, bid: int) -> dict:
        """Logit adjustments for the owner playing this brawler."""
        st = self.stats_by.get(bid)
        power = self.owned.get(bid)
        gap = self.buffies.get(bid)
        return {
            "me": st.me if st else 0.0,
            "comfort": st.comfort if st else 0.0,
            "power": -POWER_PENALTY * (11 - power) if power is not None and power < 11 else 0.0,
            "buffies": -self.buffie_penalty * gap[0] if gap else 0.0,
        }

    def bonus(self, bid: int) -> float:
        return float(sum(self.parts(bid).values()))

    def vectors(self, mm):
        """(pool_mask, bonus_vec, info) aligned with a MapModel's brawler index."""
        n = len(mm.brawler_ids)
        mask = np.zeros(n, dtype=bool)
        bonus = np.zeros(n)
        info = {}
        for i, b in enumerate(mm.brawler_ids):
            b = int(b)
            owned = b in self.owned if self.owned else True
            mask[i] = owned
            bonus[i] = self.bonus(b)
            st = self.stats_by.get(b)
            parts = self.parts(b)
            gap = self.buffies.get(b)
            info[b] = {
                "games": st.games if st else 0,
                "ranked_games": st.ranked_games if st else 0,
                "wr": 100.0 * st.wins / st.games if st and st.games else 0.0,
                "me_pct": round(25 * parts["me"], 2),
                "comfort_pct": round(25 * parts["comfort"], 2),
                "power": self.owned.get(b),
                "power_pct": round(25 * parts["power"], 2),
                "buffie_pct": round(25 * parts["buffies"], 2),
                "missing_buffies": gap[1] if gap else [],
            }
        return mask, bonus, info

    def stats(self, k: int = 8) -> dict:
        rows = []
        for b, st in self.stats_by.items():
            if st.games == 0:
                continue
            rows.append({"brawler_id": b, "name": self.names.get(b, str(b)), "games": st.games,
                         "ranked_games": st.ranked_games,
                         "wr": round(100.0 * st.wins / st.games, 1),
                         "expected_wr": round(100.0 * st.exp_wins / st.games, 1),
                         "me_factor_pct": round(25 * st.me, 2),
                         "comfort_pct": round(25 * st.comfort, 2)})
        rows.sort(key=lambda r: -r["me_factor_pct"])
        most = sorted(rows, key=lambda r: -r["games"])[:k]
        low_power = [{"brawler_id": b, "name": self.names.get(b, str(b)), "power": p}
                     for b, p in self.owned.items() if p < 11]
        missing_buffies = [{"brawler_id": b, "name": self.names.get(b, str(b)), "missing": g[1],
                            "penalty_pct": round(-25 * self.buffie_penalty * g[0], 2), "source": g[2]}
                           for b, g in sorted(self.buffies.items(), key=lambda kv: -kv[1][0])]
        return {"available": self.available, "rank": self.rank_name,
                "n_games": self.n_games, "n_scored": self.n_scored, "n_ranked": self.n_ranked,
                "n_owned": len(self.owned),
                "top": rows[:k], "bottom": rows[::-1][:k] if len(rows) > k else [],
                "most_played": most, "low_power": low_power, "missing_buffies": missing_buffies}


def build_profile(md, db_path: Path = DB_PATH, refresh_api: bool = True) -> Profile:
    tag = my_tag()
    if not tag:
        return Profile(available=False)
    names = {int(b): v["name"] for b, v in md.meta.items()}
    name_to_id = {v["name"].upper(): int(b) for b, v in md.meta.items()}
    prof = load_player_profile(tag, refresh=refresh_api)
    owned = {}
    rank_name = None
    if prof:
        owned = {int(b["id"]): int(b.get("power", 11)) for b in prof.get("brawlers", [])}
        rank_name = prof.get("rankedRankName")

    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
        rows = con.execute("SELECT ts, battle_type, mode, map, result, raw_json FROM my_games "
                           "ORDER BY ts").fetchall()
        con.close()
    except sqlite3.Error:
        rows = []

    stats: dict[int, BrawlerStat] = {}
    now = time.time()
    n_scored = n_ranked = 0
    for ts, btype, mode, mp, result, raw in rows:
        item = json.loads(raw)
        teams = (item.get("battle") or {}).get("teams")
        if not teams or len(teams) != 2 or any(len(t) != 3 for t in teams):
            continue          # showdown / 5v5 / duels: no 3v3 comp to score
        if result not in ("victory", "defeat"):
            continue
        mine_i = next((k for k, t in enumerate(teams) if any(p["tag"].upper() == tag for p in t)), None)
        if mine_i is None:
            continue
        me_p = next(p for p in teams[mine_i] if p["tag"].upper() == tag)
        bid = int((me_p.get("brawler") or {}).get("id") or name_to_id.get(
            (me_p.get("brawler") or {}).get("name", "").upper(), 0))
        if not bid:
            continue
        ours = [int(p["brawler"]["id"]) for p in teams[mine_i]]
        theirs = [int(p["brawler"]["id"]) for p in teams[1 - mine_i]]
        mm = md.map_model(mp or "", mode or "")
        try:
            a = [mm.id_to_idx[x] for x in ours]
            b = [mm.id_to_idx[x] for x in theirs]
        except KeyError:
            continue
        p = float(sigmoid(mm.team_logit(a, b)))
        y = 1.0 if result == "victory" else 0.0
        ranked = btype in RANKED_TYPES
        w = (1.0 if ranked else OTHER_MODE_WEIGHT) * 0.5 ** ((now - ts) / (ME_HALF_LIFE_DAYS * 86400))
        st = stats.setdefault(bid, BrawlerStat())
        st.games += 1
        st.wins += int(y)
        st.ranked_games += int(ranked)
        st.w_resid += w * (y - p)
        st.w_info += w * p * (1 - p)
        st.w_games += w
        st.exp_wins += p
        n_scored += 1
        n_ranked += int(ranked)
    cfg = load_my_brawlers()
    gaps = buffie_gaps(prof, cfg, name_to_id, missing_file=read_buffies_file())
    return Profile(available=True, tag=tag, owned=owned, stats_by=stats, names=names,
                   n_games=len(rows), n_scored=n_scored, n_ranked=n_ranked, rank_name=rank_name,
                   buffies=gaps, buffie_penalty=float(cfg["buffie_penalty_pct"]) / 25.0)

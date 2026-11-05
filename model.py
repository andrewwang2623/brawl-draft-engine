"""Comp-strength model: regularized logistic regression with hierarchical shrinkage.

logit P(team A wins) =
      sum_{a in A} S(a) - sum_{b in B} S(b)                    brawler strength on this map
    + sum_{pairs in A} syn(a,a') - sum_{pairs in B} syn(b,b')   teammate synergy
    + sum_{a in A, b in B} C(a, b)                             counters (C antisymmetric)
    + skill controls (zeroed at draft time)

Brawler strength on a map (each layer has its own L2 penalty = Gaussian prior):

    S(x) = g[x] + m[mode,x] + s[map,x]                 global -> mode -> map hierarchy
         + u[x] . e[map]                                 map-similarity prior (see below)
         + beta_pr[mode] * lpr(x, map)                   pick-rate prior (Legendary+ pick share)
         + beta_ro[mode] * range(x) * openness(map)      range x openness
         (+ class x mode + class x map: legacy, off by default)

- Map similarity: each map is represented by its Legendary+ pick-rate vector. A kernel
  K(m, m') = exp(gamma * (cos(m, m') - 1)) over those vectors is factorized as E E^T, and each
  brawler gets coefficients u[x] on the map embedding e[map]. That is a Gaussian-process prior
  with covariance K / lam_sim across maps: sparse brawler/map cells are pulled toward how the
  brawler does on *similar* maps, instead of toward its class.
- Pick-rate prior: lpr = log(smoothed map pick rate + floor), centered per map. Its slope is
  learned. Brawlers almost nobody picks on a map start from a lower prior mean.
- Range x openness: the leading component of the (double-centered) brawler x map log pick-rate
  matrix. Brawler loadings = learned "range" (long-range brawlers are picked on the same maps),
  map loadings = "openness". Brawlify publishes no range stats, so it's learned from picks.
- Low-pick-rate cells are shrunk harder: a cell's penalty is multiplied by
  1 + rare_shrink * n0 / (n + n0), with n its games and n0 = rare_ref * (games in scope).
  Same for synergy / counter pairs (pair_rare_shrink, pair_rare_ref).
- Counters and synergy: per-pair terms shrink toward class-vs-class (counters) and
  class+class (synergy) terms, global + per mode + per map. Those have huge pooled data, so
  they're lightly regularized and learn patterns like "assassins beat throwers". Everything is
  still pairwise, so it folds into the MapModel's Syn / C matrices.

Skill controls (all zeroed at draft time, and in the "draft-only" holdout metric):
rank-tier difference; a per-player skill effect (player tag, L2); rankedElo and the player's
trophies on the brawler they played (from data/profiles.db, with a coverage indicator); and a
"main" signal = how much of a player's games are on that brawler (leave-one-out). They soak up
specialist/selection bias: rare picks are mostly played by mains, which inflates their raw
win rates.

Posterior uncertainty: a block-diagonal Laplace approximation (per brawler, all its strength
terms jointly; per pair for synergy / counters) gives sd(S) per map, used for conservative
(25th percentile) ranking in draft.py.

There's no intercept: the model is antisymmetric under swapping teams. Because the logit is a
sum of per-brawler and per-pair terms, any pick's contribution decomposes exactly (MapModel).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from data import Matches, load_brawler_meta

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model_legendary.pkl"
PATCH_CONFIG = ROOT / "patch.json"
CLASS_OVERRIDES = ROOT / "meta" / "class_overrides.json"
ROLES_PATH = ROOT / "roles.yaml"
ROLE_NAMES = ["assassin", "thrower", "sniper", "tank", "controller", "support", "wall_breaker", "mid_dps"]
ROLE_LABEL = {"assassin": ("diver", "divers"), "thrower": ("thrower", "throwers"),
              "sniper": ("sniper", "snipers"), "tank": ("tank", "tanks"),
              "controller": ("controller", "controllers"), "support": ("support", "supports"),
              "wall_breaker": ("wall-breaker", "wall-breakers"), "mid_dps": ("mid-range DPS", "mid-range DPS")}


def model_path(bracket: str = "legendary") -> Path:
    return ROOT / "models" / f"model_{bracket}.pkl"


def load_patch_config(path: Path = PATCH_CONFIG) -> dict:
    """Current balance patch: {'patch_ts': unix seconds, 'changed_ids': [...], ...}.
    Change patch_time_utc in patch.json when a new patch lands (one line)."""
    if not path.exists():
        return {"patch_ts": 0, "changed_ids": [], "name": None}
    cfg = json.loads(path.read_text())
    ts = int(datetime.strptime(cfg["patch_time_utc"], "%Y-%m-%dT%H:%M:%SZ")
             .replace(tzinfo=timezone.utc).timestamp())
    meta = load_brawler_meta()
    by_name = {v["name"].upper(): k for k, v in meta.items()}
    changed = [by_name[n.upper()] for n in cfg.get("changed_brawlers", []) if n.upper() in by_name]
    return {"patch_ts": ts, "changed_ids": sorted(changed), "name": cfg.get("name"),
            "patch_time_utc": cfg["patch_time_utc"]}


def load_class_overrides(path: Path = CLASS_OVERRIDES) -> dict:
    """brawler name (upper) -> class, from meta/class_overrides.json (see DECISIONS.md)."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {k.upper(): v for k, v in raw.get("overrides", {}).items()}


def load_roles(path: Path = ROLES_PATH) -> dict:
    """brawler name (upper) -> list of role tags, from roles.yaml (user-editable)."""
    if not path.exists():
        return {}
    import yaml
    raw = yaml.safe_load(path.read_text()) or {}
    out = {}
    for name, v in (raw.get("brawlers") or {}).items():
        roles = v.get("roles", []) if isinstance(v, dict) else list(v or [])
        out[str(name).upper()] = [r for r in roles if r in ROLE_NAMES]
    return out


# The pre-fix model (v1), for paired before/after comparisons.
LEGACY_PARAMS = {
    "use_class": True, "use_sim": False, "use_pr": False, "use_ro": False,
    "rare_shrink": 0.0, "pair_rare_shrink": 0.0, "use_class_pairs": False,
    "use_player": False, "use_profiles": False, "use_main": False, "class_overrides": False,
    "use_role_pairs": False, "lat_k": 0,
}

DEFAULT_PARAMS = {
    # L2 precision per parameter group (larger = stronger shrinkage toward the parent level)
    "lam_g": 20.0,       # global brawler strength
    "lam_mode": 80.0,    # mode deviation
    "lam_map": 200.0,    # map deviation
    "lam_cmode": 40.0,   # class x mode (legacy, use_class)
    "lam_cmap": 100.0,   # class x map (legacy, use_class)
    "lam_syn": 300.0,    # teammate pairs (deviation from class+class synergy)
    "lam_cnt": 300.0,    # cross-team pairs (deviation from class-vs-class counter)
    "lam_rank": 0.1,     # skill control
    "half_life_days": 14.0,
    "use_rank": True,
    "prior_smooth": 30.0,   # pseudo-games when smoothing map pick rates toward mode
    # Rank bracket: games whose lowest tier >= bracket_min_tier train bracket-specific
    # deviations on top of shared (all-rank) terms; other games only inform the shared
    # terms, down-weighted by low_tier_weight, so they act as a weak prior.
    "bracket_min_tier": 16,     # 16 = Legendary I; 0 = plain all-ranks model
    "low_tier_weight": 0.25,
    "lam_dev_scale": 1.0,       # penalty on bracket deviations relative to the shared group
    # Balance patch (time + changed brawlers come from patch.json). See DECISIONS.md.
    "use_patch": True,
    "pre_patch_weight": 0.3,
    "changed_prior_share": 0.1,
    # ---- v2 (recommendation fix, see DECISIONS.md) ----
    "use_class": False,         # legacy class x mode / class x map strength priors
    "class_overrides": True,    # apply meta/class_overrides.json
    "use_sim": True,            # map-similarity prior u[x] . e[map]
    "lam_sim": 100.0,
    "sim_gamma": 3.0,           # kernel sharpness: K = exp(gamma (cos - 1))
    "sim_rank": 5,              # embedding dimensions kept
    "use_pr": True,             # pick-rate prior covariate
    "lam_pr": 1.0,
    "pr_floor": 0.01,           # lpr = log(pick rate + floor): caps how negative it gets
    "use_ro": True,             # range x openness
    "lam_ro": 1.0,
    "rare_shrink": 6.0,         # extra shrinkage for low-pick-rate cells
    "rare_ref": 0.02,
    "pair_rare_shrink": 6.0,
    "pair_rare_ref": 0.005,
    "use_class_pairs": False,   # in-game class-vs-class terms (superseded by roles; experiments only)
    "lam_ccls": 5.0,
    "lam_ccls_mode": 20.0,
    "lam_ccls_map": 100.0,
    "use_role_pairs": True,     # role-vs-role counters / role+role synergy (roles.yaml)
    "roles_file": "roles.yaml", # role tags file, relative to the repo root (for A/B tests)
    "lam_role": 100.0,
    "lam_role_mode": 300.0,
    "lat_k": 4,                 # latent matchup embedding size (0 = off)
    "lam_lat": 100.0,
    "use_player": True,         # per-player skill effect (player tags)
    "lam_player": 40.0,
    "use_profiles": True,       # rankedElo + brawler trophies from data/profiles.db
    "use_main": True,           # leave-one-out share of the player's games on that brawler
    "lam_ctrl": 1.0,
}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class MapModel:
    """Everything the draft engine needs for one map, indexed by brawler index."""
    map: str
    mode: str
    brawler_ids: np.ndarray            # (n,)
    S: np.ndarray                      # (n,) total strength on this map
    S_parts: dict                      # name -> (n,) components of S
    Syn: np.ndarray                    # (n, n) symmetric, zero diagonal
    C: np.ndarray                      # (n, n) antisymmetric: C[i, j] = edge of i over j
    prior: np.ndarray                  # (n,) realistic pick-rate distribution on this map
    games_on_map: int = 0
    brawler_games: np.ndarray = None   # (n,) games featuring each brawler on this map
    id_to_idx: dict = field(default_factory=dict)
    S_sd: np.ndarray = None            # (n,) posterior sd of S (Laplace approximation)
    Syn_sd: np.ndarray = None          # (n, n) posterior sd of pair synergy
    C_sd: np.ndarray = None            # (n, n) posterior sd of pair counters
    roles: np.ndarray = None           # (n, r) 0/1 role tags (roles.yaml)
    role_T: np.ndarray = None          # (n, r) role weights (tags split evenly)
    role_names: list = None
    Syn_role: np.ndarray = None        # (r, r) role+role synergy on this map (symmetric)
    C_role: np.ndarray = None          # (r, r) role-vs-role counter on this map (antisym.)

    def __post_init__(self):
        n = len(self.S)
        if self.S_sd is None:
            self.S_sd = np.zeros(n)
        if self.Syn_sd is None:
            self.Syn_sd = np.zeros((n, n))
        if self.C_sd is None:
            self.C_sd = np.zeros((n, n))

    def team_logit(self, a_idx, b_idx) -> float:
        a = list(a_idx); b = list(b_idx)
        z = self.S[a].sum() - self.S[b].sum()
        for i, j in combinations(a, 2):
            z += self.Syn[i, j]
        for i, j in combinations(b, 2):
            z -= self.Syn[i, j]
        if a and b:
            z += self.C[np.ix_(a, b)].sum()
        return float(z)

    def win_prob(self, a_idx, b_idx) -> float:
        return float(sigmoid(self.team_logit(a_idx, b_idx)))

    def marginal(self, own_idx, opp_idx) -> np.ndarray:
        """Logit gain (for 'own' team) of adding each brawler, given current picks."""
        v = self.S.copy()
        if own_idx:
            v += self.Syn[:, list(own_idx)].sum(axis=1)
        if opp_idx:
            v += self.C[:, list(opp_idx)].sum(axis=1)
        return v

    def contribution(self, x, teammates, enemies) -> dict:
        """Decompose brawler x's logit contribution to its team."""
        tm = [t for t in teammates if t != x]
        return {
            "strength": float(self.S[x]),
            "synergy": float(self.Syn[x, tm].sum()) if tm else 0.0,
            "counters": float(self.C[x, list(enemies)].sum()) if enemies else 0.0,
        }

    def role_breakdown(self, ours, theirs) -> list:
        """Role-level parts of the comp logit, our perspective (exact: sums to the role terms
        folded into Syn / C). [(kind, our_role, their_role, n_ours, n_theirs, logit)] with kind
        'counter' (our role i vs their role j), 'synergy' (within our team), 'their_synergy'."""
        if self.role_T is None or self.C_role is None or not len(self.role_names or []):
            return []
        T, R = self.role_T, self.roles
        nR = T.shape[1]
        sA = T[list(ours)].sum(axis=0) if ours else np.zeros(nR)
        sB = T[list(theirs)].sum(axis=0) if theirs else np.zeros(nR)
        cA = R[list(ours)].sum(axis=0).astype(int) if ours else np.zeros(nR, int)
        cB = R[list(theirs)].sum(axis=0).astype(int) if theirs else np.zeros(nR, int)
        out = []
        for i in range(nR):
            for j in range(nR):
                if i != j and sA[i] and sB[j]:
                    out.append(("counter", i, j, cA[i], cB[j], sA[i] * sB[j] * float(self.C_role[i, j])))
        for team, s, c, kind, sgn in ((ours, sA, cA, "synergy", 1.0), (theirs, sB, cB, "their_synergy", -1.0)):
            if len(team) < 2:
                continue
            full = np.outer(s, s) - sum(np.outer(T[x], T[x]) for x in team)   # sum over pairs x != y
            for i in range(nR):
                for j in range(i, nR):
                    v = full[i, j] if i != j else 0.5 * full[i, i]
                    if v:
                        out.append((kind, i, j, c[i], c[j], sgn * v * float(self.Syn_role[i, j])))
        return out

    def comp_notes(self, ours, theirs, min_pct: float = 2.0, top: int = 4) -> list:
        """Readable role-level matchup notes, e.g. 'Their 3 divers hard-counter your 3 throwers
        (-12.0% for you)'. Percentages are always from our side. Parts worth at least min_pct
        win-% points, biggest first."""
        from model import ROLE_LABEL
        names = self.role_names or []
        lab = lambda i, n: f"{n} {ROLE_LABEL.get(names[i], (names[i], names[i]))[0 if n == 1 else 1]}"
        to_pct = lambda z: 100.0 * (sigmoid(z) - 0.5)      # win-% change from an even game
        rows = [r for r in self.role_breakdown(ours, theirs) if abs(to_pct(r[5])) >= min_pct]
        notes = []
        for kind, i, j, ni, nj, z in sorted(rows, key=lambda t: -abs(t[5]))[:top]:
            v = to_pct(z)
            strong = abs(v) >= 6
            if kind == "counter":
                if v < 0:
                    notes.append(f"Their {lab(j, nj)} {'hard-counter' if strong else 'counter'} your {lab(i, ni)} ({v:+.1f}% for you)")
                else:
                    notes.append(f"Your {lab(i, ni)} {'hard-counter' if strong else 'counter'} their {lab(j, nj)} ({v:+.1f}% for you)")
            else:
                who = "Your" if kind == "synergy" else "Their"
                what = lab(i, ni) if i == j else f"{lab(i, ni)} + {lab(j, nj)}"
                good = (v > 0) == (kind == "synergy")
                notes.append(f"{who} {what} {'work well together' if good else 'overlap badly'} ({v:+.1f}% for you)")
        return notes


class DraftModel:
    def __init__(self, params: dict | None = None):
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.meta = load_brawler_meta()
        self.theta = None
        self.patch = {"patch_ts": 0, "changed_ids": []}

    # ------------------------------------------------------------------ bracket / patch
    def in_bracket(self, m: Matches) -> np.ndarray:
        bt = self.params.get("bracket_min_tier") or 0
        if bt <= 0 or m.tier_min is None:
            return np.ones(len(m), dtype=bool)
        return m.tier_min >= bt

    def post_patch(self, m: Matches) -> np.ndarray:
        return m.ts >= self.patch.get("patch_ts", 0)

    def in_target(self, m: Matches) -> np.ndarray:
        """Games the model is *for*: in bracket and after the current patch."""
        return self.in_bracket(m) & self.post_patch(m)

    def _deviating(self) -> bool:
        return bool(self.params.get("bracket_min_tier")) or bool(self.patch.get("patch_ts"))

    def sample_weights(self, ts: np.ndarray, t_ref: float | None = None) -> np.ndarray:
        hl = self.params["half_life_days"]
        if not hl or hl <= 0:
            return np.ones(len(ts))
        t_ref = ts.max() if t_ref is None else t_ref
        w = 0.5 ** ((t_ref - ts) / (hl * 86400.0))
        return w / w.mean()

    # ------------------------------------------------------------------ vocabulary / tables
    def _classes(self):
        ov = load_class_overrides() if self.params.get("class_overrides", True) else {}
        cls = {}
        for b, v in self.meta.items():
            cls[int(b)] = ov.get(v["name"].upper(), v.get("class", "Unknown"))
        return cls

    def _build_vocab(self, m: Matches):
        ids = set(int(b) for b in self.meta)
        ids |= set(np.unique(m.team_a).tolist()) | set(np.unique(m.team_b).tolist())
        self.ids = np.array(sorted(ids), dtype=np.int64)
        self.bidx = {int(b): i for i, b in enumerate(self.ids)}
        self.modes = sorted(set(m.mode))
        self.oidx = {o: i for i, o in enumerate(self.modes)}
        self.maps = sorted(set(zip(m.mode, m.map)))
        self.midx = {mk: i for i, mk in enumerate(self.maps)}
        self.map_mode = np.array([self.oidx[mk[0]] for mk in self.maps], dtype=np.int64)
        cls = self._classes()
        self.class_names = sorted(set(cls.get(int(b), "Unknown") for b in self.ids))
        cidx = {c: i for i, c in enumerate(self.class_names)}
        self.cls = np.array([cidx[cls.get(int(b), "Unknown")] for b in self.ids], dtype=np.int64)
        roles = load_roles(ROOT / self.params.get("roles_file", "roles.yaml"))
        self.role_names = list(ROLE_NAMES)
        nR = len(ROLE_NAMES)
        self.roles_raw = np.zeros((len(self.ids), nR))      # 0/1 tags
        for i, b in enumerate(self.ids):
            nm = self.meta.get(int(b), {}).get("name", "").upper()
            for r in roles.get(nm, []):
                self.roles_raw[i, ROLE_NAMES.index(r)] = 1.0
        rs = self.roles_raw.sum(axis=1, keepdims=True)
        self.T = np.where(rs > 0, self.roles_raw / np.maximum(rs, 1), 0.0)   # weight split across tags

    def _index_rows(self, m: Matches):
        """Integer index arrays for a set of games under the fitted vocabulary."""
        look = np.vectorize(lambda b: self.bidx.get(int(b), -1), otypes=[np.int64])
        bi = np.concatenate([look(m.team_a), look(m.team_b)], axis=1) if len(m) else np.zeros((0, 6), np.int64)
        oi = np.array([self.oidx.get(o, -1) for o in m.mode], dtype=np.int64)
        mi = np.array([self.midx.get(mk, -1) for mk in zip(m.mode, m.map)], dtype=np.int64)
        return bi, oi, mi

    def _pick_tables(self, bi, oi, mi, tgt):
        """Pick-rate tables from target games: lpr per map/mode, map embedding, range/openness."""
        p = self.params
        nB, nO, nM = len(self.ids), len(self.modes), len(self.maps)
        r = np.flatnonzero(tgt)
        if len(r) == 0:
            r = np.arange(len(bi))
        b6, o6, m6 = bi[r], np.repeat(oi[r], 6), np.repeat(mi[r], 6)
        flat = b6.ravel()
        ok = flat >= 0
        n_mode = np.zeros((nO, nB)); n_map = np.zeros((nM, nB))
        np.add.at(n_mode, (o6[ok], flat[ok]), 1.0)
        okm = ok & (m6 >= 0)
        np.add.at(n_map, (m6[okm], flat[okm]), 1.0)
        N_mode = np.bincount(oi[r][oi[r] >= 0], minlength=nO).astype(float)
        N_map = np.bincount(mi[r][mi[r] >= 0], minlength=nM).astype(float)
        mode_rate = (n_mode + 0.5) / (N_mode[:, None] + 1.0)
        k = p["prior_smooth"]
        map_rate = (n_map + k * mode_rate[self.map_mode]) / (N_map[:, None] + k)
        fl = p["pr_floor"]
        Lmap = np.log(map_rate + fl)
        Lmode = np.log(mode_rate + fl)
        self.lpr_map = Lmap - Lmap.mean(axis=1, keepdims=True)
        self.lpr_mode = Lmode - Lmode.mean(axis=1, keepdims=True)
        self.n_map_cell, self.N_map = n_map, N_map
        self.n_mode_cell, self.N_mode = n_mode, N_mode
        self.n_b = n_map.sum(axis=0) if nM else n_mode.sum(axis=0)
        self.N_tgt = float(len(r))
        # map similarity kernel -> embedding
        Zc = Lmap - Lmap.mean(axis=0, keepdims=True)
        Zn = Zc / np.maximum(np.linalg.norm(Zc, axis=1, keepdims=True), 1e-9)
        G = np.clip(Zn @ Zn.T, -1, 1)
        self.map_cos = G
        if nM >= 2:
            K = np.exp(p["sim_gamma"] * (G - 1.0))
            lam, V = np.linalg.eigh(K)
            order = np.argsort(-lam)[: int(p["sim_rank"])]
            lam, V = np.clip(lam[order], 0, None), V[:, order]
            self.E = V * np.sqrt(lam)
        else:
            self.E = np.ones((nM, 1))
        # range x openness: leading component of the double-centered log pick-rate matrix
        if nM >= 2:
            D = Lmap - Lmap.mean(axis=0, keepdims=True) - Lmap.mean(axis=1, keepdims=True) + Lmap.mean()
            U, s, Vt = np.linalg.svd(D, full_matrices=False)
            o, rg = U[:, 0] * s[0], Vt[0]
            o = o / max(o.std(), 1e-9)
            rg = rg / max(rg.std(), 1e-9)
            long_c = [i for i, c in enumerate(self.class_names) if c in ("Marksman",)]
            short_c = [i for i, c in enumerate(self.class_names) if c in ("Tank", "Assassin")]
            if long_c and short_c:
                if rg[np.isin(self.cls, long_c)].mean() < rg[np.isin(self.cls, short_c)].mean():
                    o, rg = -o, -rg
            self.openness, self.range = o, rg
        else:
            self.openness, self.range = np.zeros(nM), np.zeros(nB)

    # ------------------------------------------------------------------ controls
    def _player_arrays(self, m: Matches, grow: bool):
        """Per-slot control arrays: player index, elo z, brawler-trophies z, covered, main."""
        N = len(m)
        out = {k: np.zeros((N, 6)) for k in ("elo", "exp", "cov", "main")}
        pid = np.full((N, 6), -1, dtype=np.int64)
        if not (m.tags_a and m.tags_b):
            return pid, out
        tags = [list(a) + list(b) for a, b in zip(m.tags_a, m.tags_b)]
        teams = np.concatenate([m.team_a, m.team_b], axis=1)
        if self.params.get("use_player"):
            if grow:
                self.pidx = {}
                for row in tags:
                    for t in row:
                        if t not in self.pidx:
                            self.pidx[t] = len(self.pidx)
            pidx = getattr(self, "pidx", {})
            pid = np.array([[pidx.get(t, -1) for t in row] for row in tags], dtype=np.int64)
        if self.params.get("use_main"):
            if grow:
                cnt, tot = {}, {}
                for row, br in zip(tags, teams):
                    for t, b in zip(row, br):
                        cnt[(t, int(b))] = cnt.get((t, int(b)), 0) + 1
                        tot[t] = tot.get(t, 0) + 1
                self._tag_counts = (cnt, tot)
                bc = np.bincount(np.searchsorted(self.ids, teams.ravel()), minlength=len(self.ids))
                self._base_share = (bc + 1.0) / (bc.sum() + len(self.ids))
            cnt, tot = getattr(self, "_tag_counts", ({}, {}))
            loo = 1 if grow else 0
            a = 5.0
            main = np.zeros((N, 6))
            for i, (row, br) in enumerate(zip(tags, teams)):
                for j, (t, b) in enumerate(zip(row, br)):
                    j_b = self.bidx.get(int(b))
                    if j_b is None:
                        continue
                    base = self._base_share[j_b]
                    n_tb = max(cnt.get((t, int(b)), 0) - loo, 0)
                    n_t = max(tot.get(t, 0) - loo, 0)
                    main[i, j] = np.log((n_tb + a * base) / (n_t + a)) - np.log(base)
            out["main"] = main
        if self.params.get("use_profiles"):
            prof = getattr(self, "_profiles", None)
            if prof is None:
                try:
                    from profiles import load_profiles
                    prof = load_profiles()
                except Exception:
                    prof = {}
                self._profiles = prof
            if grow:
                elos = np.array([v["elo"] for v in prof.values() if v.get("elo")], dtype=float)
                exps = np.array([np.log1p(x[1]) for v in prof.values() for x in v["b"].values()], dtype=float)
                self.ctrl_norm = {
                    "elo": (float(elos.mean()), float(elos.std() or 1)) if len(elos) else (0.0, 1.0),
                    "exp": (float(exps.mean()), float(exps.std() or 1)) if len(exps) else (0.0, 1.0)}
            em, es = self.ctrl_norm["elo"]
            xm, xs = self.ctrl_norm["exp"]
            for i, (row, br) in enumerate(zip(tags, teams)):
                for j, (t, b) in enumerate(zip(row, br)):
                    pr = prof.get(t)
                    if not pr:
                        continue
                    bt = pr["b"].get(int(b))
                    if pr.get("elo") is None or bt is None:
                        continue
                    out["cov"][i, j] = 1.0
                    out["elo"][i, j] = (pr["elo"] - em) / es
                    out["exp"][i, j] = (np.log1p(bt[1]) - xm) / xs
        return pid, out

    # ------------------------------------------------------------------ design
    def _layout(self):
        p = self.params
        nB, nO, nM, nC = len(self.ids), len(self.modes), len(self.maps), len(self.class_names)
        K = self.E.shape[1] if p.get("use_sim") else 0
        self.K = K
        groups = [("g", nB), ("m", nO * nB), ("s", nM * nB), ("u", nB * K),
                  ("cm", nO * nC if p.get("use_class") else 0),
                  ("cs", nM * nC if p.get("use_class") else 0),
                  ("syn", nB * nB), ("cnt", nB * nB)]
        cp = p.get("use_class_pairs")
        groups += [("csyn", nC * nC if cp else 0), ("csyn_m", nO * nC * nC if cp else 0),
                   ("csyn_s", nM * nC * nC if cp else 0), ("ccnt", nC * nC if cp else 0),
                   ("ccnt_m", nO * nC * nC if cp else 0), ("ccnt_s", nM * nC * nC if cp else 0)]
        nR = len(self.role_names) if p.get("use_role_pairs") else 0
        groups += [("rsyn", nR * nR), ("rsyn_m", nO * nR * nR), ("rcnt", nR * nR), ("rcnt_m", nO * nR * nR)]
        off, pos = {}, 0
        for g, n in groups:
            off[g] = (pos, n); pos += n
        self.B0 = pos
        pos *= 2                                  # bracket / patch deviations of the above
        for g, n in [("pr", (1 + nO) if p.get("use_pr") else 0), ("ro", (1 + nO) if p.get("use_ro") else 0),
                     ("rank", 1), ("pl", len(getattr(self, "pidx", {})) if p.get("use_player") else 0),
                     ("elo", 1), ("exp", 1), ("cov", 1), ("main", 1)]:
            off[g] = (pos, n); pos += n
        self.off = off
        self.ncol = pos
        self.lat_k = int(p.get("lat_k") or 0)
        self.nlat = 3 * len(self.ids) * self.lat_k       # u, v (counters), w (synergy)

    def _design(self, m: Matches, rank_zero: bool = False, as_target=None, grow: bool = False):
        """Sparse design matrix. rank_zero=True = draft-only: every skill control zeroed.
        as_target=True forces bracket/patch deviations on for every row (prediction)."""
        p = self.params
        bi, oi, mi = self._index_rows(m)
        N = len(m)
        tgt = self.in_target(m) if as_target is None else np.full(N, bool(as_target))
        dev = self._deviating()
        nB, nC, nM = len(self.ids), len(self.class_names), len(self.maps)
        R, Cc, V, D = [], [], [], []      # rows, cols, vals, dev-eligible
        rows_all = np.arange(N)

        def add(mask, col, val, devok=True):
            if not np.any(mask):
                return
            R.append(rows_all[mask]); Cc.append(np.asarray(col)[mask] if np.ndim(col) else np.full(mask.sum(), col))
            V.append(np.broadcast_to(val, (N,))[mask].astype(float) if np.ndim(val) else np.full(mask.sum(), float(val)))
            D.append(devok)

        o = lambda g: self.off[g][0]
        sgn = np.array([1, 1, 1, -1, -1, -1], dtype=float)
        okmap = mi >= 0
        okmode = oi >= 0
        mi0 = np.where(okmap, mi, 0)
        oi0 = np.where(okmode, oi, 0)
        for j in range(6):
            b = bi[:, j]; s = sgn[j]
            ok = b >= 0
            b0 = np.where(ok, b, 0)
            c0 = self.cls[b0]
            add(ok, o("g") + b0, s)
            add(ok & okmode, o("m") + oi0 * nB + b0, s)
            add(ok & okmap, o("s") + mi0 * nB + b0, s)
            if self.off["u"][1]:
                for k in range(self.K):
                    add(ok & okmap, o("u") + b0 * self.K + k, s * self.E[mi0, k])
            if self.off["cm"][1]:
                add(ok & okmode, o("cm") + oi0 * nC + c0, s)
                add(ok & okmap, o("cs") + mi0 * nC + c0, s)
            if self.off["pr"][1]:
                lpr = np.where(okmap, self.lpr_map[mi0, b0], self.lpr_mode[oi0, b0])
                add(ok & okmode, o("pr"), s * lpr, False)
                add(ok & okmode, o("pr") + 1 + oi0, s * lpr, False)
            if self.off["ro"][1]:
                ro = np.where(okmap, self.range[b0] * self.openness[mi0], 0.0)
                add(ok & okmap, o("ro"), s * ro, False)
                add(ok & okmap, o("ro") + 1 + oi0, s * ro, False)
        cp = bool(self.off["ccnt"][1])
        nCC = nC * nC
        for team, s in (((0, 1, 2), 1.0), ((3, 4, 5), -1.0)):
            for a, c in combinations(team, 2):
                x, y = bi[:, a], bi[:, c]
                ok = (x >= 0) & (y >= 0) & (x != y)
                lo, hi = np.minimum(x, y), np.maximum(x, y)
                lo0, hi0 = np.where(ok, lo, 0), np.where(ok, hi, 0)
                add(ok, o("syn") + lo0 * nB + hi0, s)
                if cp:
                    ca, cb = self.cls[lo0], self.cls[hi0]
                    cl, ch = np.minimum(ca, cb), np.maximum(ca, cb)
                    add(ok, o("csyn") + cl * nC + ch, s)
                    add(ok & okmode, o("csyn_m") + oi0 * nCC + cl * nC + ch, s)
                    add(ok & okmap, o("csyn_s") + mi0 * nCC + cl * nC + ch, s)
        for a in (0, 1, 2):
            for c in (3, 4, 5):
                x, y = bi[:, a], bi[:, c]
                ok = (x >= 0) & (y >= 0) & (x != y)
                lo0, hi0 = np.where(ok, np.minimum(x, y), 0), np.where(ok, np.maximum(x, y), 0)
                v = np.where(x < y, 1.0, -1.0)
                add(ok, o("cnt") + lo0 * nB + hi0, v)
                if cp:
                    ca, cb = self.cls[np.where(ok, x, 0)], self.cls[np.where(ok, y, 0)]
                    okc = ok & (ca != cb)
                    cl, ch = np.minimum(ca, cb), np.maximum(ca, cb)
                    vc = np.where(ca < cb, 1.0, -1.0)
                    add(okc, o("ccnt") + cl * nC + ch, vc)
                    add(okc & okmode, o("ccnt_m") + oi0 * nCC + cl * nC + ch, vc)
                    add(okc & okmap, o("ccnt_s") + mi0 * nCC + cl * nC + ch, vc)
        if self.off["rcnt"][1]:
            nR = len(self.role_names)
            T0 = np.vstack([self.T, np.zeros((1, nR))])
            bix = np.where(bi >= 0, bi, len(self.ids))
            TA, TB = T0[bix[:, :3]], T0[bix[:, 3:]]          # (N, 3, nR)
            sA, sB = TA.sum(axis=1), TB.sum(axis=1)
            iu = np.triu_indices(nR, 1)
            idg = np.triu_indices(nR)
            cnt = np.einsum("ni,nj->nij", sA, sB)
            cnt = (cnt - cnt.transpose(0, 2, 1))[:, iu[0], iu[1]]              # antisymmetric part
            def syn_feat(Tt, st):
                full = np.einsum("ni,nj->nij", st, st) - np.einsum("nki,nkj->nij", Tt, Tt)
                full = 0.5 * (full + full.transpose(0, 2, 1))                 # sum over pairs x != y
                up = full * (2.0 - np.eye(nR))                                 # i<j counts both orders
                return 0.5 * up[:, idg[0], idg[1]]
            syn = syn_feat(TA, sA) - syn_feat(TB, sB)
            for g, F, ij in (("rcnt", cnt, iu), ("rsyn", syn, idg)):
                cols = ij[0] * nR + ij[1]
                for q in range(F.shape[1]):
                    # target games only (current patch, in bracket): role matchups aren't
                    # patch-aware per brawler, so stale games must not feed them
                    nz = (F[:, q] != 0) & tgt
                    add(nz, o(g) + cols[q], F[:, q], False)
                    add(nz & okmode, o(g + "_m") + oi0 * nR * nR + cols[q], F[:, q], False)
        if not rank_zero:
            if p["use_rank"]:
                add(np.ones(N, bool), o("rank"), (m.rank_a - m.rank_b) / 3.0, False)
            pid, ctrl = self._player_arrays(m, grow=grow)
            if self.off["pl"][1]:
                for j in range(6):
                    ok = pid[:, j] >= 0
                    add(ok, o("pl") + np.where(ok, pid[:, j], 0), sgn[j], False)
            for g in ("elo", "exp", "cov", "main"):
                diff = ctrl[g][:, :3].sum(axis=1) - ctrl[g][:, 3:].sum(axis=1)
                if np.any(diff != 0):
                    add(np.ones(N, bool), o(g), diff, False)
        rows = np.concatenate(R) if R else np.zeros(0, np.int64)
        cols = np.concatenate(Cc) if Cc else np.zeros(0, np.int64)
        vals = np.concatenate(V) if V else np.zeros(0)
        if dev:
            dmask = np.concatenate([np.full(len(r), d) for r, d in zip(R, D)]) if R else np.zeros(0, bool)
            dmask &= tgt[rows]
            rows = np.concatenate([rows, rows[dmask]])
            cols = np.concatenate([cols, cols[dmask] + self.B0])
            vals = np.concatenate([vals, vals[dmask]])
        return sp.csr_matrix((vals, (rows, cols)), shape=(N, self.ncol))

    # ------------------------------------------------------------------ penalties
    def _penalties(self, bi, oi, mi, tgt) -> np.ndarray:
        p = self.params
        nB, nO, nM, nC = len(self.ids), len(self.modes), len(self.maps), len(self.class_names)
        lam = np.empty(self.ncol)
        ks, kp = p.get("rare_shrink", 0.0), p.get("pair_rare_shrink", 0.0)

        def mult(n, N, kappa, ref):
            n0 = ref * np.maximum(N, 1.0)
            return 1.0 + kappa * n0 / (n + n0)

        mg = mult(self.n_b, self.N_tgt, ks, p["rare_ref"])
        mm_ = mult(self.n_mode_cell, self.N_mode[:, None], ks, p["rare_ref"])
        ms = mult(self.n_map_cell, self.N_map[:, None], ks, p["rare_ref"])
        # pair counts in target games
        r = np.flatnonzero(tgt)
        syn_n = np.zeros(nB * nB); cnt_n = np.zeros(nB * nB)
        for team in ((0, 1, 2), (3, 4, 5)):
            for a, c in combinations(team, 2):
                x, y = bi[r, a], bi[r, c]
                ok = (x >= 0) & (y >= 0)
                np.add.at(syn_n, np.minimum(x, y)[ok] * nB + np.maximum(x, y)[ok], 1)
        for a in (0, 1, 2):
            for c in (3, 4, 5):
                x, y = bi[r, a], bi[r, c]
                ok = (x >= 0) & (y >= 0)
                np.add.at(cnt_n, np.minimum(x, y)[ok] * nB + np.maximum(x, y)[ok], 1)
        Nt = max(len(r), 1)
        msyn = mult(syn_n, Nt, kp, p["pair_rare_ref"])
        mcnt = mult(cnt_n, Nt, kp, p["pair_rare_ref"])
        base = {
            "g": p["lam_g"] * mg,
            "m": (p["lam_mode"] * mm_).ravel(),
            "s": (p["lam_map"] * ms).ravel(),
            "u": np.repeat(p["lam_sim"] * mg, self.K),
            "cm": np.full(self.off["cm"][1], p["lam_cmode"]),
            "cs": np.full(self.off["cs"][1], p["lam_cmap"]),
            "syn": p["lam_syn"] * msyn, "cnt": p["lam_cnt"] * mcnt,
            "csyn": np.full(self.off["csyn"][1], p["lam_ccls"]),
            "csyn_m": np.full(self.off["csyn_m"][1], p["lam_ccls_mode"]),
            "csyn_s": np.full(self.off["csyn_s"][1], p["lam_ccls_map"]),
            "ccnt": np.full(self.off["ccnt"][1], p["lam_ccls"]),
            "ccnt_m": np.full(self.off["ccnt_m"][1], p["lam_ccls_mode"]),
            "ccnt_s": np.full(self.off["ccnt_s"][1], p["lam_ccls_map"]),
            "rsyn": np.full(self.off["rsyn"][1], p["lam_role"]),
            "rsyn_m": np.full(self.off["rsyn_m"][1], p["lam_role_mode"]),
            "rcnt": np.full(self.off["rcnt"][1], p["lam_role"]),
            "rcnt_m": np.full(self.off["rcnt_m"][1], p["lam_role_mode"]),
        }
        # which columns involve a brawler changed in the patch
        changed = np.isin(self.ids, list(self.patch.get("changed_ids", [])))
        pair_ch = (changed[:, None] | changed[None, :]).ravel()
        ch = {"g": changed, "m": np.tile(changed, nO), "s": np.tile(changed, nM),
              "u": np.repeat(changed, self.K), "syn": pair_ch, "cnt": pair_ch}
        dev_scale = p.get("lam_dev_scale", 1.0)
        share = p.get("changed_prior_share", 0.1)
        use_split = self._deviating() and changed.any()
        for g, l0 in base.items():
            st, n = self.off[g]
            if n == 0:
                continue
            l0 = np.asarray(l0, float)[:n]
            shared, devl = l0.copy(), l0 * dev_scale
            if use_split and g in ch:
                c = ch[g][:n]
                tv = (1.0 + 1.0 / dev_scale) / l0[c]
                shared[c] = 1.0 / (share * tv)
                devl[c] = 1.0 / ((1 - share) * tv)
            lam[st:st + n] = shared
            lam[st + self.B0:st + self.B0 + n] = devl
        for g, v in (("pr", p["lam_pr"]), ("ro", p["lam_ro"]), ("rank", p["lam_rank"]),
                     ("pl", p["lam_player"]), ("elo", p["lam_ctrl"]), ("exp", p["lam_ctrl"]),
                     ("cov", p["lam_ctrl"]), ("main", p["lam_ctrl"])):
            st, n = self.off[g]
            lam[st:st + n] = v
        return lam

    # ------------------------------------------------------------------ latent matchups
    def _incidence(self, m: Matches):
        """Sparse (N, nB) team membership matrices for team A and team B."""
        bi, _, _ = self._index_rows(m)
        N, nB = len(m), len(self.ids)
        mats = []
        for sl in (slice(0, 3), slice(3, 6)):
            b = bi[:, sl]
            ok = b >= 0
            rows = np.repeat(np.arange(N), 3).reshape(N, 3)[ok]
            mats.append(sp.csr_matrix((np.ones(ok.sum()), (rows, b[ok])), shape=(N, nB)))
        return mats

    def _split_lat(self, lat):
        nB, k = len(self.ids), self.lat_k
        return lat[: nB * k].reshape(nB, k), lat[nB * k: 2 * nB * k].reshape(nB, k), lat[2 * nB * k:].reshape(nB, k)

    def _latent(self, PA, PB, lat, grad_r=None):
        """Latent matchup logit: counters sum_{a,b} (u_a.v_b - u_b.v_a) = UA.VB - UB.VA,
        synergy sum_{pairs} w_x.w_y = (|WA|^2 - sum |w_a|^2)/2 per team (A minus B).
        With grad_r (dLoss/dz per game), also returns the gradient w.r.t. lat."""
        u, v, w = self._split_lat(lat)
        UA, UB, VA, VB, WA, WB = PA @ u, PB @ u, PA @ v, PB @ v, PA @ w, PB @ w
        ww = (w ** 2).sum(axis=1)
        z = (UA * VB).sum(1) - (UB * VA).sum(1) + 0.5 * ((WA ** 2).sum(1) - PA @ ww) \
            - 0.5 * ((WB ** 2).sum(1) - PB @ ww)
        if grad_r is None:
            return z
        r = grad_r[:, None]
        gu = PA.T @ (r * VB) - PB.T @ (r * VA)
        gv = PB.T @ (r * UA) - PA.T @ (r * UB)
        gw = PA.T @ (r * WA) - PB.T @ (r * WB) - (np.asarray(PA.T @ grad_r).ravel()
                                                  - np.asarray(PB.T @ grad_r).ravel())[:, None] * w
        return z, np.concatenate([gu.ravel(), gv.ravel(), gw.ravel()])

    # ------------------------------------------------------------------ fit / predict
    def fit(self, m: Matches, verbose: bool = False) -> "DraftModel":
        self.patch = load_patch_config() if self.params.get("use_patch", True) else {"patch_ts": 0, "changed_ids": []}
        self._build_vocab(m)
        bi, oi, mi = self._index_rows(m)
        tgt = self.in_target(m)
        self._pick_tables(bi, oi, mi, tgt)
        if self.params.get("use_player") and m.tags_a:
            self._player_arrays(m, grow=True)      # builds player index before layout
        else:
            self.pidx = {}
        self._layout()
        X = self._design(m, grow=True)
        y = m.y
        w = self.sample_weights(m.ts)
        if self.params.get("bracket_min_tier"):
            w = np.where(self.in_bracket(m), w, w * self.params["low_tier_weight"])
        w = np.where(self.post_patch(m), w, w * self.params["pre_patch_weight"])
        lam = self._penalties(bi, oi, mi, tgt)
        Xt = X.T.tocsr()
        nlin = X.shape[1]
        PA, PB = self._incidence(m) if self.nlat else (None, None)
        if self.nlat:      # latent matchups learn from target games only (see role terms)
            D = sp.diags(tgt.astype(float))
            PA, PB = D @ PA, D @ PB
        lam_lat = self.params.get("lam_lat", 50.0)

        def f(th):
            theta, lat = th[:nlin], th[nlin:]
            z = X @ theta
            if self.nlat:
                z = z + self._latent(PA, PB, lat)
            loss = np.sum(w * (np.logaddexp(0.0, z) - y * z)) + 0.5 * np.sum(lam * theta ** 2)
            r = w * (sigmoid(z) - y)
            g = Xt @ r + lam * theta
            if self.nlat:
                _, gl = self._latent(PA, PB, lat, grad_r=r)
                loss += 0.5 * lam_lat * np.sum(lat ** 2)
                g = np.concatenate([g, gl + lam_lat * lat])
            return loss, g

        th0 = np.zeros(nlin + self.nlat)
        if self.nlat:      # small random init: zero is a saddle point of the bilinear terms
            th0[nlin:] = np.random.default_rng(0).normal(0, 0.05, self.nlat)
        res = minimize(f, th0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 3000, "gtol": 1e-6, "maxcor": 20})
        if verbose:
            print(f"fit: {X.shape[0]} games, {X.shape[1] + self.nlat} params, converged={res.success}, "
                  f"iters={res.nit}")
        self.theta = res.x[:nlin]
        self.lat = res.x[nlin:]
        self.lam = lam
        z_lat = self._latent(PA, PB, self.lat) if self.nlat else 0.0
        self._posterior(X, w, bi, mi, tgt, z_lat)
        inb = tgt
        self._fit_priors(m.subset(np.flatnonzero(inb)) if inb.any() else m, w[inb] if inb.any() else w)
        self.n_train = len(m)
        self.n_in_bracket = int(inb.sum())       # target games: in bracket and post-patch
        self.t_max = int(m.ts.max())
        self._map_cache = {}
        return self

    def _posterior(self, X, w, bi, mi, tgt, z_lat=0.0):
        """Block-diagonal Laplace approximation of the posterior covariance.
        Per brawler: all its strength columns (g, m, s, u and their deviations) jointly;
        per pair: shared + deviation column. Stored as variances of S per (map, brawler),
        S per (mode, brawler) for unknown maps, and per pair."""
        p = sigmoid(X @ self.theta + z_lat)
        W = w * p * (1 - p)
        nB, nO, nM, K = len(self.ids), len(self.modes), len(self.maps), self.K
        dev = self._deviating()
        # aggregate curvature per (brawler, map, target)
        A = np.zeros((nB, nM, 2))
        for j in range(6):
            b = bi[:, j]
            ok = (b >= 0) & (mi >= 0)
            np.add.at(A, (b[ok], mi[ok], tgt[ok].astype(int)), W[ok])
        D1 = 1 + nO + nM + K
        F = np.zeros((nM, D1))
        F[:, 0] = 1.0
        F[np.arange(nM), 1 + self.map_mode] = 1.0
        F[np.arange(nM), 1 + nO + np.arange(nM)] = 1.0
        if K:
            F[:, 1 + nO + nM:] = self.E
        o = lambda g: self.off[g][0]
        self.S_var_map = np.zeros((nM, nB))
        self.S_var_mode = np.zeros((nO, nB))
        Dd = 2 * D1 if dev else D1
        for b in range(nB):
            cols = np.concatenate([[o("g") + b], o("m") + np.arange(nO) * nB + b,
                                   o("s") + np.arange(nM) * nB + b,
                                   o("u") + b * K + np.arange(K)]).astype(int)
            lam_b = self.lam[cols]
            H = np.zeros((Dd, Dd))
            for t in (0, 1):
                Ft = np.hstack([F, F * t]) if dev else F
                H += (Ft * A[b, :, t][:, None]).T @ Ft
            if dev:
                lam_b = np.concatenate([lam_b, self.lam[cols + self.B0]])
            H[np.diag_indices(Dd)] += lam_b
            Hinv = np.linalg.inv(H)
            Cm = np.hstack([F, F]) if dev else F
            self.S_var_map[:, b] = np.einsum("ij,jk,ik->i", Cm, Hinv, Cm)
            Fo = np.zeros((nO, D1)); Fo[:, 0] = 1; Fo[np.arange(nO), 1 + np.arange(nO)] = 1
            Co = np.hstack([Fo, Fo]) if dev else Fo
            self.S_var_mode[:, b] = np.einsum("ij,jk,ik->i", Co, Hinv, Co)
        # pairs: [[a + l1, at], [at, at + l2]] per column
        a = np.asarray(X.power(2).T @ W).ravel()
        for g in ("syn", "cnt"):
            st, n = self.off[g]
            a_all, l1 = a[st:st + n], self.lam[st:st + n]
            if dev:
                a_t, l2 = a[st + self.B0:st + self.B0 + n], self.lam[st + self.B0:st + self.B0 + n]
                h11, h12, h22 = a_all + l1, a_t, a_t + l2
                det = h11 * h22 - h12 ** 2
                var = (h22 + h11 - 2 * h12) / det
            else:
                var = 1.0 / (a_all + l1)
            setattr(self, f"{g}_var", var.reshape(nB, nB))

    def _fit_priors(self, m: Matches, w: np.ndarray):
        """Recency-weighted pick counts per (map, brawler) and (mode, brawler)."""
        self.pick_map, self.pick_mode, self.games_map, self.games_mode = {}, {}, {}, {}
        self.raw_map_counts = {}
        self.n_games_map_raw = {}
        for i in range(len(m)):
            mk = (m.mode[i], m.map[i])
            self.games_map[mk] = self.games_map.get(mk, 0.0) + w[i]
            self.games_mode[m.mode[i]] = self.games_mode.get(m.mode[i], 0.0) + w[i]
            self.n_games_map_raw[mk] = self.n_games_map_raw.get(mk, 0) + 1
            for x in list(m.team_a[i]) + list(m.team_b[i]):
                x = int(x)
                self.pick_map[(mk, x)] = self.pick_map.get((mk, x), 0.0) + w[i]
                self.pick_mode[(m.mode[i], x)] = self.pick_mode.get((m.mode[i], x), 0.0) + w[i]
                self.raw_map_counts[(mk, x)] = self.raw_map_counts.get((mk, x), 0) + 1

    def predict_logit(self, m: Matches, rank_zero: bool = False) -> np.ndarray:
        """Logits as if every game is in the model's bracket. rank_zero=True = draft-only
        (all skill controls zeroed: what the draft engine sees)."""
        X = self._design(m, rank_zero=rank_zero, as_target=True)
        z = X @ self.theta
        if getattr(self, "nlat", 0):
            PA, PB = self._incidence(m)
            z = z + self._latent(PA, PB, self.lat)
        return z

    def predict_proba(self, m: Matches, rank_zero: bool = False) -> np.ndarray:
        return sigmoid(self.predict_logit(m, rank_zero))

    # ------------------------------------------------------------------ per-map export
    def brawler_ids(self) -> np.ndarray:
        return self.ids

    def _th(self, g, dev=True):
        st, n = self.off[g]
        v = self.theta[st:st + n].copy()
        if dev and self._deviating() and g not in ("pr", "ro"):
            v += self.theta[st + self.B0:st + self.B0 + n]
        return v

    def map_model(self, mp: str, mode: str) -> MapModel:
        ck = (mode, mp)
        if ck in self._map_cache:
            return self._map_cache[ck]
        p = self.params
        ids = self.ids
        n = len(ids)
        nO, nM, nC = len(self.modes), len(self.maps), len(self.class_names)
        oi, mi = self.oidx.get(mode, -1), self.midx.get(ck, -1)
        id_to_idx = {int(b): i for i, b in enumerate(ids)}
        z = np.zeros(n)
        parts = {"global": self._th("g")}
        parts["mode"] = self._th("m").reshape(nO, n)[oi] if oi >= 0 else z.copy()
        parts["map"] = self._th("s").reshape(nM, n)[mi] if mi >= 0 else z.copy()
        parts["similar_maps"] = (self._th("u").reshape(n, self.K) @ self.E[mi]) if (self.K and mi >= 0) else z.copy()
        if self.off["pr"][1] and oi >= 0:
            beta = self._th("pr")
            lpr = self.lpr_map[mi] if mi >= 0 else self.lpr_mode[oi]
            parts["pick_rate"] = (beta[0] + beta[1 + oi]) * lpr
        else:
            parts["pick_rate"] = z.copy()
        if self.off["ro"][1] and oi >= 0 and mi >= 0:
            rho = self._th("ro")
            parts["range_fit"] = (rho[0] + rho[1 + oi]) * self.range * self.openness[mi]
        else:
            parts["range_fit"] = z.copy()
        cl = np.zeros(n)
        if self.off["cm"][1]:
            if oi >= 0:
                cl += self._th("cm").reshape(nO, nC)[oi][self.cls]
            if mi >= 0:
                cl += self._th("cs").reshape(nM, nC)[mi][self.cls]
        parts["class"] = cl
        # Syn / C: pair deviations + class-level terms
        U = self._th("syn").reshape(n, n)
        Syn = np.triu(U, 1); Syn = Syn + Syn.T
        Uc = self._th("cnt").reshape(n, n)
        C = np.triu(Uc, 1); C = C - C.T
        Sc = np.zeros((nC, nC)); Cc = np.zeros((nC, nC))
        if self.off["ccnt"][1]:
            Sc += self._th("csyn").reshape(nC, nC)
            Cc += self._th("ccnt").reshape(nC, nC)
            if oi >= 0:
                Sc += self._th("csyn_m").reshape(nO, nC, nC)[oi]
                Cc += self._th("ccnt_m").reshape(nO, nC, nC)[oi]
            if mi >= 0:
                Sc += self._th("csyn_s").reshape(nM, nC, nC)[mi]
                Cc += self._th("ccnt_s").reshape(nM, nC, nC)[mi]
            Sc = np.triu(Sc); Sc = Sc + np.triu(Sc, 1).T
            Cc = np.triu(Cc, 1); Cc = Cc - Cc.T
            Syn = Syn + Sc[self.cls[:, None], self.cls[None, :]]
            np.fill_diagonal(Syn, 0.0)
            C = C + Cc[self.cls[:, None], self.cls[None, :]]
        nR = len(getattr(self, "role_names", []))
        Rs = np.zeros((nR, nR)); Rc = np.zeros((nR, nR))
        if self.off.get("rcnt", (0, 0))[1]:
            Rs += self._th("rsyn").reshape(nR, nR)
            Rc += self._th("rcnt").reshape(nR, nR)
            if oi >= 0:
                Rs += self._th("rsyn_m").reshape(nO, nR, nR)[oi]
                Rc += self._th("rcnt_m").reshape(nO, nR, nR)[oi]
            Rs = np.triu(Rs); Rs = Rs + np.triu(Rs, 1).T
            Rc = np.triu(Rc, 1); Rc = Rc - Rc.T
            Syn = Syn + self.T @ Rs @ self.T.T
            C = C + self.T @ Rc @ self.T.T
        if getattr(self, "nlat", 0):
            u, v, w = self._split_lat(self.lat)
            Syn = Syn + w @ w.T
            C = C + u @ v.T - v @ u.T
        np.fill_diagonal(Syn, 0.0)
        np.fill_diagonal(C, 0.0)
        # realistic pick distribution: map pick share smoothed toward mode pick share
        k = p["prior_smooth"]
        gm = self.games_map.get(ck, 0.0)
        gmode = self.games_mode.get(mode, 0.0)
        prior = np.zeros(n)
        bg = np.zeros(n, dtype=np.int64)
        for i, b in enumerate(ids):
            b = int(b)
            mode_rate = (self.pick_mode.get((mode, b), 0.0) + 0.05) / (gmode + 0.05 * n / 6 + 1e-9)
            map_cnt = self.pick_map.get((ck, b), 0.0)
            prior[i] = (map_cnt + k * mode_rate) / (gm + k)
            bg[i] = self.raw_map_counts.get((ck, b), 0)
        prior = prior / prior.sum()
        # Center S on the typical (pick-weighted) pick: prediction-invariant for 3v3, and makes
        # "map strength" read as "vs a typical pick on this map".
        # Re-express pair terms relative to the typical pick (exact for 3v3, see DECISIONS.md):
        # g(x) = expected counter edge vs a typical opponent, h(x) = expected synergy with a
        # typical teammate. C' = C - g(x) + g(y), Syn' = Syn - h(x) - h(y), S' = S + 3g + 2h.
        # Then S is "strength vs typical opponents with typical teammates", and Syn / C are
        # what a specific teammate / enemy changes on top of that.
        gC = C @ prior
        hS = Syn @ prior
        C = C - gC[:, None] + gC[None, :]
        Syn = Syn - hS[:, None] - hS[None, :]
        np.fill_diagonal(Syn, 0.0)
        parts["avg_matchups"] = 3.0 * gC + 2.0 * hS
        S_raw = sum(parts.values())
        parts["baseline"] = np.full(n, -float(np.dot(prior, S_raw)))
        S = S_raw + parts["baseline"]
        if mi >= 0:
            S_var = self.S_var_map[mi]
        elif oi >= 0:
            S_var = self.S_var_mode[oi]
        else:
            S_var = 1.0 / self.lam[self.off["g"][0]:self.off["g"][0] + n]
        mm = MapModel(mp, mode, ids, S, parts, Syn, C, prior,
                      games_on_map=self.n_games_map_raw.get(ck, 0), brawler_games=bg,
                      id_to_idx=id_to_idx, S_sd=np.sqrt(S_var),
                      Syn_sd=np.sqrt(np.triu(self.syn_var, 1) + np.triu(self.syn_var, 1).T),
                      C_sd=np.sqrt(np.triu(self.cnt_var, 1) + np.triu(self.cnt_var, 1).T),
                      roles=getattr(self, "roles_raw", None), role_T=getattr(self, "T", None),
                      role_names=list(getattr(self, "role_names", [])), Syn_role=Rs, C_role=Rc)
        self._map_cache[ck] = mm
        return mm

    def known_maps(self) -> list[tuple[str, str, int]]:
        """(mode, map, raw game count) sorted by mode then count desc."""
        out = [(mk[0], mk[1], c) for mk, c in self.n_games_map_raw.items()]
        return sorted(out, key=lambda t: (t[0], -t[2]))

    def brawler_class(self, bid: int) -> str:
        i = self.bidx.get(int(bid))
        return self.class_names[self.cls[i]] if i is not None else "Unknown"

    # ------------------------------------------------------------------ persistence
    _TRANSIENT = ("_map_cache", "_profiles", "_tag_counts")

    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v for k, v in self.__dict__.items() if k not in self._TRANSIENT}
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "DraftModel":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        obj._map_cache = {}
        obj.__dict__.setdefault("patch", {"patch_ts": 0, "changed_ids": []})
        return obj

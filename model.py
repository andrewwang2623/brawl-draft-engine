"""Comp-strength model: regularized logistic regression with hierarchical shrinkage.

logit P(team A wins) =
      sum_{a in A} S(a) - sum_{b in B} S(b)                    brawler strength on this map
    + sum_{pairs in A} syn(a,a') - sum_{pairs in B} syn(b,b')   teammate synergy
    + sum_{a in A, b in B} C(a, b)                             counters (C antisymmetric)
    + beta_rank * (rank_A - rank_B) / 3                         skill control (zeroed at draft time)

with S(x) = g[x] + m[mode,x] + s[map,x] + cm[mode,class(x)] + cs[map,class(x)].

Each additive layer has its own L2 penalty, which is the MAP estimate under a Gaussian
hierarchy: the map effect is shrunk toward the mode effect, and that toward the global effect.
Sparse maps therefore fall back to mode/global numbers, and unseen brawler-map
combinations get a class x map prior (cold start). There's no intercept: the model is
antisymmetric under swapping teams, so P(A beats B) = 1 - P(B beats A).

Because the logit is a sum of per-brawler and per-pair terms, any pick's contribution
can be decomposed exactly (see MapModel).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from data import Matches, load_brawler_meta

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model_legendary.pkl"


def model_path(bracket: str = "legendary") -> Path:
    return ROOT / "models" / f"model_{bracket}.pkl"

DEFAULT_PARAMS = {
    # L2 precision per parameter group (larger = stronger shrinkage toward the parent level)
    "lam_g": 20.0,       # global brawler strength
    "lam_mode": 80.0,    # mode deviation
    "lam_map": 200.0,    # map deviation (shrinks toward mode)
    "lam_cmode": 40.0,   # class x mode
    "lam_cmap": 100.0,   # class x map
    "lam_syn": 300.0,    # teammate pairs
    "lam_cnt": 300.0,    # cross-team pairs
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


class DraftModel:
    def __init__(self, params: dict | None = None):
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.meta = load_brawler_meta()
        self.theta = None

    # ------------------------------------------------------------------ features
    def _cls(self, bid: int) -> str:
        return self.meta.get(int(bid), {}).get("class", "Unknown")

    def _game_terms(self, mp, mode, A, B, rank_diff):
        """Yield (key, value) feature entries for one game (team A perspective)."""
        mk = (mode, mp)
        for team, sgn in ((A, 1.0), (B, -1.0)):
            for x in team:
                x = int(x)
                c = self._cls(x)
                yield ("g", x), sgn
                yield ("m", mode, x), sgn
                yield ("s", mk, x), sgn
                yield ("cm", mode, c), sgn
                yield ("cs", mk, c), sgn
            for x, y in combinations(sorted(int(t) for t in team), 2):
                yield ("syn", x, y), sgn
        for a in A:
            for b in B:
                a, b = int(a), int(b)
                if a < b:
                    yield ("cnt", a, b), 1.0
                elif b < a:
                    yield ("cnt", b, a), -1.0
        if self.params["use_rank"]:
            yield ("rank",), rank_diff / 3.0

    def in_bracket(self, m: Matches) -> np.ndarray:
        bt = self.params.get("bracket_min_tier") or 0
        if bt <= 0 or m.tier_min is None:
            return np.ones(len(m), dtype=bool)
        return m.tier_min >= bt

    def _design(self, m: Matches, grow: bool, rank_zero: bool = False, as_bracket=None):
        """Sparse design matrix. as_bracket=True forces bracket deviations on for every
        row (used at prediction time: we predict for bracket games)."""
        bracketed = bool(self.params.get("bracket_min_tier"))
        if as_bracket is None:
            inb = self.in_bracket(m)
        else:
            inb = np.full(len(m), bool(as_bracket))
        rows, cols, vals = [], [], []
        for i in range(len(m)):
            rd = 0.0 if rank_zero else float(m.rank_a[i] - m.rank_b[i])
            for key, v in self._game_terms(m.map[i], m.mode[i], m.team_a[i], m.team_b[i], rd):
                keys = (key, ("d",) + key) if (bracketed and inb[i] and key[0] != "rank") else (key,)
                for k in keys:
                    j = self.index.get(k)
                    if j is None:
                        if not grow:
                            continue
                        j = self.index[k] = len(self.index)
                    rows.append(i); cols.append(j); vals.append(v)
        X = sp.csr_matrix((vals, (rows, cols)), shape=(len(m), len(self.index)))
        return X

    def _penalties(self) -> np.ndarray:
        p = self.params
        lam_by_group = {"g": p["lam_g"], "m": p["lam_mode"], "s": p["lam_map"],
                        "cm": p["lam_cmode"], "cs": p["lam_cmap"], "syn": p["lam_syn"],
                        "cnt": p["lam_cnt"], "rank": p["lam_rank"]}
        lam = np.empty(len(self.index))
        for key, j in self.index.items():
            if key[0] == "d":
                lam[j] = lam_by_group[key[1]] * p.get("lam_dev_scale", 1.0)
            else:
                lam[j] = lam_by_group[key[0]]
        return lam

    def sample_weights(self, ts: np.ndarray, t_ref: float | None = None) -> np.ndarray:
        hl = self.params["half_life_days"]
        if not hl or hl <= 0:
            return np.ones(len(ts))
        t_ref = ts.max() if t_ref is None else t_ref
        w = 0.5 ** ((t_ref - ts) / (hl * 86400.0))
        return w / w.mean()

    # ------------------------------------------------------------------ fit / predict
    def fit(self, m: Matches, verbose: bool = False) -> "DraftModel":
        self.index = {}
        X = self._design(m, grow=True)
        y = m.y
        w = self.sample_weights(m.ts)
        if self.params.get("bracket_min_tier"):
            w = np.where(self.in_bracket(m), w, w * self.params["low_tier_weight"])
        lam = self._penalties()
        Xt = X.T.tocsr()

        def f(theta):
            z = X @ theta
            # log(1 + e^z) - y z, computed stably
            loss = np.sum(w * (np.logaddexp(0.0, z) - y * z)) + 0.5 * np.sum(lam * theta ** 2)
            g = Xt @ (w * (sigmoid(z) - y)) + lam * theta
            return loss, g

        res = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                       options={"maxiter": 2000, "gtol": 1e-6})
        if verbose:
            print(f"fit: {X.shape[0]} games, {X.shape[1]} params, converged={res.success}, "
                  f"iters={res.nit}")
        self.theta = res.x
        # Pick-rate priors describe what opponents in the bracket actually pick.
        inb = self.in_bracket(m)
        self._fit_priors(m.subset(np.flatnonzero(inb)) if inb.any() else m, w[inb] if inb.any() else w)
        self.n_train = len(m)
        self.n_in_bracket = int(inb.sum())
        self.t_max = int(m.ts.max())
        self._map_cache = {}
        return self

    def _fit_priors(self, m: Matches, w: np.ndarray):
        """Recency-weighted pick counts per (map, brawler) and (mode, brawler)."""
        self.pick_map, self.pick_mode, self.games_map, self.games_mode = {}, {}, {}, {}
        self.raw_map_counts = {}
        for i in range(len(m)):
            mk = (m.mode[i], m.map[i])
            self.games_map[mk] = self.games_map.get(mk, 0.0) + w[i]
            self.games_mode[m.mode[i]] = self.games_mode.get(m.mode[i], 0.0) + w[i]
            for x in list(m.team_a[i]) + list(m.team_b[i]):
                x = int(x)
                self.pick_map[(mk, x)] = self.pick_map.get((mk, x), 0.0) + w[i]
                self.pick_mode[(m.mode[i], x)] = self.pick_mode.get((m.mode[i], x), 0.0) + w[i]
                self.raw_map_counts[(mk, x)] = self.raw_map_counts.get((mk, x), 0) + 1
        self.n_games_map_raw = {}
        for i in range(len(m)):
            mk = (m.mode[i], m.map[i])
            self.n_games_map_raw[mk] = self.n_games_map_raw.get(mk, 0) + 1

    def predict_logit(self, m: Matches, rank_zero: bool = False) -> np.ndarray:
        """Logits as if every game is in the model's bracket."""
        X = self._design(m, grow=False, rank_zero=rank_zero, as_bracket=True)
        return X @ self.theta

    def predict_proba(self, m: Matches, rank_zero: bool = False) -> np.ndarray:
        return sigmoid(self.predict_logit(m, rank_zero))

    # ------------------------------------------------------------------ per-map export
    def _w(self, *key) -> float:
        """Weight of a term for bracket games: shared value + bracket deviation."""
        j = self.index.get(key)
        d = self.index.get(("d",) + key)
        return (0.0 if j is None else float(self.theta[j])) + (0.0 if d is None else float(self.theta[d]))

    def brawler_ids(self) -> np.ndarray:
        ids = set(self.meta.keys())
        ids |= {k[1] for k in self.index if k[0] == "g"}
        return np.array(sorted(ids), dtype=np.int64)

    def map_model(self, mp: str, mode: str) -> MapModel:
        ck = (mode, mp)
        if ck in self._map_cache:
            return self._map_cache[ck]
        ids = self.brawler_ids()
        n = len(ids)
        id_to_idx = {int(b): i for i, b in enumerate(ids)}
        parts = {k: np.zeros(n) for k in ("global", "mode", "map", "class")}
        for i, b in enumerate(ids):
            b = int(b)
            c = self._cls(b)
            parts["global"][i] = self._w("g", b)
            parts["mode"][i] = self._w("m", mode, b)
            parts["map"][i] = self._w("s", ck, b)
            parts["class"][i] = self._w("cm", mode, c) + self._w("cs", ck, c)
        S = sum(parts.values())
        Syn = np.zeros((n, n))
        C = np.zeros((n, n))
        for key, j in self.index.items():
            base = key[1:] if key[0] == "d" else key
            if base[0] == "syn":
                a, b = id_to_idx.get(base[1]), id_to_idx.get(base[2])
                if a is not None and b is not None:
                    Syn[a, b] += self.theta[j]
                    Syn[b, a] = Syn[a, b]
            elif base[0] == "cnt":
                a, b = id_to_idx.get(base[1]), id_to_idx.get(base[2])
                if a is not None and b is not None:
                    C[a, b] += self.theta[j]
                    C[b, a] = -C[a, b]
        # realistic pick distribution: map pick share smoothed toward mode pick share
        k = self.params["prior_smooth"]
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
        mm = MapModel(mp, mode, ids, S, parts, Syn, C, prior,
                      games_on_map=self.n_games_map_raw.get(ck, 0), brawler_games=bg,
                      id_to_idx=id_to_idx)
        self._map_cache[ck] = mm
        return mm

    def known_maps(self) -> list[tuple[str, str, int]]:
        """(mode, map, raw game count) sorted by mode then count desc."""
        out = [(mk[0], mk[1], c) for mk, c in self.n_games_map_raw.items()]
        return sorted(out, key=lambda t: (t[0], -t[2]))

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v for k, v in self.__dict__.items() if k != "_map_cache"}
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "DraftModel":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        obj._map_cache = {}
        return obj

"""Draft engine: next-pick and ban recommendations via beam-searched expectimax.

Ranked pick order is 1-2-2-1: with F = first-pick team and S = second,
    slot:  0  1  2  3  4  5
    team:  F  S  S  F  F  S

The pick being recommended is a max node. Everyone else's picks are chance nodes drawn
from a *realistic* distribution:

    P(c) ∝ prior_map(c)^alpha * exp(beta * marginal_gain(c))

i.e., pick-rate weighted and leaning toward strong picks, but not perfect play. That holds
for the opponent AND (by default, `our_policy="realistic"`) for our own later picks: in solo
ranked the user controls one slot, and modelling teammates as perfect argmax players while the
opponent is merely realistic inflated every win% by ~10 points (winner's curse + asymmetry;
see DECISIONS.md). Teammates use their own pick-rate weight (alpha_us) and the conservative
gain (S - z_cons * sd(S)). `our_policy="max"` keeps max nodes for our later picks (a
coordinated team), choosing them by the conservative, prior-regularized value.

Uncertainty: each candidate also gets win_prob_p25, the win probability with the candidate's
own contribution at the 25th percentile of its posterior (Laplace sd of S plus its current
pair terms). Picks are ranked by win_prob_p25, so low-data brawlers can't top the list on a
noisy point estimate. Leaves are
full 3v3 comps scored by the comp model. Because each pick's value accounts for what
the opponent can still answer with, early picks come out "safe" and the final pick
(nothing left to answer it) comes out a pure counter.

Explanations follow the expected final comp: for a candidate x we track the
expectation (under the search policy) of
    strength(x) | synergy with current teammates | synergy with future teammates |
    counters vs current enemies | counters vs future enemies (= risk of being countered)
Those components sum exactly to x's logit contribution to the final comp.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from model import MapModel, sigmoid

PICK_ORDER = (0, 1, 1, 0, 0, 1)   # 0 = first-pick team, 1 = second-pick team
COMP_NAMES = ("strength", "synergy_now", "synergy_future", "counters_now", "countered_future")


@dataclass
class SearchConfig:
    root_width: int = 12      # candidates fully searched at the root
    max_width: int = 5        # beam at interior max nodes (our future picks)
    chance_width: int = 5     # opponent candidates expanded at interior chance nodes
    alpha: float = 1.0        # weight on pick-rate prior in opponent model
    beta: float = 4.0         # how greedily the opponent chases strong picks (per logit)
    ban_candidates: int = 12
    ban_root_width: int = 6
    ban_max_width: int = 4
    ban_chance_width: int = 4
    z_cons: float = 0.674     # 25th percentile of a normal posterior
    our_policy: str = "realistic"   # "realistic" teammates (chance nodes) or "max"
    alpha_us: float = 1.0     # teammates' pick-rate weight (realistic policy)
    beta_us: float = 4.0      # teammates' greed per logit (realistic policy)
    prior_us: float = 0.05    # "max" policy: logit bonus per nat of log pick rate (selection only)
    conf_high: float = 1.5    # confidence: sd of the pick's contribution in win-% points
    conf_med: float = 3.0


def pct(z: float) -> float:
    """Logit contribution -> approximate win-probability points (slope at 50% is 1/4)."""
    return 25.0 * z


@dataclass
class DraftState:
    our: list = field(default_factory=list)       # brawler indices, in pick order
    opp: list = field(default_factory=list)
    bans: list = field(default_factory=list)
    we_first: bool = True
    # Order of the REMAINING picks (True = us). None = standard 1-2-2-1 continuing from
    # `step`. Set it when picks were entered out of order (e.g. the UI's free slot entry):
    # then it's the teams of the still-empty slots, in draft order.
    seq: tuple | None = None

    @property
    def step(self) -> int:
        return len(self.our) + len(self.opp)

    def remaining(self) -> tuple:
        if self.seq is not None:
            return tuple(bool(t) for t in self.seq)
        return tuple((PICK_ORDER[s] == 0) == self.we_first for s in range(self.step, 6))

    def team_to_move(self) -> int | None:
        """0 = us, 1 = opponent, None = draft over."""
        r = self.remaining()
        if not r:
            return None
        return 0 if r[0] else 1

    def validate(self):
        if self.seq is None:
            exp_ours = sum(1 for s in range(self.step) if (PICK_ORDER[s] == 0) == self.we_first)
            if len(self.our) != exp_ours:
                raise ValueError(f"after {self.step} picks we should have {exp_ours} picks "
                                 f"(1-2-2-1 order), got {len(self.our)}")
        else:
            if len(self.our) > 3 or len(self.opp) > 3:
                raise ValueError("a team can't have more than 3 picks")
            if len(self.seq) != 6 - self.step or sum(map(bool, self.seq)) != 3 - len(self.our):
                raise ValueError("remaining pick order doesn't match the picks made")
        used = self.our + self.opp + self.bans
        if len(set(self.our + self.opp)) != len(self.our + self.opp):
            raise ValueError("a brawler was picked twice")
        if set(self.our + self.opp) & set(self.bans):
            raise ValueError("a banned brawler was picked")
        return used


class DraftEngine:
    def __init__(self, mm: MapModel, cfg: SearchConfig | None = None):
        self.mm = mm
        self.cfg = cfg or SearchConfig()
        self.n = len(mm.S)
        self.log_prior = np.log(np.maximum(mm.prior, 1e-9))
        sd = mm.S_sd if getattr(mm, "S_sd", None) is not None else np.zeros(self.n)
        self.S_sd = sd
        # selection offset for OUR future picks: conservative strength (+ pick-rate prior in max mode)
        self.sel_off = -self.cfg.z_cons * sd
        if self.cfg.our_policy == "max":
            self.sel_off = self.sel_off + self.cfg.prior_us * (self.log_prior - float(np.dot(mm.prior, self.log_prior)))

    # ---------------------------------------------------------------- helpers
    def _avail(self, used) -> np.ndarray:
        mask = np.ones(self.n, dtype=bool)
        if used:
            mask[list(used)] = False
        return mask

    def opponent_dist(self, opp, our, avail) -> np.ndarray:
        """Realistic probability of each brawler being the opponent's next pick."""
        gain = self.mm.marginal(opp, our)
        s = self.cfg.alpha * self.log_prior + self.cfg.beta * gain
        s = np.where(avail, s, -np.inf)
        s = s - s[avail].max()
        p = np.exp(s)
        return p / p.sum()

    def teammate_dist(self, our, opp, avail) -> np.ndarray:
        """Realistic distribution of our own later picks (solo-queue teammates): pick-rate
        weighted, leaning toward picks that are good with conservative (p25-ish) strength."""
        gain = self.mm.marginal(our, opp) + self.sel_off
        s = self.cfg.alpha_us * self.log_prior + self.cfg.beta_us * gain
        s = np.where(avail, s, -np.inf)
        s = s - s[avail].max()
        p = np.exp(s)
        return p / p.sum()

    def _track(self, x, our_root, opp_root, our, opp) -> np.ndarray:
        """x's contribution decomposed against the (final) comp."""
        mm = self.mm
        now_tm = [t for t in our_root if t != x]
        fut_tm = [t for t in our if t not in our_root and t != x]
        fut_en = [t for t in opp if t not in opp_root]
        return np.array([
            mm.S[x],
            mm.Syn[x, now_tm].sum() if now_tm else 0.0,
            mm.Syn[x, fut_tm].sum() if fut_tm else 0.0,
            mm.C[x, list(opp_root)].sum() if opp_root else 0.0,
            mm.C[x, fut_en].sum() if fut_en else 0.0,
        ])

    # ---------------------------------------------------------------- expectimax
    def _value(self, our, opp, z, used, we_first, widths, ctx, zs=0.0):
        """Expected outcome from this state, plus the tracked component vector.

        our/opp: tuples of indices; z: current logit (our perspective) of partial comps;
        zs: selection offset accumulated over our future max-node picks. Returns
        (vals, vec) with vals = [P(win), P(win | root pick at its p25), selection value].
        """
        step = len(our) + len(opp)
        x, our_root, opp_root, bonus, delta = ctx
        if step >= 6:
            vec = self._track(x, our_root, opp_root, our, opp) if x is not None else None
            zz = z + bonus
            return np.array([sigmoid(zz), sigmoid(zz - delta), sigmoid(zz + zs)]), vec
        key = (tuple(sorted(our)), tuple(sorted(opp)))
        memo = widths["memo"]
        if key in memo:
            return memo[key]
        ours_turn = widths["seq"][step - widths["root"]]
        avail = self._avail(used)
        mm = self.mm
        last = step == 5
        if ours_turn and self.cfg.our_policy == "max":
            gain = mm.marginal(our, opp)
            score = np.where(avail, gain + self.sel_off, -np.inf)
            if last:
                c = int(np.argmax(score))
                res = self._value(our + (c,), opp, z + gain[c], used | {c}, we_first, widths, ctx,
                                  zs + self.sel_off[c])
            else:
                best = None
                for c in np.argsort(-score)[: widths["max"]]:
                    c = int(c)
                    r = self._value(our + (c,), opp, z + gain[c], used | {c}, we_first, widths, ctx,
                                    zs + self.sel_off[c])
                    if best is None or r[0][2] > best[0][2]:
                        best = r
                res = best
        else:
            if ours_turn:
                p = self.teammate_dist(our, opp, avail)
                dz = mm.marginal(our, opp)          # our logit change per pick
            else:
                p = self.opponent_dist(opp, our, avail)
                dz = -mm.marginal(opp, our)
            if last:
                idx = np.flatnonzero(avail)
                pv = p[idx]
                zz = z + dz[idx] + bonus
                vals = np.array([np.dot(pv, sigmoid(zz)), np.dot(pv, sigmoid(zz - delta)),
                                 np.dot(pv, sigmoid(zz + zs))])
                vec = None
                if x is not None:
                    base = self._track(x, our_root, opp_root, our, opp)
                    if ours_turn:
                        base[2] += float(np.dot(pv, mm.Syn[x, idx]))
                    else:
                        base[4] += float(np.dot(pv, mm.C[x, idx]))
                    vec = base
                res = (vals, vec)
            else:
                cands = np.argsort(-p)[: widths["chance"]]
                w = p[cands] / p[cands].sum()
                vals = np.zeros(3)
                vec = np.zeros(len(COMP_NAMES)) if x is not None else None
                for c, wc in zip(cands, w):
                    c = int(c)
                    nxt = (our + (c,), opp) if ours_turn else (our, opp + (c,))
                    r = self._value(nxt[0], nxt[1], z + dz[c], used | {c}, we_first, widths, ctx, zs)
                    vals += wc * r[0]
                    if vec is not None:
                        vec += wc * r[1]
                res = (vals, vec)
        memo[key] = res
        return res

    def pick_sd(self, c, our, opp) -> float:
        """Posterior sd (logit) of candidate c's contribution given the current picks."""
        mm = self.mm
        v = float(self.S_sd[c]) ** 2
        tm = [t for t in our if t != c]
        if tm:
            v += float(np.sum(mm.Syn_sd[c, tm] ** 2))
        if opp:
            v += float(np.sum(mm.C_sd[c, list(opp)] ** 2))
        return float(np.sqrt(v))

    def _root_children(self, st: DraftState, cands, widths_cfg, bonus_vec=None):
        """Search each candidate pick for us at the root. Returns {c: (value, vec)}."""
        mm = self.mm
        used = set(st.validate())
        our, opp = tuple(st.our), tuple(st.opp)
        z0 = mm.team_logit(st.our, st.opp)
        gain = mm.marginal(our, opp)
        out = {}
        for c in cands:
            c = int(c)
            bonus = float(bonus_vec[c]) if bonus_vec is not None else 0.0
            widths = dict(widths_cfg, memo={}, seq=st.remaining(), root=st.step)
            delta = self.cfg.z_cons * self.pick_sd(c, our, opp)
            ctx = (c, our, opp, bonus, delta)
            out[c] = self._value(our + (c,), opp, z0 + gain[c], used | {c}, st.we_first,
                                 widths, ctx)
        return out

    # ---------------------------------------------------------------- public API
    def recommend_picks(self, st: DraftState, top: int = 5, pool_mask=None,
                        bonus_vec=None, bonus_info=None) -> dict:
        """Top picks for us. pool_mask restricts *our next* pick (e.g. owned brawlers);
        bonus_vec adds a per-brawler logit bonus for that pick (personal mode)."""
        used = st.validate()
        if st.team_to_move() != 0:
            raise ValueError("not our turn to pick")
        mm, cfg = self.mm, self.cfg
        avail = self._avail(used)
        gain = mm.marginal(st.our, st.opp)
        # candidate pre-selection on the conservative gain
        order_score = gain - cfg.z_cons * self.S_sd + (bonus_vec if bonus_vec is not None else 0.0)
        cand_mask = avail & (pool_mask if pool_mask is not None else True)
        order_score = np.where(cand_mask, order_score, -np.inf)
        k = min(cfg.root_width, int(cand_mask.sum()))
        cands = list(np.argsort(-order_score)[:k])

        # Baseline: a realistic (pick-rate weighted) pick in this slot, no bonus.
        p_real = self.opponent_dist(st.our, st.opp, avail)   # same model, our perspective
        real_c = list(np.argsort(-p_real)[: cfg.chance_width])
        widths = {"max": cfg.max_width, "chance": cfg.chance_width}
        children = self._root_children(st, set(cands) | set(real_c), widths, bonus_vec)
        w = p_real[real_c] / p_real[real_c].sum()
        baseline = float(sum(wc * children[int(c)][0][0] for c, wc in zip(real_c, w)))

        ranked = sorted(cands, key=lambda c: -children[int(c)][0][1])[:top]
        recs = []
        for c in ranked:
            c = int(c)
            vals, vec = children[c]
            v, v25 = float(vals[0]), float(vals[1])
            sd_pct = pct(self.pick_sd(c, st.our, st.opp))
            comps = {name: round(float(pct(val)), 2) for name, val in zip(COMP_NAMES, vec)}
            rec = {
                "brawler_id": int(mm.brawler_ids[c]),
                "win_prob": round(100 * v, 1),
                "delta_vs_typical": round(100 * (v - baseline), 2),
                "breakdown_pct": {
                    "map_strength": comps["strength"],
                    "synergy": round(comps["synergy_now"] + comps["synergy_future"], 2),
                    "counters": comps["counters_now"],
                    "risk_countered": comps["countered_future"],
                },
                "strength_parts_pct": {k2: round(pct(mm.S_parts[k2][c]), 2) for k2 in mm.S_parts},
                "games_on_map": int(mm.brawler_games[c]) if mm.brawler_games is not None else 0,
                "pick_rate": round(100 * float(mm.prior[c]) * 6, 1),
                "win_prob_p25": round(100 * v25, 1),
                "confidence": ("high" if sd_pct <= cfg.conf_high else
                               "medium" if sd_pct <= cfg.conf_med else "low"),
                "uncertainty_pct": round(sd_pct, 2),
                "comp_notes": mm.comp_notes(list(st.our) + [c], list(st.opp)) if hasattr(mm, "comp_notes") else [],
            }
            if bonus_vec is not None:
                rec["breakdown_pct"]["you"] = round(pct(float(bonus_vec[c])), 2)
                if bonus_info:
                    rec["you_info"] = bonus_info.get(int(mm.brawler_ids[c]))
            rec["explanation"] = explain_pick(rec, st, self)
            recs.append(rec)
        return {"baseline_win_prob": round(100 * baseline, 1), "picks": recs}

    def predict_opponent(self, st: DraftState, top: int = 5) -> list:
        used = st.validate()
        avail = self._avail(used)
        p = self.opponent_dist(st.opp, st.our, avail)
        idx = np.argsort(-p)[:top]
        return [{"brawler_id": int(self.mm.brawler_ids[i]), "prob": round(100 * float(p[i]), 1)}
                for i in idx]

    def draft_value(self, st: DraftState, widths_cfg) -> float:
        """Expected P(we win) of the remaining draft (no candidate tracking)."""
        used = set(st.validate())
        z0 = self.mm.team_logit(st.our, st.opp)
        widths = dict(widths_cfg, memo={}, seq=st.remaining(), root=st.step)
        return float(self._value(tuple(st.our), tuple(st.opp), z0, used, st.we_first, widths,
                                 (None, (), (), 0.0, 0.0))[0][0])

    def recommend_bans(self, st: DraftState, top: int = 5) -> dict:
        """Bans that most improve our expected draft outcome (i.e. most reduce the
        opponent's best expected comp), given which side picks first. Bans are
        simultaneous and hidden in ranked, so the opponent's bans are unknown;
        already-entered bans are respected."""
        cfg = self.cfg
        used = st.validate()
        avail = self._avail(used)
        mm = self.mm
        widths = {"max": cfg.ban_max_width, "chance": cfg.ban_chance_width}
        base = self.draft_value(st, widths)
        # Candidates: brawlers the opponent would realistically want (prior x strength),
        # plus the strongest on the map regardless of pick rate.
        threat = self.cfg.alpha * self.log_prior + self.cfg.beta * mm.S
        threat = np.where(avail, threat, -np.inf)
        strong = np.where(avail, mm.S - cfg.z_cons * self.S_sd, -np.inf)
        cands = list(dict.fromkeys(
            [int(c) for c in np.argsort(-threat)[: cfg.ban_candidates]]
            + [int(c) for c in np.argsort(-strong)[:4]]))
        p_open = self.opponent_dist(st.opp, st.our, avail)
        out = []
        for c in cands:
            st2 = DraftState(list(st.our), list(st.opp), list(st.bans) + [c], st.we_first, st.seq)
            v = self.draft_value(st2, widths)
            out.append((v - base, c))
        out.sort(reverse=True)
        recs = []
        for dv, c in out[:top]:
            # Which of the opponent's likely picks it would have countered / beaten
            rec = {
                "brawler_id": int(mm.brawler_ids[c]),
                "delta_win_prob": round(100 * dv, 2),
                "map_strength_pct": round(pct(float(mm.S[c])), 2),
                "pick_rate": round(100 * float(mm.prior[c]) * 6, 1),
                "opp_first_choice_prob": round(100 * float(p_open[c]), 1),
                "games_on_map": int(mm.brawler_games[c]) if mm.brawler_games is not None else 0,
            }
            rec["explanation"] = explain_ban(rec, st)
            recs.append(rec)
        return {"baseline_win_prob": round(100 * base, 1), "bans": recs}


# -------------------------------------------------------------------- text
def _sgn(v: float) -> str:
    return f"{v:+.1f}%"


def explain_pick(rec: dict, st: DraftState, eng: DraftEngine) -> str:
    b = rec["breakdown_pct"]
    step = st.step
    parts = [f"map strength {_sgn(b['map_strength'])}"]
    if abs(b["synergy"]) >= 0.05:
        parts.append(f"synergy {_sgn(b['synergy'])}")
    if st.opp:
        parts.append(f"vs current enemies {_sgn(b['counters'])}")
    remaining_enemy = sum(1 for t in st.remaining()[1:] if not t)
    if remaining_enemy:
        parts.append(f"vs likely future enemy picks {_sgn(b['risk_countered'])} "
                     f"({remaining_enemy} enemy pick{'s' if remaining_enemy > 1 else ''} left)")
    else:
        parts.append("no enemy picks left, so it can't be countered")
    if "you" in b:
        info = rec.get("you_info") or {}
        form = info.get("me_pct", 0.0) + info.get("comfort_pct", 0.0)
        if info.get("games"):
            parts.append(f"you {_sgn(form)} ({info['games']} games on it, {info['wr']:.0f}% WR)")
        if info.get("buffie_pct"):
            parts.append(f"{_sgn(info['buffie_pct'])}: you're missing buffies "
                         f"({', '.join(info.get('missing_buffies', []))})")
        if info.get("power_pct"):
            parts.append(f"{_sgn(info['power_pct'])}: power {info.get('power')}")
    role = ("Safe blind pick" if step == 0 else
            "Last pick: pure counter" if remaining_enemy == 0 else
            "Mid-draft pick")
    notes = rec.get("comp_notes") or []
    extra = (" " + notes[0] + ".") if notes else ""
    conf = ""
    if rec.get("confidence") == "low" or rec["games_on_map"] < 30:
        conf = (f" Low confidence ({rec['games_on_map']} games on this map): leaning on similar maps "
                "and pick-rate priors.")
    return f"{role}. " + ", ".join(parts) + "." + extra + conf


def explain_ban(rec: dict, st: DraftState) -> str:
    side = "you pick first" if st.we_first else "they pick first"
    msg = (f"Banning raises your expected win chance by {rec['delta_win_prob']:+.2f}% ({side}). "
           f"Map strength {_sgn(rec['map_strength_pct'])}, picked in {rec['pick_rate']:.0f}% of games here; "
           f"{rec['opp_first_choice_prob']:.0f}% chance it's the opponent's top choice if left open.")
    return msg

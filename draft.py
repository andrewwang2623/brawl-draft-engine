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

Ban intent: the enemy's known bans hint at their plan (banning three divers -> they're likely
on throwers). Their opponent model gets an extra factor exp(gamma * intent(c)), with
    intent(c) = min(cap, sum_b w(b) * max(C[b, c], 0) / sigma_C)
    w(b) = 1 / (1 + odds * n * P_ban(b)),  P_ban(b) ∝ prior(b)^ban_alpha * exp(ban_beta * S[b])
w(b) is the posterior that ban b is a "plan" ban rather than a "meta" ban (a 50/50 mixture of
P_ban and a uniform plan ban at odds = 1), so meta bans count little. sigma_C is the
pick-weighted RMS of C on the map, so intent is in "typical counters" and gamma means the
same on every map. gamma = 0 (or no known enemy bans) is exactly the old model.

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
# gamma per "read enemy bans" level (per unit of normalized intent; see DECISIONS.md). "low" moves
# P(next enemy pick is a thrower) on Flaring Phoenix from 13% to 17% after 3 diver bans; "medium"
# to 22%. Three meta bans move the whole distribution by < 1% (total variation).
BAN_INTENT_PRESETS = {"off": 0.0, "low": 0.10, "medium": 0.20}
BAN_READ_MIN_TV = 0.04      # show the ban read if the enemy's next-pick distribution moves >= this (TV)
BAN_READ_MIN_PCT = 1.0      # ... or a top pick rec's win% moves >= this many points


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
    # ban intent (enemy's known bans -> their likely plan); gamma 0 = off
    gamma: float = BAN_INTENT_PRESETS["low"]
    ban_alpha: float = 1.0    # ban-worthiness P_ban ∝ prior^ban_alpha * exp(ban_beta * S)
    ban_beta: float = 4.0     # (same shape as the opponent's "threat" score used for ban recs)
    ban_meta_odds: float = 1.0   # prior odds that a ban is a meta ban rather than a plan ban
    intent_cap: float = 8.0   # cap on a brawler's normalized intent (bounds exp(gamma * intent))


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
    # The enemy's known bans (subset of `bans`; drives ban intent). Bans of unknown team go
    # only in `bans`.
    opp_bans: list = field(default_factory=list)

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
        if set(self.opp_bans) - set(self.bans):
            raise ValueError("opp_bans must be a subset of bans")
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
        # ban intent: how ban-worthy each brawler is (null "meta" ban model) -> surprise weight
        s = self.cfg.ban_alpha * self.log_prior + self.cfg.ban_beta * mm.S
        pb = np.exp(s - s.max())
        self.p_ban = pb / pb.sum()
        self.ban_surprise = 1.0 / (1.0 + self.cfg.ban_meta_odds * self.n * self.p_ban)
        # typical |counter| on this map (pick-weighted RMS of C): intent is in these units
        c_scale = float(np.sqrt(max(mm.prior @ (mm.C ** 2) @ mm.prior, 0.0)))
        self.c_scale = c_scale if c_scale > 1e-9 else 1.0

    def with_gamma(self, gamma: float) -> "DraftEngine":
        """A shallow copy that reads enemy bans with a different gamma (per request; the cached
        engine is shared between threads, so it's never mutated)."""
        import copy
        from dataclasses import replace
        if gamma == self.cfg.gamma:
            return self
        e = copy.copy(self)
        e.cfg = replace(self.cfg, gamma=float(gamma))
        return e

    # ---------------------------------------------------------------- ban intent
    def ban_intent(self, opp_bans) -> np.ndarray | None:
        """Normalized intent per brawler from the enemy's bans (None if no bans):
        sum_b w(b) * max(C[b, c], 0) / sigma_C, capped. High = what the bans protect."""
        B = list(dict.fromkeys(int(b) for b in opp_bans or []))
        if not B:
            return None
        w = self.ban_surprise[B]
        intent = (w[:, None] * np.maximum(self.mm.C[B], 0.0)).sum(axis=0) / self.c_scale
        return np.minimum(intent, self.cfg.intent_cap)

    def opp_bias(self, st: "DraftState") -> np.ndarray | None:
        """gamma * intent for the enemy's pick model, or None when off (exact old behaviour)."""
        if not self.cfg.gamma or not st.opp_bans:
            return None
        intent = self.ban_intent(st.opp_bans)
        return None if intent is None else self.cfg.gamma * intent

    # ---------------------------------------------------------------- helpers
    def _avail(self, used) -> np.ndarray:
        mask = np.ones(self.n, dtype=bool)
        if used:
            mask[list(used)] = False
        return mask

    def opponent_dist(self, opp, our, avail, bias=None) -> np.ndarray:
        """Realistic probability of each brawler being the opponent's next pick.
        bias: optional per-brawler log-weight (gamma * ban intent)."""
        gain = self.mm.marginal(opp, our)
        s = self.cfg.alpha * self.log_prior + self.cfg.beta * gain
        if bias is not None:
            s = s + bias
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
                p = self.opponent_dist(opp, our, avail, widths.get("bias"))
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

    def _root_children(self, st: DraftState, cands, widths_cfg, bonus_vec=None, intent=True):
        """Search each candidate pick for us at the root. Returns {c: (value, vec)}.
        intent=False ignores the enemy's ban intent (for the with/without comparison)."""
        mm = self.mm
        used = set(st.validate())
        our, opp = tuple(st.our), tuple(st.opp)
        z0 = mm.team_logit(st.our, st.opp)
        gain = mm.marginal(our, opp)
        bias = self.opp_bias(st) if intent else None
        out = {}
        for c in cands:
            c = int(c)
            bonus = float(bonus_vec[c]) if bonus_vec is not None else 0.0
            widths = dict(widths_cfg, memo={}, seq=st.remaining(), root=st.step, bias=bias)
            delta = self.cfg.z_cons * self.pick_sd(c, our, opp)
            ctx = (c, our, opp, bonus, delta)
            out[c] = self._value(our + (c,), opp, z0 + gain[c], used | {c}, st.we_first,
                                 widths, ctx)
        return out

    # ---------------------------------------------------------------- public API
    def recommend_picks(self, st: DraftState, top: int = 5, pool_mask=None,
                        bonus_vec=None, bonus_info=None, compare_intent: int = 0) -> dict:
        """Top picks for us. pool_mask restricts *our next* pick (e.g. owned brawlers);
        bonus_vec adds a per-brawler logit bonus for that pick (personal mode).
        compare_intent=k: when the enemy's ban intent is active, also search the top k recs
        without it and report the difference as `ban_intent_pct` (win-% points)."""
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
        no_intent = {}
        if compare_intent and self.opp_bias(st) is not None:
            no_intent = self._root_children(st, [int(c) for c in ranked[:compare_intent]], widths,
                                            bonus_vec, intent=False)
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
            if c in no_intent:
                rec["ban_intent_pct"] = round(100 * (v - float(no_intent[c][0][0])), 2)
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
        bias = self.opp_bias(st)
        p = self.opponent_dist(st.opp, st.our, avail, bias)
        p0 = self.opponent_dist(st.opp, st.our, avail) if bias is not None else p
        idx = np.argsort(-p)[:top]
        out = []
        for i in idx:
            r = {"brawler_id": int(self.mm.brawler_ids[i]), "prob": round(100 * float(p[i]), 1)}
            if bias is not None:
                r["prob_without_bans"] = round(100 * float(p0[i]), 1)
            out.append(r)
        return out

    def draft_value(self, st: DraftState, widths_cfg, intent: bool = True) -> float:
        """Expected P(we win) of the remaining draft (no candidate tracking)."""
        used = set(st.validate())
        z0 = self.mm.team_logit(st.our, st.opp)
        widths = dict(widths_cfg, memo={}, seq=st.remaining(), root=st.step,
                      bias=self.opp_bias(st) if intent else None)
        return float(self._value(tuple(st.our), tuple(st.opp), z0, used, st.we_first, widths,
                                 (None, (), (), 0.0, 0.0))[0][0])

    def recommend_bans(self, st: DraftState, top: int = 5) -> dict:
        """Bans that most improve our expected draft outcome (i.e. most reduce the
        opponent's best expected comp), given which side picks first. Bans are
        simultaneous and hidden in ranked, so the opponent's bans are usually unknown;
        already-entered bans are respected, and the enemy's known bans (st.opp_bans) shift
        their pick model through ban intent."""
        cfg = self.cfg
        used = st.validate()
        avail = self._avail(used)
        mm = self.mm
        widths = {"max": cfg.ban_max_width, "chance": cfg.ban_chance_width}
        base = self.draft_value(st, widths)
        bias = self.opp_bias(st)
        # Candidates: brawlers the opponent would realistically want (prior x strength [x ban
        # intent]), plus the strongest on the map regardless of pick rate.
        threat = self.cfg.alpha * self.log_prior + self.cfg.beta * mm.S
        if bias is not None:
            threat = threat + bias
        threat = np.where(avail, threat, -np.inf)
        strong = np.where(avail, mm.S - cfg.z_cons * self.S_sd, -np.inf)
        cands = [c for c in dict.fromkeys(
            [int(c) for c in np.argsort(-threat)[: cfg.ban_candidates]]
            + [int(c) for c in np.argsort(-strong)[:4]]) if avail[c]]   # (tiny pools: skip -inf)
        p_open = self.opponent_dist(st.opp, st.our, avail, bias)
        out = []
        for c in cands:
            st2 = DraftState(list(st.our), list(st.opp), list(st.bans) + [c], st.we_first, st.seq,
                             opp_bans=list(st.opp_bans))
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

    def read_bans(self, st: DraftState, pick_recs: list | None = None, name_of=None) -> dict | None:
        """What the enemy's bans say about their plan, in plain words, and how much it matters.

        None when intent is off, no enemy bans are known, or they have no picks left. Otherwise
        a dict with `text`, `meaningful` (the enemy's next-pick distribution moves by >=
        BAN_READ_MIN_TV in total variation, or a top pick rec's win% by >= BAN_READ_MIN_PCT
        points) and the numbers behind it. pick_recs: recs from recommend_picks(...,
        compare_intent=k), whose `ban_intent_pct` are exact with/without differences."""
        from model import ROLE_LABEL
        bias = self.opp_bias(st)
        if bias is None or len(st.opp) >= 3:
            return None
        name_of = name_of or (lambda bid: str(bid))
        mm = self.mm
        avail = self._avail(st.validate())
        p1 = self.opponent_dist(st.opp, st.our, avail, bias)
        p0 = self.opponent_dist(st.opp, st.our, avail)
        dp = p1 - p0
        tv = 0.5 * float(np.abs(dp).sum())
        B = list(dict.fromkeys(st.opp_bans))
        bid = lambda i: int(mm.brawler_ids[i])
        bans = [{"brawler_id": bid(b), "name": name_of(bid(b)),
                 "surprise": round(float(self.ban_surprise[b]), 2)} for b in B]
        rn = list(mm.role_names or [])
        R = mm.roles if mm.roles is not None and rn else None
        plural = lambda r: ROLE_LABEL.get(rn[r], (rn[r], rn[r]))[1]
        # likely plan: role mix of their next pick, with vs without reading the bans
        roles = []
        if R is not None:
            r0, r1 = p0 @ R, p1 @ R
            for r in np.argsort(-(r1 - r0)):
                if r1[r] - r0[r] >= 0.02 and len(roles) < 2:
                    roles.append({"role": rn[r], "label": plural(r),
                                  "prob_before": round(100 * float(r0[r]), 1),
                                  "prob": round(100 * float(r1[r]), 1)})
        more = [i for i in np.argsort(-dp) if dp[i] >= 0.01][:3]     # >= 1 point more likely
        more_likely = [{"brawler_id": bid(i), "name": name_of(bid(i)),
                        "prob_before": round(100 * float(p0[i]), 1), "prob": round(100 * float(p1[i]), 1)}
                       for i in more]
        # who profits: our edge vs the shifted enemy picks, per remaining enemy pick (approx.)
        k_rem = sum(1 for t in st.remaining() if not t)
        dgain = pct(1.0) * k_rem * (mm.C @ dp)             # win-% points, per brawler (for us)
        role_gain = None
        if R is not None:
            best = None
            for r in range(len(rn)):
                m = avail & (R[:, r] > 0)
                if m.sum() >= 2 and mm.prior[m].sum() > 0:
                    g = float(np.dot(mm.prior[m], dgain[m]) / mm.prior[m].sum())
                    if best is None or g > best[1]:
                        best = (r, g)
            if best and best[1] >= 0.3:
                role_gain = {"role": rn[best[0]], "label": plural(best[0]), "gain_pct": round(best[1], 1)}
        rec_changes = [{"brawler_id": r["brawler_id"], "name": r.get("name") or name_of(r["brawler_id"]),
                        "delta_win_prob": r["ban_intent_pct"]}
                       for r in (pick_recs or []) if "ban_intent_pct" in r]
        max_rec = max((abs(r["delta_win_prob"]) for r in rec_changes), default=0.0)
        meaningful = tv >= BAN_READ_MIN_TV or max_rec >= BAN_READ_MIN_PCT

        # text, e.g. "They banned Mortis/Edgar/Kenji → likely throwers (13% → 18% of their next
        # pick); divers gain ≈+0.9%; Mico now +1.2%"
        txt = "They banned " + "/".join(b["name"] for b in bans)
        meta = [b["name"] for b in bans if b["surprise"] < 0.25]
        if meta and len(meta) < len(bans):
            txt += f" ({'/'.join(meta)}: meta ban{'s' if len(meta) > 1 else ''}, counted little)"
        if roles:
            r = roles[0]
            txt += (f" → likely {r['label']} ({r['prob_before']:.0f}% → {r['prob']:.0f}% of their next pick)")
        elif more_likely:
            txt += " → more likely: " + ", ".join(f"{m['name']} ({m['prob_before']:.0f}% → {m['prob']:.0f}%)"
                                                  for m in more_likely[:2])
        else:
            txt += " → no clear plan"
        gains = []
        if role_gain:
            gains.append(f"{role_gain['label']} gain ≈{_sgn(role_gain['gain_pct'])}")
        best_rec = max(rec_changes, key=lambda r: abs(r["delta_win_prob"]), default=None)
        if best_rec and abs(best_rec["delta_win_prob"]) >= 0.5:
            gains.append(f"{best_rec['name']} now {_sgn(best_rec['delta_win_prob'])}")
        if gains:
            txt += "; " + "; ".join(gains)
        return {"text": txt + ".", "meaningful": bool(meaningful), "gamma": self.cfg.gamma,
                "tv_next_pick": round(tv, 3), "bans": bans, "likely_roles": roles,
                "more_likely": more_likely, "role_gain": role_gain, "rec_changes": rec_changes}


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

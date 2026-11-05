"""Recommendation-quality checks.

- Synthetic: conservative (p25) ranking keeps a noisy low-data brawler off the top, the API
  fields exist, and role-level counters are learned from data.
- Trained model (models/model_legendary.pkl + data/matches.db; skipped cleanly if either is
  missing or the model predates the v2 format): global audit over every map, the user's
  Bridge Too Far domain check, and the throwers-vs-divers draft.
"""
from pathlib import Path

import numpy as np
import pytest

from data import Matches
from draft import DraftEngine, DraftState
from model import MapModel, DraftModel, sigmoid

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "model_legendary.pkl"
DB = ROOT / "data" / "matches.db"
N = 10
MIN_GAMES_TOP5 = 30          # no top-5 pick under this many games on the map
MIN_OVERLAP = 4              # every map: top 10 shares >= this many with the 15 most picked
MIN_OVERLAP_PICKED_WINNING = 6
MIN_MEAN_OVERLAP = 6.0       # mean over maps


def make_mm(S, S_sd, prior=None, games=None):
    S = np.asarray(S, float)
    prior = np.full(N, 1.0 / N) if prior is None else np.asarray(prior, float) / np.sum(prior)
    parts = {"global": S.copy(), "mode": np.zeros(N), "map": np.zeros(N), "class": np.zeros(N)}
    return MapModel("T", "gemGrab", np.arange(N) + 16000000, S, parts, np.zeros((N, N)),
                    np.zeros((N, N)), prior, games_on_map=1000,
                    brawler_games=np.full(N, 100) if games is None else np.asarray(games),
                    id_to_idx={16000000 + i: i for i in range(N)}, S_sd=np.asarray(S_sd, float))


def test_conservative_ranking_demotes_noisy_pick():
    S = np.zeros(N); S[0] = 0.30; S[1] = 0.24        # 0: slightly higher mean, very uncertain
    sd = np.full(N, 0.02); sd[0] = 0.25
    games = np.full(N, 500); games[0] = 5
    r = DraftEngine(make_mm(S, sd, games=games)).recommend_picks(DraftState(we_first=True), top=N)
    top = r["picks"][0]
    assert top["brawler_id"] - 16000000 == 1
    noisy = next(p for p in r["picks"] if p["brawler_id"] == 16000000)
    assert noisy["win_prob"] > top["win_prob"] - 0.5       # mean is about as high...
    assert noisy["win_prob_p25"] < top["win_prob_p25"]     # ...but the p25 is lower
    assert noisy["confidence"] == "low" and top["confidence"] == "high"
    for p in r["picks"]:
        assert 0 <= p["win_prob_p25"] <= p["win_prob"] <= 100
        assert p["confidence"] in ("low", "medium", "high")


def test_recommended_pick_is_only_a_few_points_above_typical():
    """Teammates are realistic chance nodes, so a typical first pick is ~50% and the best pick
    is a few points above it (not 60%+ from assuming perfect teammates)."""
    rng = np.random.default_rng(3)
    S = rng.normal(0, 0.1, N)
    r = DraftEngine(make_mm(S, np.full(N, 0.02))).recommend_picks(DraftState(we_first=True))
    assert abs(r["baseline_win_prob"] - 50) < 4
    assert r["picks"][0]["win_prob"] - r["baseline_win_prob"] < 8


def test_role_counter_is_learned():
    """Synthetic: every assassin beats every thrower (+0.35 logit per pair). The role-level
    term should pick that up, so even a sparse assassin/thrower pair gets the right sign."""
    name_to_id = {v["name"]: int(b) for b, v in DraftModel().meta.items()}
    assassins = [name_to_id[n] for n in ("Edgar", "Mortis", "Mico", "Leon", "Kenji")]
    throwers = [name_to_id[n] for n in ("Tick", "Dynamike", "Larry & Lawrie", "Barley", "Sprout")]
    others = [name_to_id[n] for n in ("Colt", "Piper", "Poco", "Gene", "Rosa", "Spike")]
    pool = assassins + throwers + others
    rng = np.random.default_rng(11)
    A, B, y = [], [], []
    for _ in range(8000):
        picks = rng.choice(pool, 6, replace=False)
        a, b = picks[:3], picks[3:]
        z = 0.35 * (sum(x in assassins and t in throwers for x in a for t in b)
                    - sum(x in assassins and t in throwers for x in b for t in a))
        A.append(a); B.append(b); y.append(float(rng.random() < sigmoid(z)))
    n = len(y)
    m = Matches(ts=np.arange(n) * 60 + 1_800_000_000, map=["M"] * n, mode=["gemGrab"] * n,
                team_a=np.array(A), team_b=np.array(B), y=np.array(y),
                rank_a=np.full(n, 18.0), rank_b=np.full(n, 18.0))
    md = DraftModel({"half_life_days": 0, "bracket_min_tier": 0, "use_patch": False}).fit(m)
    mm = md.map_model("M", "gemGrab")
    ia = [mm.id_to_idx[x] for x in assassins]
    it = [mm.id_to_idx[x] for x in throwers]
    # C is re-expressed vs the typical opponent: C'(a,t) = C - g(a) + g(t), and with 5/16 of
    # the pool being throwers/assassins that leaves ~0.35 - 2 * 0.35 * 5/16 = 0.13 per pair.
    assert mm.C[np.ix_(ia, it)].mean() > 0.08
    assert (mm.C[np.ix_(ia, it)] > 0).mean() > 0.9
    notes = mm.comp_notes(ia[:3], it[:3])
    assert any("divers" in s and "throwers" in s for s in notes)


# ---------------------------------------------------------------- trained model
@pytest.fixture(scope="module")
def trained():
    if not MODEL.exists() or not DB.exists():
        pytest.skip("no trained model / match database")
    md = DraftModel.load(MODEL)
    if not hasattr(md, "off"):
        pytest.skip("trained model predates the v2 format; run train.py")
    return md


@pytest.fixture(scope="module")
def audit_result(trained):
    import audit
    return audit.audit_model(trained)


def test_audit_all_maps(audit_result):
    bad_low = [(x["map"], r["name"], r["games"]) for x in audit_result["maps"]
               for r in x["top"][:5] if r["games"] < MIN_GAMES_TOP5]
    assert not bad_low, f"top-5 picks with < {MIN_GAMES_TOP5} games: {bad_low}"
    # Per map, the top 10 must overlap with what's actually picked OR with what's picked and
    # winning there (the user's target). A map whose most-picked brawlers are losing post-patch
    # (e.g. Gem Fort) can legitimately recommend winners outside the 15 most picked.
    low_overlap = [(x["map"], x["overlap_top15"], x.get("overlap_picked_winning"))
                   for x in audit_result["maps"]
                   if x["overlap_top15"] < MIN_OVERLAP
                   and x.get("overlap_picked_winning", 0) < MIN_OVERLAP_PICKED_WINNING]
    assert not low_overlap, (f"maps with top-10 overlap < {MIN_OVERLAP} (most picked) and "
                             f"< {MIN_OVERLAP_PICKED_WINNING} (picked and winning): {low_overlap}")
    assert audit_result["mean_overlap_top15"] >= MIN_MEAN_OVERLAP
    assert audit_result["n_flagged"] <= 1


def test_matchup_checks_mostly_pass(audit_result):
    mu = audit_result["matchups"]
    assert mu["n"] > 0 and mu["pass_rate"] >= 0.8, mu


def test_bridge_too_far_domain_check(trained):
    mm = trained.map_model("Bridge Too Far", "heist")
    r = DraftEngine(mm).recommend_picks(DraftState(we_first=True), top=5)
    name = {int(b): v["name"] for b, v in trained.meta.items()}
    top5 = [name[p["brawler_id"]] for p in r["picks"]]
    assert not {"Ollie", "Bull", "Clancy", "Bolt"} & set(top5), top5
    # Mostly ranged picks: learned range above the median brawler, or tagged sniper / thrower /
    # mid-range DPS in roles.yaml; and no tanks. (Strictly "sniper" would fail on the post-patch
    # data: on this map Brock/Piper/Pierce/Mandy win 42-48% over thousands of games while
    # Melodie/Nori win ~56%, and the engine follows the data. See reference/AUDIT.md.)
    from model import load_roles
    roles = load_roles()
    med = np.median(trained.range)
    ranged = [n for n in top5 if trained.range[trained.bidx[trained_id(trained, n)]] > med
              or {"sniper", "thrower", "mid_dps"} & set(roles.get(n.upper(), []))]
    assert len(ranged) >= 3, (top5, ranged)
    assert not [n for n in top5 if "tank" in roles.get(n.upper(), [])], top5


def trained_id(md, name):
    return next(int(b) for b, v in md.meta.items() if v["name"] == name)


def test_throwers_vs_divers(trained):
    us = [trained_id(trained, n) for n in ("Tick", "Larry & Lawrie", "Dynamike")]
    them = [trained_id(trained, n) for n in ("Edgar", "Mortis", "Mico")]
    # the most played map of four modes
    best = {}
    for mode, mp, c in trained.known_maps():
        if mode not in best or c > best[mode][1]:
            best[mode] = (mp, c)
    for mode in ("gemGrab", "brawlBall", "knockout", "heist"):
        if mode not in best:
            continue
        mm = trained.map_model(best[mode][0], mode)
        a = [mm.id_to_idx[x] for x in us]
        b = [mm.id_to_idx[x] for x in them]
        assert mm.win_prob(a, b) < 0.40, (best[mode][0], mm.win_prob(a, b))
        assert mm.win_prob(b, a) > 0.60
        notes = mm.comp_notes(a, b)
        assert notes and "divers" in notes[0] and "throwers" in notes[0], notes

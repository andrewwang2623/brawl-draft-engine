import numpy as np
import pytest

from draft import DraftEngine, DraftState, SearchConfig
from model import MapModel

N = 10


def make_mm(S=None, C=None, Syn=None, prior=None):
    S = np.zeros(N) if S is None else np.asarray(S, float)
    C = np.zeros((N, N)) if C is None else C
    Syn = np.zeros((N, N)) if Syn is None else Syn
    prior = np.full(N, 1.0 / N) if prior is None else np.asarray(prior, float) / np.sum(prior)
    parts = {"global": S.copy(), "mode": np.zeros(N), "map": np.zeros(N), "class": np.zeros(N)}
    return MapModel("T", "gemGrab", np.arange(N) + 16000000, S, parts, Syn, C, prior,
                    games_on_map=1000, brawler_games=np.full(N, 100),
                    id_to_idx={16000000 + i: i for i in range(N)})


def set_counter(C, a, b, v):
    C[a, b] = v
    C[b, a] = -v


def idx(rec):
    return rec["brawler_id"] - 16000000


def test_last_pick_is_pure_counter():
    C = np.zeros((N, N))
    set_counter(C, 7, 0, 1.0)            # 7 hard-counters enemy brawler 0
    S = np.zeros(N); S[5] = 0.3          # 5 is generically stronger than 7
    eng = DraftEngine(make_mm(S=S, C=C))
    # we pick second; 5 picks made, last pick (slot 5) is ours
    st = DraftState(our=[1, 2], opp=[0, 3, 4], we_first=False)
    r = eng.recommend_picks(st)
    assert idx(r["picks"][0]) == 7
    assert r["picks"][0]["breakdown_pct"]["risk_countered"] == 0
    assert "pure counter" in r["picks"][0]["explanation"]


def test_first_pick_is_safe():
    """Brawler 0 is strongest but hard-countered by 1, which the opponent is likely to
    take; 2 is a bit weaker but uncounterable. Blind first pick should be 2, while 0 is
    still the best last pick when nobody can answer it."""
    S = np.zeros(N); S[0] = 0.6; S[2] = 0.45
    C = np.zeros((N, N)); set_counter(C, 1, 0, 1.5)
    prior = np.ones(N); prior[1] = 5.0
    eng = DraftEngine(make_mm(S=S, C=C, prior=prior))
    r = eng.recommend_picks(DraftState(we_first=True), top=N)
    assert idx(r["picks"][0]) == 2
    zero = next(p for p in r["picks"] if idx(p) == 0)
    assert zero["breakdown_pct"]["risk_countered"] < -5        # visibly flagged as counterable
    # 1 is gone (banned) -> 0 becomes safe and is now the top pick
    r2 = eng.recommend_picks(DraftState(bans=[1], we_first=True))
    assert idx(r2["picks"][0]) == 0


def test_synergy_drives_mid_pick():
    Syn = np.zeros((N, N)); Syn[3, 8] = Syn[8, 3] = 0.8
    eng = DraftEngine(make_mm(Syn=Syn))
    st = DraftState(our=[3], opp=[0, 1], we_first=True)    # slot 3: our double pick
    r = eng.recommend_picks(st)
    assert idx(r["picks"][0]) == 8
    assert r["picks"][0]["breakdown_pct"]["synergy"] > 15


def test_ban_removes_overpowered_brawler_and_respects_side():
    S = np.zeros(N); S[4] = 1.0
    eng = DraftEngine(make_mm(S=S))
    they_first = eng.recommend_bans(DraftState(we_first=False))
    we_first = eng.recommend_bans(DraftState(we_first=True))
    assert idx(they_first["bans"][0]) == 4
    assert they_first["bans"][0]["delta_win_prob"] > 0
    # If we pick first we can just take it ourselves, so banning it is worth much less
    d_we = next((b["delta_win_prob"] for b in we_first["bans"] if idx(b) == 4), 0.0)
    assert d_we < they_first["bans"][0]["delta_win_prob"]


def test_pool_mask_and_bonus():
    S = np.zeros(N); S[0] = 0.5
    eng = DraftEngine(make_mm(S=S))
    mask = np.ones(N, bool); mask[0] = False
    r = eng.recommend_picks(DraftState(we_first=True), pool_mask=mask)
    assert all(idx(p) != 0 for p in r["picks"])
    bonus = np.zeros(N); bonus[6] = 2.0
    r = eng.recommend_picks(DraftState(we_first=True), pool_mask=mask, bonus_vec=bonus)
    assert idx(r["picks"][0]) == 6
    assert r["picks"][0]["breakdown_pct"]["you"] == pytest.approx(50.0)


def test_opponent_distribution():
    S = np.zeros(N); S[2] = 1.0
    eng = DraftEngine(make_mm(S=S))
    avail = np.ones(N, bool); avail[[0, 1]] = False
    p = eng.opponent_dist([], [], avail)
    assert p.sum() == pytest.approx(1.0)
    assert p[0] == 0 and p[1] == 0
    assert p.argmax() == 2


def test_turn_order_validation():
    eng = DraftEngine(make_mm())
    with pytest.raises(ValueError):
        eng.recommend_picks(DraftState(our=[1], opp=[], we_first=True))   # opponent's turn
    with pytest.raises(ValueError):
        DraftState(our=[1, 2], opp=[], we_first=True).validate()          # impossible state
    with pytest.raises(ValueError):
        DraftState(our=[1], opp=[], bans=[1], we_first=True).validate()   # picked a ban
    st = DraftState(our=[1], opp=[], we_first=True)
    assert st.team_to_move() == 1
    assert [DraftState(our=[0] * a, opp=[0] * b, we_first=True).team_to_move()
            for a, b in [(0, 0), (1, 1), (1, 2), (2, 2), (3, 2)]] == [0, 1, 0, 0, 1]


def test_values_are_probabilities_and_fast():
    import time
    rng = np.random.default_rng(0)
    C = rng.normal(0, 0.2, (N, N)); C = C - C.T
    eng = DraftEngine(make_mm(S=rng.normal(0, 0.3, N), C=C))
    t = time.time()
    r = eng.recommend_picks(DraftState(we_first=True))
    assert time.time() - t < 2.0
    for p in r["picks"]:
        assert 0 <= p["win_prob"] <= 100
    vals = [p["win_prob"] for p in r["picks"]]
    assert vals == sorted(vals, reverse=True)

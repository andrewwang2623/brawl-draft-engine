"""Ban intent (the enemy's bans shift their pick model) and the draft log.

Synthetic checks always run; checks on the trained model skip cleanly without
models/model_legendary.pkl + data/matches.db (same as test_recommendations.py).
"""
from pathlib import Path

import numpy as np
import pytest

import drafts
from draft import BAN_INTENT_PRESETS, DraftEngine, DraftState, SearchConfig
from model import ROLE_NAMES, DraftModel, MapModel

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "model_legendary.pkl"
DB = ROOT / "data" / "matches.db"
N = 12
DIVERS, THROWERS = [0, 1, 2], [6, 7]


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def synth_mm(S=None, prior=None):
    """Divers 0-2 beat throwers 6-7 (+0.3 logit); a little noise elsewhere."""
    rng = np.random.default_rng(5)
    C = rng.normal(0, 0.03, (N, N))
    for d in DIVERS:
        for t in THROWERS:
            C[d, t] += 0.3
    C = C - C.T
    S = np.zeros(N) if S is None else np.asarray(S, float)
    prior = np.full(N, 1.0 / N) if prior is None else np.asarray(prior, float) / np.sum(prior)
    parts = {"global": S.copy(), "mode": np.zeros(N), "map": np.zeros(N)}
    return MapModel("T", "gemGrab", np.arange(N) + 16000000, S, parts, np.zeros((N, N)), C, prior,
                    games_on_map=1000, brawler_games=np.full(N, 100),
                    id_to_idx={16000000 + i: i for i in range(N)})


# ---------------------------------------------------------------- synthetic
def test_synthetic_diver_bans_point_to_throwers():
    eng = DraftEngine(synth_mm())
    avail = np.ones(N, bool); avail[DIVERS] = False
    p0 = eng.opponent_dist([], [], avail)
    p1 = eng.opponent_dist([], [], avail, eng.opp_bias(DraftState(bans=DIVERS, opp_bans=DIVERS)))
    assert p1[THROWERS].sum() > 1.3 * p0[THROWERS].sum()
    assert p1.sum() == pytest.approx(1.0) and p1[DIVERS].sum() == 0


def test_synthetic_meta_bans_count_little():
    """The same counter structure, but the divers are the meta (high pick rate + strong):
    banning them is expected anyway, so it says little about the plan."""
    S = np.zeros(N); S[DIVERS] = 0.4
    prior = np.ones(N); prior[DIVERS] = 12.0
    eng = DraftEngine(synth_mm(S, prior))
    assert eng.ban_surprise[DIVERS].max() < 0.25 and eng.ban_surprise[THROWERS].min() > 0.8
    avail = np.ones(N, bool); avail[DIVERS] = False
    st = DraftState(bans=DIVERS, opp_bans=DIVERS)
    shift = lambda e: tv(e.opponent_dist([], [], avail), e.opponent_dist([], [], avail, e.opp_bias(st)))
    rare = np.ones(N); rare[DIVERS] = 0.25
    off_meta = DraftEngine(synth_mm(prior=rare))    # same counters, but the divers are off-meta
    assert off_meta.ban_surprise[DIVERS].min() > 0.7
    assert shift(eng) < 0.4 * shift(off_meta)


def test_gamma_zero_is_exactly_the_old_model_synthetic():
    mm = synth_mm(S=np.linspace(-0.2, 0.2, N))
    old = DraftEngine(mm)                                   # no enemy bans known = old behaviour
    off = DraftEngine(mm, SearchConfig(gamma=0.0))
    for st_old, st_new in [
        (DraftState(bans=[0, 1, 2], we_first=True), DraftState(bans=[0, 1, 2], we_first=True, opp_bans=[0, 1, 2])),
        (DraftState(our=[5], opp=[4, 3], bans=[0, 1], we_first=True),
         DraftState(our=[5], opp=[4, 3], bans=[0, 1], we_first=True, opp_bans=[0, 1])),
    ]:
        assert old.recommend_picks(st_old, compare_intent=3) == off.recommend_picks(st_new, compare_intent=3)
        assert old.recommend_bans(st_old) == off.recommend_bans(st_new)
        assert off.read_bans(st_new) is None
    st_o, st_n = DraftState(our=[5], bans=[0], we_first=True), DraftState(our=[5], bans=[0], we_first=True, opp_bans=[0])
    assert old.predict_opponent(st_o) == off.predict_opponent(st_n)
    # with_gamma(0) on a default engine is the same thing
    assert old.predict_opponent(st_o) == old.with_gamma(0.0).predict_opponent(st_n)


def test_opp_bans_must_be_banned():
    with pytest.raises(ValueError):
        DraftState(bans=[1], opp_bans=[2]).validate()


# ---------------------------------------------------------------- trained model
@pytest.fixture(scope="module")
def trained():
    if not MODEL.exists() or not DB.exists():
        pytest.skip("no trained model / match database")
    md = DraftModel.load(MODEL)
    if not hasattr(md, "off"):
        pytest.skip("trained model predates the v2 format; run train.py")
    return md


def thrower_map(md):
    """The map where throwers take the largest share of picks (from the pick-rate prior)."""
    ti = ROLE_NAMES.index("thrower")
    best = max(md.known_maps(), key=lambda t: float(md.map_model(t[1], t[0]).prior[
        md.map_model(t[1], t[0]).roles[:, ti] > 0].sum()))
    return md.map_model(best[1], best[0])


def test_diver_bans_raise_thrower_picks(trained):
    mm = thrower_map(trained)
    ti, ai = ROLE_NAMES.index("thrower"), ROLE_NAMES.index("assassin")
    thr = mm.roles[:, ti] > 0
    # Off-meta divers (like the user's Mortis/Edgar/Kenji example): skip the map's 10 most-picked
    # brawlers, whose bans are expected anyway and are deliberately discounted by the surprise weight.
    meta = set(int(i) for i in np.argsort(-mm.prior)[:10])
    # Pure divers only (assassin is their only tag), so the
    # bans clearly say "we fear divers" rather than hybrids like Mina (assassin + mid_dps).
    divers = [int(i) for i in np.argsort(-mm.prior)
              if mm.roles[i, ai] > 0 and mm.roles[i].sum() == 1 and not thr[i] and int(i) not in meta][:3]
    st = DraftState(bans=divers, we_first=False, opp_bans=divers)
    avail = np.ones(len(mm.S), bool); avail[divers] = False
    probs = {}
    for lvl, g in BAN_INTENT_PRESETS.items():
        eng = DraftEngine(mm).with_gamma(g)
        probs[lvl] = float(eng.opponent_dist([], [], avail, eng.opp_bias(st))[thr].sum())
    assert probs["low"] > probs["off"] + 0.02, (mm.map, probs)
    assert probs["low"] > 1.15 * probs["off"], (mm.map, probs)
    assert probs["medium"] > probs["low"] + 0.02, (mm.map, probs)
    # the likely-enemy-picks panel and the plain-words read follow
    eng = DraftEngine(mm).with_gamma(BAN_INTENT_PRESETS["medium"])
    pred = eng.predict_opponent(st, top=len(mm.S))
    thr_ids = {int(mm.brawler_ids[i]) for i in np.flatnonzero(thr)}
    assert sum(r["prob"] for r in pred if r["brawler_id"] in thr_ids) > sum(
        r["prob_without_bans"] for r in pred if r["brawler_id"] in thr_ids) + 2
    read = eng.read_bans(st)
    assert read["meaningful"] and "throwers" in read["text"], read


def test_meta_bans_barely_change_anything(trained):
    """Banning the three most ban-worthy brawlers (strong + often picked) is what everyone does:
    on every map it moves the enemy's pick distribution by < 2% (TV) at 'low', < 4% at 'medium'."""
    worst = {}
    for mode, mp, _ in trained.known_maps():
        mm = trained.map_model(mp, mode)
        base = DraftEngine(mm)
        meta = [int(i) for i in np.argsort(-base.p_ban)[:3]]
        st = DraftState(bans=meta, we_first=False, opp_bans=meta)
        avail = np.ones(len(mm.S), bool); avail[meta] = False
        p0 = base.opponent_dist([], [], avail)
        for lvl in ("low", "medium"):
            eng = base.with_gamma(BAN_INTENT_PRESETS[lvl])
            d = tv(p0, eng.opponent_dist([], [], avail, eng.opp_bias(st)))
            worst[lvl] = max(worst.get(lvl, 0.0), d)
            if lvl == "low":
                assert not eng.read_bans(st)["meaningful"], mp
    assert worst["low"] < 0.02 and worst["medium"] < 0.04, worst


def test_off_reproduces_current_behaviour_via_api(trained):
    """ban_intent "off" with per-team bans == the old request with the union in `bans`."""
    import server
    md = trained
    mm = thrower_map(md)
    name_to_id = {v["name"]: int(b) for b, v in md.meta.items()}
    divers = [name_to_id[n] for n in ("Mortis", "Shade", "Leon")]
    ours = [name_to_id["Brock"]]
    base = dict(map=mm.map, mode=mm.mode, we_first=False)
    for slots in ([None] * 6, [name_to_id["Gray"], None, None, None, None, None]):
        old = server.recommend(server.DraftRequest(**base, slots=slots, bans=divers + ours))
        new = server.recommend(server.DraftRequest(**base, slots=slots, our_bans=ours, their_bans=divers,
                                                   ban_intent="off"))
        for r in (old, new):
            r.pop("ms")
            r.pop("ban_intent")
        assert old == new
        assert new["ban_read"] is None and new["ban_read_info"] is None
        low = server.recommend(server.DraftRequest(**base, slots=slots, our_bans=ours, their_bans=divers))
        assert low["ban_intent"] == {"level": "low", "gamma": BAN_INTENT_PRESETS["low"]}
        assert low["ban_read_info"] is not None


# ---------------------------------------------------------------- draft log
def test_draft_log_roundtrip(tmp_path):
    db = tmp_path / "drafts.db"
    rec = {"id": "draft-test-1", "map": "Flaring Phoenix", "mode": "knockout", "bracket": "legendary",
           "we_first": False, "our_bans": [16000001], "their_bans": [16000011, 16000086],
           "slots": [None] * 6, "my_slot": 1, "ban_intent": "low"}
    assert drafts.upsert(rec, db) == {"logged": False, "draft": None}        # no picks: not logged
    assert drafts.list_drafts(path=db) == []
    rec["slots"] = [16000000, None, None, None, None, None]
    r1 = drafts.upsert(dict(rec, win_prob=48.7), db)["draft"]
    assert r1["n_picks"] == 1 and not r1["complete"] and r1["their_bans"] == [16000011, 16000086]
    rec["slots"] = [16000000, 16000002, 16000003, 16000004, 16000005, 16000006]
    r2 = drafts.upsert(dict(rec, win_prob=61.5), db)["draft"]                # same row, updated
    assert r2["created_at"] == r1["created_at"] and r2["complete"] and r2["win_prob"] == 61.5
    assert len(drafts.list_drafts(path=db)) == 1
    assert drafts.upsert({"id": "draft-test-1", "result": "win"}, db)["draft"]["result"] == "win"
    r3 = drafts.upsert({"id": "draft-test-1", "result": "loss"}, db)["draft"]   # fix a mis-click
    assert r3["result"] == "loss" and r3["slots"] == rec["slots"] and r3["result_at"]
    r4 = drafts.upsert({"id": "draft-test-1", "result": None}, db)["draft"]     # undo
    assert r4["result"] is None and r4["result_at"] is None and r4["win_prob"] == 61.5
    drafts.upsert(dict(rec, id="draft-test-2"), db)
    assert [d["id"] for d in drafts.list_drafts(limit=1, path=db)] == ["draft-test-2"]
    assert drafts.get("draft-test-1", db)["my_slot"] == 1
    # a logged draft whose picks are all undone (and has no result) is dropped
    assert drafts.upsert(dict(rec, id="draft-test-2", slots=[None] * 6), db)["logged"] is False
    assert drafts.get("draft-test-2", db) is None
    with pytest.raises(ValueError):
        drafts.upsert({"id": "draft-test-1", "result": "draw"}, db)
    with pytest.raises(ValueError):
        drafts.upsert({"id": "x"}, db)


def test_draft_log_api_uses_separate_db(tmp_path, monkeypatch):
    import server
    db = tmp_path / "d.db"
    monkeypatch.setenv("DRAFTS_DB", str(db))
    out = server.log_draft(server.DraftLog(id="api-draft-1", map="M", mode="knockout",
                                           slots=[16000000, None, None, None, None, None]))
    assert out["logged"] and db.exists()
    assert server.log_draft(server.DraftLog(id="api-draft-1", result="win"))["draft"]["result"] == "win"
    assert [d["id"] for d in server.list_logged_drafts(limit=5)["drafts"]] == ["api-draft-1"]
    assert drafts.DEFAULT_PATH.name == "drafts.db" and drafts.DEFAULT_PATH.parent.name == "data"

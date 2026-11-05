import numpy as np
import pytest

from data import Matches, time_split
from evaluate import baseline_map_wr, metrics
from model import DraftModel, sigmoid

RNG = np.random.default_rng(0)
IDS = list(range(16000000, 16000012))          # 12 real brawler ids (meta classes exist)


def synth(n=4000, strong=16000000, counter=(16000001, 16000002), maps=("M1", "M2"),
          rare_map=None, seed=0):
    """Random 3v3 games where `strong` gets +0.8 logit and counter[0] beats counter[1] by 1.0."""
    rng = np.random.default_rng(seed)
    A, B, y, mp = [], [], [], []
    for i in range(n):
        picks = rng.choice(IDS, 6, replace=False)
        a, b = picks[:3], picks[3:]
        z = 0.8 * ((strong in a) - (strong in b))
        z += 1.0 * ((counter[0] in a and counter[1] in b) - (counter[0] in b and counter[1] in a))
        A.append(a); B.append(b)
        y.append(float(rng.random() < sigmoid(z)))
        mp.append(maps[i % len(maps)])
    if rare_map:
        for j in range(10):
            mp[j] = rare_map
    n = len(y)
    return Matches(ts=np.arange(n) * 60 + 1_700_000_000, map=mp, mode=["gemGrab"] * n,
                   team_a=np.array(A), team_b=np.array(B), y=np.array(y),
                   rank_a=np.full(n, 18.0), rank_b=np.full(n, 18.0))


PARAMS = {"lam_g": 2.0, "lam_mode": 8.0, "lam_map": 20.0, "lam_cmode": 4.0, "lam_cmap": 10.0,
          "lam_syn": 30.0, "lam_cnt": 30.0, "half_life_days": 0, "bracket_min_tier": 0,
          "use_patch": False}


@pytest.fixture(scope="module")
def fitted():
    m = synth()
    return m, DraftModel(PARAMS).fit(m)


def test_recovers_strong_brawler(fitted):
    m, md = fitted
    mm = md.map_model("M1", "gemGrab")
    i = mm.id_to_idx[16000000]
    others = [mm.id_to_idx[b] for b in IDS if b != 16000000]
    assert mm.S[i] > mm.S[others].max() + 0.3


def test_recovers_counter(fitted):
    _, md = fitted
    mm = md.map_model("M1", "gemGrab")
    a, b = mm.id_to_idx[16000001], mm.id_to_idx[16000002]
    assert mm.C[a, b] > 0.4
    assert mm.C[b, a] == pytest.approx(-mm.C[a, b])
    assert np.allclose(mm.C, -mm.C.T)
    assert np.allclose(mm.Syn, mm.Syn.T)


def test_antisymmetric_predictions(fitted):
    m, md = fitted
    p = md.predict_proba(m)
    pf = md.predict_proba(m.flipped())
    assert np.allclose(p + pf, 1.0)


def test_decomposition_matches_prediction(fitted):
    """MapModel.team_logit (what the draft engine uses) == the model's logit."""
    m, md = fitted
    z = md.predict_logit(m, rank_zero=True)
    for i in range(20):
        mm = md.map_model(m.map[i], m.mode[i])
        a = [mm.id_to_idx[int(x)] for x in m.team_a[i]]
        b = [mm.id_to_idx[int(x)] for x in m.team_b[i]]
        assert mm.team_logit(a, b) == pytest.approx(z[i], abs=1e-9)
        # per-brawler contributions sum to the team logit
        tot = sum(sum(mm.contribution(x, a, b).values()) for x in a)
        tot -= sum(mm.S[x] for x in b) + sum(mm.Syn[x, y] for k, x in enumerate(b) for y in b[k + 1:])
        tot -= 0.5 * sum(mm.Syn[x, y] for x in a for y in a if x != y)  # pairs counted twice
        assert tot == pytest.approx(z[i], abs=1e-9)


def test_sparse_map_is_shrunk_and_unseen_map_falls_back():
    m = synth(rare_map="RARE")
    md = DraftModel(PARAMS).fit(m)
    rare = md.map_model("RARE", "gemGrab")
    unseen = md.map_model("NEVER_SEEN", "gemGrab")
    # map-level deviations on a 10-game map are tiny compared to the global signal
    assert np.abs(rare.S_parts["map"]).max() < 0.15
    assert np.all(unseen.S_parts["map"] == 0)
    i = unseen.id_to_idx[16000000]
    assert unseen.S[i] == pytest.approx(unseen.S.max())


def test_recency_weights():
    md = DraftModel({"half_life_days": 1.0})
    ts = np.array([0, 86400, 2 * 86400], dtype=float)
    w = md.sample_weights(ts)
    assert w[2] / w[1] == pytest.approx(2.0)
    assert w.mean() == pytest.approx(1.0)


def test_model_beats_baselines_on_synthetic_holdout():
    m = synth(n=6000, seed=1)
    tr, te = time_split(m)
    md = DraftModel(PARAMS).fit(tr)
    ll_model = metrics(md.predict_proba(te), te.y)["log_loss"]
    ll_wr = metrics(baseline_map_wr(tr, te), te.y)["log_loss"]
    assert ll_model < np.log(2)
    assert ll_model <= ll_wr + 1e-3


def test_save_load_roundtrip(tmp_path, fitted):
    m, md = fitted
    p = tmp_path / "m.pkl"
    md.save(p)
    md2 = DraftModel.load(p)
    assert np.allclose(md.predict_proba(m), md2.predict_proba(m))


def test_bracket_model_follows_high_tier_games():
    """Low-tier games say brawler 3 is strong, high-tier games say brawler 4 is.
    The Legendary+ bracket model should rank 4 above 3; low-tier data is only a weak prior."""
    rng = np.random.default_rng(5)
    A, B, y, tiers = [], [], [], []
    for i in range(6000):
        picks = rng.choice(IDS, 6, replace=False)
        a, b = picks[:3], picks[3:]
        high = i % 2 == 0
        star = 16000004 if high else 16000003
        z = 0.9 * ((star in a) - (star in b))
        A.append(a); B.append(b); y.append(float(rng.random() < sigmoid(z)))
        tiers.append(18 if high else 12)
    n = len(y)
    m = Matches(ts=np.arange(n) * 60, map=["M"] * n, mode=["gemGrab"] * n, team_a=np.array(A),
                team_b=np.array(B), y=np.array(y), rank_a=np.full(n, 15.0),
                rank_b=np.full(n, 15.0), tier_min=np.array(tiers))
    p = dict(PARAMS, bracket_min_tier=16, low_tier_weight=0.25, lam_dev_scale=1.0)
    md = DraftModel(p).fit(m)
    mm = md.map_model("M", "gemGrab")
    s3, s4 = mm.S[mm.id_to_idx[16000003]], mm.S[mm.id_to_idx[16000004]]
    assert s4 > s3 + 0.3
    allr = DraftModel(dict(PARAMS, bracket_min_tier=0)).fit(m).map_model("M", "gemGrab")
    # the all-ranks model can't tell them apart
    assert abs(allr.S[allr.id_to_idx[16000003]] - allr.S[allr.id_to_idx[16000004]]) < 0.2


def test_me_factor_is_heavily_shrunk():
    from personal import BrawlerStat, ME_CAP
    st = BrawlerStat()
    for _ in range(5):                      # 5 straight wins at an expected 50%
        st.w_resid += 0.5; st.w_info += 0.25
    assert 0 < 25 * st.me < 2.5             # under ~2.5 win-% points
    for _ in range(200):                    # lots of evidence -> larger, but capped
        st.w_resid += 0.2; st.w_info += 0.25
    assert st.me == pytest.approx(ME_CAP)


def test_patch_changed_brawler_adapts_faster(monkeypatch):
    """Pre-patch: brawlers 5 and 6 both strong. Post-patch: both nerfed to neutral, but only 5
    is listed as changed. The changed brawler should move further toward its post-patch stats."""
    import model as model_mod
    rng = np.random.default_rng(7)
    A, B, y, ts = [], [], [], []
    patch_ts = 1_000_000
    for i in range(6000):
        post = i >= 5000
        picks = rng.choice(IDS, 6, replace=False)
        a, b = picks[:3], picks[3:]
        z = 0.0 if post else 0.8 * (((16000005 in a) - (16000005 in b)) + ((16000006 in a) - (16000006 in b)))
        A.append(a); B.append(b); y.append(float(rng.random() < sigmoid(z)))
        ts.append(patch_ts + (i - 5000) * 60)
    n = len(y)
    m = Matches(ts=np.array(ts), map=["M"] * n, mode=["gemGrab"] * n, team_a=np.array(A),
                team_b=np.array(B), y=np.array(y), rank_a=np.full(n, 18.0), rank_b=np.full(n, 18.0),
                tier_min=np.full(n, 18))
    monkeypatch.setattr(model_mod, "load_patch_config",
                        lambda: {"patch_ts": patch_ts, "changed_ids": [16000005]})
    from model import DEFAULT_PARAMS
    p = dict(DEFAULT_PARAMS, half_life_days=0, use_rank=False, use_patch=True,
             bracket_min_tier=16, lam_dev_scale=10.0, changed_prior_share=0.1, pre_patch_weight=0.3)
    md = DraftModel(p).fit(m)
    mm = md.map_model("M", "gemGrab")
    s5, s6 = mm.S[mm.id_to_idx[16000005]], mm.S[mm.id_to_idx[16000006]]
    assert s5 < s6 - 0.04         # changed brawler sheds its stale pre-patch strength
    assert abs(s5) < 0.05
    assert md.n_in_bracket == 1000


def test_buffie_gaps_api_and_overrides(tmp_path, monkeypatch):
    import json as _json
    import personal
    avail = tmp_path / "avail.json"
    avail.write_text(_json.dumps({"SHELLY": ["gadget", "hyperCharge", "starPower"],
                                  "COLT": ["gadget", "hyperCharge", "starPower"]}))
    monkeypatch.setattr(personal, "BUFFIES_AVAILABLE", avail)
    prof = {"brawlers": [
        {"id": 1, "name": "SHELLY", "buffies": {"gadget": True, "starPower": False, "hyperCharge": False}},
        {"id": 2, "name": "COLT", "buffies": {"gadget": True, "starPower": True, "hyperCharge": True}},
        {"id": 3, "name": "PIPER", "buffies": {"gadget": False, "starPower": False, "hyperCharge": False}},
        {"id": 4, "name": "GALE", "buffies": {"gadget": False, "starPower": False, "hyperCharge": False}},
    ]}
    names = {"SHELLY": 1, "COLT": 2, "PIPER": 3, "GALE": 4}
    cfg = {"use_api_buffies": True, "missing_buffies": ["Gale"], "assume_full": [], "buffie_penalty_pct": 3}
    g = personal.buffie_gaps(prof, cfg, names)
    assert g[1][0] == pytest.approx(2 / 3)          # Shelly missing 2 of 3
    assert 2 not in g                               # Colt fully built
    assert 3 not in g                               # Piper has no buffies in the game -> not "missing"
    assert g[4][0] == 1.0 and g[4][2] == "my_brawlers.yaml"
    g2 = personal.buffie_gaps(prof, dict(cfg, assume_full=["Shelly"]), names)
    assert 1 not in g2
    p = personal.Profile(available=True, buffies=g, buffie_penalty=3 / 25)
    assert 25 * p.parts(4)["buffies"] == pytest.approx(-3.0)
    assert 25 * p.parts(1)["buffies"] == pytest.approx(-2.0)

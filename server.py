"""Local web UI + JSON API for the draft engine.

    .venv/bin/python server.py            # http://127.0.0.1:8000
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from draft import DraftEngine, DraftState, pct
from data import BRACKETS, tier_name
from model import DraftModel, model_path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"

app = FastAPI(title="Brawl Draft Engine")
_lock = threading.Lock()
DEFAULT_BRACKET = "legendary"
BRACKET_LABELS = {"legendary": "Legendary+", "masters": "Masters+", "all": "All ranks"}
_state = {"models": {}, "mtimes": {}, "engines": {}, "personal": {}}


def available_brackets() -> list[str]:
    return [b for b in BRACKETS if model_path(b).exists()]


def get_model(bracket: str = DEFAULT_BRACKET) -> DraftModel:
    """Load a bracket's model, hot-reloading if train.py wrote a newer one."""
    path = model_path(bracket)
    with _lock:
        if not path.exists():
            raise HTTPException(503, f"No trained model for bracket '{bracket}'. "
                                     f"Run: .venv/bin/python train.py --brackets {bracket}")
        mtime = path.stat().st_mtime
        if bracket not in _state["models"] or mtime > _state["mtimes"][bracket]:
            _state["models"][bracket] = DraftModel.load(path)
            _state["mtimes"][bracket] = mtime
            _state["engines"] = {k: v for k, v in _state["engines"].items() if k[0] != bracket}
            _state["personal"].pop(bracket, None)
        return _state["models"][bracket]


def get_engine(mp: str, mode: str, bracket: str = DEFAULT_BRACKET) -> DraftEngine:
    md = get_model(bracket)
    key = (bracket, mode, mp)
    eng = _state["engines"].get(key)
    if eng is None:
        eng = _state["engines"][key] = DraftEngine(md.map_model(mp, mode))
    return eng


def get_personal(bracket: str = DEFAULT_BRACKET):
    """Personal-mode profile (lazy; None if no player tag / no games)."""
    try:
        import personal
    except ImportError:
        return None
    md = get_model(bracket)
    p = _state["personal"].get(bracket)
    if p is None or time.time() - p.built_at > 600:
        p = _state["personal"][bracket] = personal.build_profile(md)
    return p


class DraftRequest(BaseModel):
    map: str
    mode: str
    we_first: bool = True
    bans: list[int] = Field(default_factory=list)
    our: list[int] = Field(default_factory=list)
    opp: list[int] = Field(default_factory=list)
    personal: bool = False
    bracket: str = DEFAULT_BRACKET
    top: int = 5


def _name(md, bid):
    return md.meta.get(int(bid), {}).get("name", str(bid))


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/meta")
def meta(bracket: str = DEFAULT_BRACKET):
    md = get_model(bracket)
    brawlers = [{"id": int(b), "name": v["name"], "class": v["class"], "image": v.get("image")}
                for b, v in sorted(md.meta.items(), key=lambda kv: kv[1]["name"])]
    maps = [{"mode": mode, "map": mp, "games": int(n)} for mode, mp, n in md.known_maps()]
    prof = get_personal(bracket)
    bt = md.params.get("bracket_min_tier", 0)
    return {
        "brawlers": brawlers,
        "maps": maps,
        "brackets": [{"id": b, "label": BRACKET_LABELS.get(b, b)} for b in available_brackets()],
        "bracket": bracket,
        "model": {"n_train": md.n_train, "trained_through": md.t_max,
                  "bracket_min_tier": bt, "bracket_tier_name": tier_name(bt) if bt else "any",
                  "n_in_bracket": int(getattr(md, "n_in_bracket", md.n_train))},
        "personal_available": bool(prof and prof.available),
    }


@app.post("/api/recommend")
def recommend(req: DraftRequest):
    t0 = time.time()
    md = get_model(req.bracket)
    eng = get_engine(req.map, req.mode, req.bracket)
    mm = eng.mm
    try:
        to_idx = lambda ids: [mm.id_to_idx[int(b)] for b in ids]
        st = DraftState(to_idx(req.our), to_idx(req.opp), to_idx(req.bans), req.we_first)
        st.validate()
    except KeyError as e:
        raise HTTPException(400, f"unknown brawler id {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))

    out = {"map": req.map, "mode": req.mode, "bracket": req.bracket, "games_on_map": mm.games_on_map,
           "step": st.step, "personal": False}
    turn = st.team_to_move()
    out["turn"] = {0: "us", 1: "opp", None: "done"}[turn]

    pool_mask = bonus = info = None
    if req.personal:
        prof = get_personal(req.bracket)
        if prof and prof.available:
            pool_mask, bonus, info = prof.vectors(mm)
            out["personal"] = True

    if st.step == 0:
        out["ban_recs"] = eng.recommend_bans(st, top=req.top)
    if turn == 0:
        out["pick_recs"] = eng.recommend_picks(st, top=req.top, pool_mask=pool_mask,
                                               bonus_vec=bonus, bonus_info=info)
    elif turn == 1:
        out["opponent_prediction"] = eng.predict_opponent(st, top=req.top)
    else:
        z = mm.team_logit(st.our, st.opp)
        out["final"] = {
            "win_prob": round(100 * float(1 / (1 + np.exp(-z))), 1),
            "ours": [{"brawler_id": int(mm.brawler_ids[x]),
                      **{k: round(pct(v), 2) for k, v in mm.contribution(x, st.our, st.opp).items()}}
                     for x in st.our],
            "theirs": [{"brawler_id": int(mm.brawler_ids[x]),
                        **{k: round(pct(v), 2) for k, v in mm.contribution(x, st.opp, st.our).items()}}
                       for x in st.opp],
        }
    # Live estimate of the current partial comps (equal skill), for the header.
    out["current_win_prob"] = round(100 * float(1 / (1 + np.exp(-mm.team_logit(st.our, st.opp)))), 1)
    for key in ("pick_recs", "ban_recs"):
        if key in out:
            for r in out[key].get("picks", []) + out[key].get("bans", []):
                r["name"] = _name(md, r["brawler_id"])
    for r in out.get("opponent_prediction", []):
        r["name"] = _name(md, r["brawler_id"])
    out["ms"] = round(1000 * (time.time() - t0))
    return out


@app.get("/api/my_stats")
def my_stats(bracket: str = DEFAULT_BRACKET):
    prof = get_personal(bracket)
    if not prof or not prof.available:
        return {"available": False}
    return prof.stats()


if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")


if __name__ == "__main__":
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")

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

import drafts
from draft import BAN_INTENT_PRESETS, PICK_ORDER, DraftEngine, DraftState, pct
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
    bf = personal.MY_BUFFIES
    stale = p is not None and bf.exists() and bf.stat().st_mtime > p.built_at
    if p is None or stale or time.time() - p.built_at > 600:
        p = _state["personal"][bracket] = personal.build_profile(md)
    return p


class DraftRequest(BaseModel):
    map: str
    mode: str
    we_first: bool = True
    # Bans of unknown team (legacy; no ban intent). Prefer our_bans / their_bans: all three
    # are unioned into the banned set, and their_bans also drive ban intent.
    bans: list[int] = Field(default_factory=list)
    our_bans: list[int] = Field(default_factory=list)
    their_bans: list[int] = Field(default_factory=list)
    # How much the enemy's bans shift their predicted picks: off / low / medium
    ban_intent: str = "low"
    our: list[int] = Field(default_factory=list)
    opp: list[int] = Field(default_factory=list)
    personal: bool = False
    bracket: str = DEFAULT_BRACKET
    top: int = 5
    # Optional free-order entry: 6 pick slots in draft order (1-2-2-1), brawler id or null.
    # When given, it replaces our/opp, and slots may be filled in any order.
    slots: list[int | None] | None = None
    # Which slot (0-5, draft order) is the user. Personal mode applies only to that pick.
    my_slot: int | None = None
    # Which empty slot to advise on (default: first empty slot in draft order). Earlier
    # slots that are still empty are treated as unknown picks (searched over), so you get a
    # usable answer even if you didn't have time to enter everything.
    target_slot: int | None = None


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
    if req.ban_intent not in BAN_INTENT_PRESETS:
        raise HTTPException(400, f"ban_intent must be one of {', '.join(BAN_INTENT_PRESETS)}")
    eng = eng.with_gamma(BAN_INTENT_PRESETS[req.ban_intent])
    try:
        to_idx = lambda ids: [mm.id_to_idx[int(b)] for b in ids]
        # both teams may ban the same brawler
        bans = to_idx(dict.fromkeys(list(req.bans) + list(req.our_bans) + list(req.their_bans)))
        opp_bans = to_idx(dict.fromkeys(req.their_bans))
        next_slot = None
        if req.slots is not None:
            if len(req.slots) != 6:
                raise ValueError("slots must have 6 entries (draft order)")
            is_us = [(PICK_ORDER[i] == 0) == req.we_first for i in range(6)]
            our = [b for i, b in enumerate(req.slots) if b is not None and is_us[i]]
            opp = [b for i, b in enumerate(req.slots) if b is not None and not is_us[i]]
            empty = [i for i in range(6) if req.slots[i] is None]
            next_slot = empty[0] if empty else None
            if req.target_slot is not None and req.target_slot in empty:
                next_slot = req.target_slot
            order = ([next_slot] + [i for i in empty if i != next_slot]) if empty else []
            st = DraftState(to_idx(our), to_idx(opp), bans, req.we_first,
                            seq=tuple(is_us[i] for i in order), opp_bans=opp_bans)
            unknown_before = sum(1 for i in empty if i < (next_slot if next_slot is not None else 6))
        else:
            st = DraftState(to_idx(req.our), to_idx(req.opp), bans, req.we_first, opp_bans=opp_bans)
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
    # Personal mode only for the user's own slot (teammates are assumed fully built).
    mine = req.my_slot is None or next_slot is None or req.my_slot == next_slot
    out["for_my_slot"] = bool(mine and turn == 0)
    out["next_slot"] = next_slot
    out["unknown_before"] = unknown_before if req.slots is not None else 0
    if req.personal and mine:
        prof = get_personal(req.bracket)
        if prof and prof.available:
            pool_mask, bonus, info = prof.vectors(mm)
            out["personal"] = True

    if st.step == 0:
        out["ban_recs"] = eng.recommend_bans(st, top=req.top)
    if turn == 0:
        out["pick_recs"] = eng.recommend_picks(st, top=req.top, pool_mask=pool_mask,
                                               bonus_vec=bonus, bonus_info=info, compare_intent=3)
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
            # role-level matchup notes for the finished comps, e.g. "Their 3 divers hard-counter
            # your 3 throwers (-12.0%)"
            "comp_notes": mm.comp_notes(st.our, st.opp) if hasattr(mm, "comp_notes") else [],
        }
    # What the draft is worth from here if every remaining pick is realistic (pick-rate weighted,
    # leaning strong): the apples-to-apples reference for pick recs' win_prob, unlike
    # current_win_prob, which scores only the picks made so far.
    if turn == 0:
        out["expected_win_prob"] = out["pick_recs"]["baseline_win_prob"]
    elif turn == 1:
        cfg = eng.cfg
        out["expected_win_prob"] = round(100 * eng.draft_value(
            st, {"max": cfg.max_width, "chance": cfg.chance_width}), 1)
    else:
        out["expected_win_prob"] = out["final"]["win_prob"]
    # Role-level notes on the current comps (also per pick rec, and in `final`).
    out["comp_notes"] = mm.comp_notes(st.our, st.opp) if hasattr(mm, "comp_notes") else []
    # Live estimate of the current partial comps (equal skill), for the header.
    out["current_win_prob"] = round(100 * float(1 / (1 + np.exp(-mm.team_logit(st.our, st.opp)))), 1)
    for key in ("pick_recs", "ban_recs"):
        if key in out:
            for r in out[key].get("picks", []) + out[key].get("bans", []):
                r["name"] = _name(md, r["brawler_id"])
    for r in out.get("opponent_prediction", []):
        r["name"] = _name(md, r["brawler_id"])
    # Ban intent: what the enemy's bans say about their plan. `ban_read` is a plain-words note,
    # only when it matters (moves their next-pick distribution or a top rec noticeably).
    out["ban_intent"] = {"level": req.ban_intent, "gamma": eng.cfg.gamma}
    br = eng.read_bans(st, (out.get("pick_recs") or {}).get("picks"), lambda b: _name(md, b))
    out["ban_read_info"] = br
    out["ban_read"] = br["text"] if br and br["meaningful"] else None
    out["pick_rates"] = {int(b): round(600 * float(p), 1) for b, p in zip(mm.brawler_ids, mm.prior)}
    out["ms"] = round(1000 * (time.time() - t0))
    return out


class DraftLog(BaseModel):
    """A draft as the UI ran it. Only fields that are sent get updated (so {id, result} alone
    changes just the result; result null clears it)."""
    id: str
    map: str | None = None
    mode: str | None = None
    bracket: str | None = None
    we_first: bool | None = None
    our_bans: list[int | None] | None = None
    their_bans: list[int | None] | None = None
    slots: list[int | None] | None = None
    my_slot: int | None = None
    win_prob: float | None = None
    expected_win_prob: float | None = None
    ban_intent: str | None = None
    personal: bool | None = None
    result: str | None = None


@app.post("/api/drafts")
def log_draft(d: DraftLog):
    """Create/update the draft log row for a client-generated draft id (data/drafts.db)."""
    rec = d.model_dump(exclude_unset=True)
    if "slots" in rec and "model_trained_through" not in rec:
        try:
            rec["model_trained_through"] = float(get_model(d.bracket or DEFAULT_BRACKET).t_max)
        except Exception:
            pass
    try:
        return drafts.upsert(rec)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/drafts")
def list_logged_drafts(limit: int = 50):
    return {"drafts": drafts.list_drafts(limit)}


class BuffiesUpdate(BaseModel):
    text: str


@app.get("/api/my_buffies")
def get_my_buffies():
    """Contents of reference/my_buffies.txt (brawlers you're MISSING buffies on)."""
    import personal
    md = get_model()
    name_to_id = {v["name"]: int(b) for b, v in md.meta.items()}
    names = personal.read_buffies_file() or []
    return {"path": str(personal.MY_BUFFIES.relative_to(ROOT)),
            "exists": personal.MY_BUFFIES.exists(),
            "text": personal.MY_BUFFIES.read_text() if personal.MY_BUFFIES.exists() else "",
            "resolved": [{"typed": n, "brawler": md.meta.get(personal.resolve_name(n, name_to_id), {}).get("name")}
                         for n in names]}


@app.put("/api/my_buffies")
def put_my_buffies(upd: BuffiesUpdate):
    """Overwrite reference/my_buffies.txt; personal mode picks it up immediately."""
    import personal
    md = get_model()
    name_to_id = {v["name"]: int(b) for b, v in md.meta.items()}
    text = upd.text if upd.text.endswith("\n") else upd.text + "\n"
    names = [l.split("#", 1)[0].strip() for l in text.splitlines()]
    unknown = [n for n in names if n and personal.resolve_name(n, name_to_id) is None]
    if unknown:
        raise HTTPException(400, f"unknown brawler name(s): {', '.join(unknown)}")
    personal.MY_BUFFIES.write_text(text)
    _state["personal"].clear()
    return get_my_buffies()


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

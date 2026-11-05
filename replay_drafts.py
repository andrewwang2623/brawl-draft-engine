"""Replay logged drafts (data/drafts.db) through the current engine.

For every logged draft and every one of OUR pick slots, rebuild the state just before that pick
(the picks that were actually made earlier, in draft order) and ask the engine for its top 5
for that slot. Reports where the brawler actually picked ranks, and what the enemy's
remaining picks turned out to be.

    .venv/bin/python replay_drafts.py --out models/replay_after.json [--set risk=safe]
    .venv/bin/python replay_drafts.py --compare models/replay_before.json models/replay_after.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DRAFTS_DB = ROOT / "data" / "drafts.db"
PICK_ORDER = (0, 1, 1, 0, 0, 1)


def load_drafts(path: Path = DRAFTS_DB) -> list[dict]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cols = [r[1] for r in con.execute("PRAGMA table_info(drafts)")]
    out = []
    for row in con.execute("SELECT * FROM drafts WHERE n_picks >= 1 ORDER BY created_at"):
        d = dict(zip(cols, row))
        for k in ("slots", "our_bans", "their_bans"):
            if isinstance(d.get(k), str):
                d[k] = json.loads(d[k])
        out.append(d)
    return out


def replay(extra: dict) -> list[dict]:
    warnings.filterwarnings("ignore")
    from fastapi.testclient import TestClient
    import server
    c = TestClient(server.app)
    names = {b["id"]: b["name"] for b in c.get("/api/meta").json()["brawlers"]}
    results = []
    for d in load_drafts():
        we_first = bool(d["we_first"])
        is_us = [(PICK_ORDER[i] == 0) == we_first for i in range(6)]
        slots = d["slots"]
        draft = {"id": d["id"], "map": d["map"], "mode": d["mode"], "result": d.get("result"),
                 "we_first": we_first, "my_slot": d.get("my_slot"),
                 "slots": [names.get(s) if s else None for s in slots], "decisions": []}
        for k in range(6):
            if not is_us[k] or slots[k] is None:
                continue
            state = [s if i < k else None for i, s in enumerate(slots)]
            body = {"map": d["map"], "mode": d["mode"], "we_first": we_first,
                    "our_bans": d.get("our_bans") or [], "their_bans": d.get("their_bans") or [],
                    "slots": state, "target_slot": k, "my_slot": d.get("my_slot"),
                    "ban_intent": d.get("ban_intent") or "low", "top": 5, **extra}
            r = c.post("/api/recommend", json=body).json()
            picks = (r.get("pick_recs") or {}).get("picks", [])
            actual = slots[k]
            rank = next((i + 1 for i, p in enumerate(picks) if p["brawler_id"] == actual), None)
            later_enemy = [names.get(slots[i]) for i in range(k + 1, 6) if not is_us[i] and slots[i]]
            draft["decisions"].append({
                "slot": k, "actual": names.get(actual), "actual_rank_in_top5": rank,
                "later_enemy_picks": later_enemy,
                "top5": [{"name": p["name"], "win_prob": p["win_prob"],
                          "risk_adjusted": p.get("risk_adjusted_win_prob"),
                          "biggest_risk": (p.get("risk_answers") or [{}])[0].get("name"),
                          "biggest_risk_pct": (p.get("risk_answers") or [{}])[0].get("damage_pct")}
                         for p in picks]})
        results.append(draft)
    return results


def compare(a_path: str, b_path: str):
    A = {d["id"]: d for d in json.loads(Path(a_path).read_text())}
    B = {d["id"]: d for d in json.loads(Path(b_path).read_text())}
    for did, b in B.items():
        a = A.get(did)
        if not a:
            continue
        print(f"\n{b['map']} ({'we pick first' if b['we_first'] else 'they pick first'}, result: {b['result']})"
              f"  picks: {' / '.join(str(s) for s in b['slots'])}")
        for da, db in zip(a["decisions"], b["decisions"]):
            ta = [p["name"] for p in da["top5"]]
            tb = [p["name"] for p in db["top5"]]
            changed = ta != tb
            print(f"  pick {db['slot'] + 1}: actual {db['actual']}  (enemy later: {', '.join(db['later_enemy_picks']) or '-'})")
            print(f"     before: {', '.join(ta)}   [actual ranked {da['actual_rank_in_top5'] or '>5'}]")
            print(f"     after : {', '.join(tb)}   [actual ranked {db['actual_rank_in_top5'] or '>5'}]"
                  + ("" if changed else "  (unchanged)"))
            for p in db["top5"][:3]:
                if p.get("biggest_risk"):
                    print(f"        {p['name']}: biggest risk {p['biggest_risk']} ({p['biggest_risk_pct']:+.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--set", action="append", default=[], help="extra request field key=value")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
        return
    extra = {}
    for s in a.set:
        k, v = s.split("=", 1)
        extra[k] = v
    res = replay(extra)
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    else:
        sys.stdout.write(text)
    print(f"replayed {len(res)} drafts, {sum(len(d['decisions']) for d in res)} decisions", file=sys.stderr)


if __name__ == "__main__":
    main()

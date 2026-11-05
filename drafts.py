"""Draft log: every draft the user runs in the UI, in its own SQLite file (data/drafts.db).

Kept separate from data/matches.db, which the crawler writes continuously. One row per draft,
keyed by a client-generated id; the UI upserts the same row as the draft evolves and when the
user records the result. Purpose: fit the ban-intent gamma (and check calibration) on real
drafts later (see DECISIONS.md, "Ban intent").

    .venv/bin/python drafts.py            # print the latest drafts
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_PATH = ROOT / "data" / "drafts.db"
RESULTS = ("win", "loss")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
# JSON-encoded list columns
_LISTS = ("our_bans", "their_bans", "slots")

SCHEMA = """
CREATE TABLE IF NOT EXISTS drafts (
    id TEXT PRIMARY KEY,               -- client-generated draft id
    created_at REAL NOT NULL,          -- unix seconds (server clock)
    updated_at REAL NOT NULL,
    map TEXT, mode TEXT, bracket TEXT,
    we_first INTEGER,                  -- 1 = we picked first
    our_bans TEXT, their_bans TEXT,    -- JSON lists of brawler ids
    slots TEXT,                        -- JSON list of 6 brawler ids / null, in draft order (1-2-2-1)
    my_slot INTEGER,                   -- the user's slot (0-5) or NULL
    n_picks INTEGER,
    complete INTEGER,                  -- all 6 picks entered
    win_prob REAL,                     -- last win% shown (final comp if complete, else "so far")
    expected_win_prob REAL,            -- the draft's expected value as last shown
    ban_intent TEXT,                   -- off / low / medium at the time
    personal INTEGER,
    model_trained_through REAL,        -- model snapshot used (for refits)
    result TEXT,                       -- 'win' / 'loss' / NULL (not recorded)
    result_at REAL
);
CREATE INDEX IF NOT EXISTS drafts_updated ON drafts(updated_at);
"""


def db_path() -> Path:
    return Path(os.environ.get("DRAFTS_DB", DEFAULT_PATH))


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(path) if path is not None else db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=5)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


@contextmanager
def _db(path=None):
    """Connection that commits on success, rolls back on error, and always closes."""
    con = connect(path)
    try:
        with con:
            yield con
    finally:
        con.close()


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    for k in _LISTS:
        d[k] = json.loads(d[k]) if d.get(k) else []
    for k in ("we_first", "complete", "personal"):
        if d.get(k) is not None:
            d[k] = bool(d[k])
    return d


def get(draft_id: str, path=None) -> dict | None:
    with _db(path) as con:
        r = con.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    return _row(r) if r else None


def upsert(rec: dict, path=None) -> dict:
    """Create or update the draft `rec["id"]`. Only keys present in rec are changed (so
    {"id": ..., "result": None} just clears a mis-clicked result). Drafts with no picks are not
    logged; a logged draft that drops back to no picks (and has no result) is deleted.
    Returns {"logged": bool, "draft": row or None}."""
    did = str(rec.get("id") or "")
    if not ID_RE.match(did):
        raise ValueError("draft id must be 6-64 chars of [A-Za-z0-9_-]")
    if "result" in rec and rec["result"] not in (None, *RESULTS):
        raise ValueError("result must be 'win', 'loss' or null")
    if "slots" in rec and rec["slots"] is not None and len(rec["slots"]) != 6:
        raise ValueError("slots must have 6 entries (draft order)")
    now = time.time()
    with _db(path) as con:
        old = con.execute("SELECT * FROM drafts WHERE id = ?", (did,)).fetchone()
        cur = _row(old) if old else {}
        new = dict(cur)
        for k, v in rec.items():
            if k in ("id", "created_at", "updated_at", "n_picks", "complete", "result_at"):
                continue
            new[k] = v
        slots = list(new.get("slots") or [None] * 6)
        n_picks = sum(1 for x in slots if x is not None)
        if n_picks == 0:
            if old and not new.get("result"):
                con.execute("DELETE FROM drafts WHERE id = ?", (did,))
            if not old or not new.get("result"):
                return {"logged": False, "draft": None}
        result_at = cur.get("result_at")
        if new.get("result") != cur.get("result"):
            result_at = now if new.get("result") else None
        row = {
            "id": did, "created_at": cur.get("created_at", now), "updated_at": now,
            "map": new.get("map"), "mode": new.get("mode"), "bracket": new.get("bracket"),
            "we_first": None if new.get("we_first") is None else int(bool(new["we_first"])),
            "our_bans": json.dumps([int(x) for x in new.get("our_bans") or [] if x is not None]),
            "their_bans": json.dumps([int(x) for x in new.get("their_bans") or [] if x is not None]),
            "slots": json.dumps([None if x is None else int(x) for x in slots]),
            "my_slot": new.get("my_slot"), "n_picks": n_picks, "complete": int(n_picks == 6),
            "win_prob": new.get("win_prob"), "expected_win_prob": new.get("expected_win_prob"),
            "ban_intent": new.get("ban_intent"),
            "personal": None if new.get("personal") is None else int(bool(new["personal"])),
            "model_trained_through": new.get("model_trained_through"),
            "result": new.get("result"), "result_at": result_at,
        }
        cols = list(row)
        con.execute(f"INSERT INTO drafts ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
                    f"ON CONFLICT(id) DO UPDATE SET {', '.join(f'{c} = excluded.{c}' for c in cols if c != 'id')}",
                    [row[c] for c in cols])
        r = con.execute("SELECT * FROM drafts WHERE id = ?", (did,)).fetchone()
    return {"logged": True, "draft": _row(r)}


def list_drafts(limit: int = 50, path=None) -> list[dict]:
    limit = max(1, min(int(limit), 1000))
    with _db(path) as con:
        rows = con.execute("SELECT * FROM drafts ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


if __name__ == "__main__":
    for d in list_drafts(20):
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(d["updated_at"]))
        print(f"{when}  {d['map']} ({d['mode']})  picks {d['n_picks']}/6  "
              f"win% {d['win_prob']}  result {d['result'] or '-'}")

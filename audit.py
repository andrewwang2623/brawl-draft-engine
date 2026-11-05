"""Global recommendation audit: top-10 first picks on EVERY map, with sanity flags.

    .venv/bin/python audit.py                         # audit models/model_legendary.pkl
    .venv/bin/python audit.py --label after --compare models/audit_before.json
    .venv/bin/python audit.py --json                  # machine-readable

For each map it runs the real draft engine for a blind first pick (we pick first, nothing
banned) and reports the top 10 with games on the map, pick rate, raw win rate and every
score component. Flags:

  low_data_top5   a top-5 pick with < MIN_GAMES_TOP5 games on the map
  zero_game_map   a brawler with 0 games on the map whose map strength (relative to the
                  pick-weighted typical pick) is > ZERO_GAME_PCT win-% points
  low_overlap     top 10 shares < MIN_OVERLAP brawlers with the map's 15 most picked

A map is "flagged" if it has any flag. It also runs the matchup sanity checks in
checks/matchups.yaml (domain knowledge, never training data): the model's head-to-head effect
must point the right way by at least margin_pct, reported with raw head-to-head sample sizes
in reference/MATCHUP_CHECKS.md. experiment.py runs this for baseline vs candidate: more
flags or a lower matchup pass rate = failed experiment, whatever the log loss says.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from data import Matches, load_matches
from draft import DraftEngine, DraftState, pct
from model import DraftModel, model_path

ROOT = Path(__file__).resolve().parent
MATCHUPS = ROOT / "checks" / "matchups.yaml"
MIN_GAMES_TOP5 = 30
ZERO_GAME_PCT = 3.0
MIN_OVERLAP = 4
TOP_PICKED = 15


def map_stats(md: DraftModel, m: Matches) -> dict:
    """(mode, map) -> {'n': games, 'games': {bid: n}, 'wins': {bid: w}} on target games."""
    tgt = np.flatnonzero(md.in_target(m))
    out = defaultdict(lambda: {"n": 0, "games": defaultdict(int), "wins": defaultdict(float)})
    for i in tgt:
        s = out[(m.mode[i], m.map[i])]
        s["n"] += 1
        for team, won in ((m.team_a[i], m.y[i]), (m.team_b[i], 1 - m.y[i])):
            for x in team:
                s["games"][int(x)] += 1
                s["wins"][int(x)] += won
    return out


def audit_model(md: DraftModel, m: Matches | None = None, top: int = 10) -> dict:
    m = load_matches() if m is None else m
    stats = map_stats(md, m)
    name = {int(b): v["name"] for b, v in md.meta.items()}
    maps, flagged = [], 0
    counts = {"low_data_top5": 0, "zero_game_map": 0, "low_overlap": 0}
    tot_overlap = tot_overlap_win = 0
    for mode, mp, _ in md.known_maps():
        st = stats.get((mode, mp))
        if not st or st["n"] == 0:
            continue
        mm = md.map_model(mp, mode)
        eng = DraftEngine(mm)
        r = eng.recommend_picks(DraftState(we_first=True), top=top)
        n = st["n"]
        g = lambda b: st["games"].get(int(b), 0)
        wr = lambda b: (st["wins"].get(int(b), 0.0) / g(b)) if g(b) else None
        picked = sorted(st["games"], key=lambda b: -st["games"][b])
        top_picked = set(picked[:TOP_PICKED])
        # "picked AND winning": among the 25 most picked, raw win rate >= 50%
        pick_win = {b for b in picked[:25] if g(b) >= 50 and wr(b) >= 0.5}
        # map strength relative to the pick-weighted typical pick (prediction-invariant)
        S = mm.S - float(np.dot(mm.prior, mm.S))
        recs = []
        for p in r["picks"]:
            b = p["brawler_id"]
            recs.append({
                "name": name.get(b, str(b)), "brawler_id": b, "games": g(b),
                "pick_rate": round(100 * g(b) / n, 2),
                "win_rate": None if wr(b) is None else round(100 * wr(b), 1),
                "win_prob": p["win_prob"], "win_prob_p25": p.get("win_prob_p25"),
                "confidence": p.get("confidence"), "delta_vs_typical": p["delta_vs_typical"],
                "map_strength_centered": round(pct(S[mm.id_to_idx[b]]), 2),
                "breakdown_pct": p["breakdown_pct"], "strength_parts_pct": p["strength_parts_pct"],
            })
        flags = []
        for rec in recs[:5]:
            if rec["games"] < MIN_GAMES_TOP5:
                flags.append(f"low_data_top5: {rec['name']} ({rec['games']} games)")
        zero = [(name.get(int(b), str(b)), round(pct(S[i]), 1)) for i, b in enumerate(mm.brawler_ids)
                if g(b) == 0 and pct(S[i]) > ZERO_GAME_PCT]
        for nm, v in sorted(zero, key=lambda t: -t[1]):
            flags.append(f"zero_game_map: {nm} ({v:+.1f}%)")
        ids10 = {rec["brawler_id"] for rec in recs}
        overlap = len(ids10 & top_picked)
        overlap_win = len(ids10 & pick_win)
        tot_overlap += overlap
        tot_overlap_win += overlap_win
        if overlap < MIN_OVERLAP:
            flags.append(f"low_overlap: {overlap}/10 in the {TOP_PICKED} most picked")
        for k in counts:
            counts[k] += any(f.startswith(k) for f in flags)
        flagged += bool(flags)
        maps.append({"mode": mode, "map": mp, "games": n, "baseline_win_prob": r["baseline_win_prob"],
                     "overlap_top15": overlap, "overlap_picked_winning": overlap_win,
                     "n_picked_winning": len(pick_win), "flags": flags, "top": recs})
    nm = max(1, len(maps))
    return {"n_maps": len(maps), "n_flagged": flagged, "flag_counts": counts,
            "n_flags": sum(len(x["flags"]) for x in maps),
            "mean_overlap_top15": round(tot_overlap / nm, 2),
            "mean_overlap_picked_winning": round(tot_overlap_win / nm, 2),
            "mean_baseline_win_prob": round(float(np.mean([x["baseline_win_prob"] for x in maps])), 1),
            "low_data_top5_total": sum(1 for x in maps for f in x["flags"] if f.startswith("low_data")),
            "matchups": matchup_checks(md, m) if hasattr(md, "off") else {},
            "maps": maps}


def regressed(base: dict, cand: dict) -> bool:
    """Audit regression = more flagged maps, more total flags, or a lower matchup-check pass
    rate than the baseline."""
    mb, mc = base.get("matchups", {}), cand.get("matchups", {})
    return (cand["n_flagged"] > base["n_flagged"] or cand["n_flags"] > base["n_flags"]
            or mc.get("pass_rate", 1.0) < mb.get("pass_rate", 0.0) - 1e-9)


def matchup_checks(md: DraftModel, m: Matches, path: Path = MATCHUPS) -> dict:
    """Head-to-head checks from checks/matchups.yaml against the model's counter matrix C.
    Model effect = C[counter, victim] in win-% points, games-weighted over the maps that match
    the entry's map/mode filter. Raw = target games where they faced each other."""
    import yaml
    from model import ROLE_NAMES, load_roles
    if not path.exists():
        return {"n": 0, "n_pass": 0, "pass_rate": 1.0, "rows": []}
    cfg = yaml.safe_load(path.read_text()) or {}
    margin = float(cfg.get("margin_pct", 0.5))
    name_to_id = {v["name"].upper(): int(b) for b, v in md.meta.items()}
    roles = load_roles()
    by_role = {r: [name_to_id[n] for n, rs in roles.items() if r in rs and n in name_to_id] for r in ROLE_NAMES}
    tgt = np.flatnonzero(md.in_target(m))
    A, B, Y = m.team_a[tgt], m.team_b[tgt], m.y[tgt]
    modes = [m.mode[i] for i in tgt]; maps = [m.map[i] for i in tgt]
    maps_all = md.known_maps()

    def model_effect(xs, ys, mp=None, mode=None):
        tot = wsum = 0.0
        for mo, mpn, c in maps_all:
            if (mp and mpn != mp) or (mode and mo != mode):
                continue
            mm = md.map_model(mpn, mo)
            ix = [mm.id_to_idx[x] for x in xs if x in mm.id_to_idx]
            iy = [mm.id_to_idx[y] for y in ys if y in mm.id_to_idx]
            if not ix or not iy:
                continue
            tot += c * float(mm.C[np.ix_(ix, iy)].mean())
            wsum += c
        return 25.0 * tot / wsum if wsum else 0.0

    def raw(xs, ys, mp=None, mode=None):
        xs, ys = np.array(xs), np.array(ys)
        ok = np.ones(len(tgt), bool)
        if mp:
            ok &= np.array([q == mp for q in maps])
        if mode:
            ok &= np.array([q == mode for q in modes])
        xa, xb = np.isin(A, xs).any(1), np.isin(B, xs).any(1)
        ya, yb = np.isin(A, ys).any(1), np.isin(B, ys).any(1)
        s1 = ok & xa & yb & ~xb & ~ya          # counter on A, victim on B
        s2 = ok & xb & ya & ~xa & ~yb
        n = int(s1.sum() + s2.sum())
        w = float(Y[s1].sum() + (1 - Y[s2]).sum())
        return n, (w / n if n else None)

    rows = []
    for e in cfg.get("checks", []):
        mp, mode, note = e.get("map"), e.get("mode"), e.get("note", "")
        if "counter_role" in e:
            pairs = [(f"role:{e['counter_role']}", f"role:{e['victim_role']}",
                      by_role.get(e["counter_role"], []), by_role.get(e["victim_role"], []))]
        else:
            vs = e["victim"] if isinstance(e["victim"], list) else [e["victim"]]
            pairs = [(e["counter"], v, [name_to_id.get(e["counter"].upper())], [name_to_id.get(v.upper())])
                     for v in vs]
        for cn, vn, xs, ys in pairs:
            if not xs or not ys or None in xs or None in ys:
                rows.append({"counter": cn, "victim": vn, "status": "unknown name", "note": note})
                continue
            eff = model_effect(xs, ys, mp, mode)
            n, wr = raw(xs, ys, mp, mode)
            z = (wr - 0.5) / np.sqrt(0.25 / n) if n else 0.0
            disagree = n >= 200 and z <= -2.0
            status = "pass" if eff >= margin else ("data disagrees" if disagree else "fail")
            rows.append({"counter": cn, "victim": vn, "map": mp, "mode": mode, "model_pct": round(eff, 2),
                         "games": n, "raw_wr": None if wr is None else round(100 * wr, 1),
                         "z": round(float(z), 1), "status": status, "note": note})
    scored = [r for r in rows if r["status"] != "unknown name"]
    n_pass = sum(r["status"] == "pass" for r in scored)
    return {"n": len(scored), "n_pass": n_pass, "margin_pct": margin,
            "pass_rate": round(n_pass / len(scored), 3) if scored else 1.0, "rows": rows}


def matchups_markdown(res: dict) -> str:
    L = ["# Matchup sanity checks", "",
         "_Generated by `audit.py` from `checks/matchups.yaml` (domain knowledge; never used for "
         "training). Model = counter term C(counter, victim) in win-% points, games-weighted over "
         "matching maps. Raw = Legendary+ post-patch games where they faced each other (counter-side "
         "win rate, not adjusted for anything else). \"data disagrees\" = the model fails the check AND "
         "≥200 raw games point the other way (z ≤ -2): left to the user, not forced._", "",
         f"**{res['n_pass']}/{res['n']} pass** (margin {res.get('margin_pct', 0.5)} win-% points).", "",
         "| counter | victim | filter | model | raw games | raw WR | status | note |",
         "|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        filt = r.get("map") or r.get("mode") or "all maps"
        L.append(f"| {r['counter']} | {r['victim']} | {filt} | {r.get('model_pct', '-'):+} | "
                 f"{r.get('games', '-')} | {r.get('raw_wr', '-')}% | {r['status']} | {r.get('note', '')} |"
                 if "model_pct" in r else f"| {r['counter']} | {r['victim']} | - | - | - | - | {r['status']} | |")
    return "\n".join(L) + "\n"


def summary_line(a: dict) -> str:
    c = a["flag_counts"]
    return (f"{a['n_flagged']}/{a['n_maps']} maps flagged ({a['n_flags']} flags: low-data top-5 on "
            f"{c['low_data_top5']} maps, 0-game map term on {c['zero_game_map']}, low overlap on "
            f"{c['low_overlap']}); mean top-10 overlap with 15 most picked "
            f"{a['mean_overlap_top15']}/10, with picked-and-winning {a['mean_overlap_picked_winning']}; "
            f"mean typical-pick baseline {a['mean_baseline_win_prob']}%"
            + (f"; matchup checks {a['matchups']['n_pass']}/{a['matchups']['n']}" if a.get("matchups") else ""))


def to_markdown(a: dict, title: str) -> str:
    L = [f"### {title}", "", "- " + summary_line(a), ""]
    worst = sorted(a["maps"], key=lambda x: -len(x["flags"]))
    worst = [x for x in worst if x["flags"]][:8]
    if worst:
        L += ["Worst offenders:", "", "| map | flags |", "|---|---|"]
        for x in worst:
            L.append(f"| {x['map']} ({x['mode']}) | {'; '.join(x['flags'][:6])}"
                     f"{' …' if len(x['flags']) > 6 else ''} |")
        L.append("")
    L += ["Top 5 per map (games on map / pick rate / raw WR; win% [p25]; map strength vs typical pick):", "",
          "| map | overlap | top 5 |", "|---|---|---|"]
    for x in a["maps"]:
        cells = []
        for r in x["top"][:5]:
            p25 = f" [{r['win_prob_p25']}]" if r.get("win_prob_p25") is not None else ""
            wrs = f"{r['win_rate']:.0f}%" if r["win_rate"] is not None else "-"
            cells.append(f"{r['name']} ({r['games']}/{r['pick_rate']:.0f}%/{wrs}; "
                         f"{r['win_prob']}{p25}; {r['map_strength_centered']:+.1f})")
        L.append(f"| {x['map']} | {x['overlap_top15']} | {', '.join(cells)} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(model_path("legendary")))
    ap.add_argument("--label", default="latest")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--map", default=None, help="print the full top 10 for one map")
    ap.add_argument("--write-md", default=None, metavar="BEFORE_JSON",
                    help="write reference/AUDIT.md: before (this saved audit JSON) vs now")
    a = ap.parse_args()
    md = DraftModel.load(Path(a.model))
    res = audit_model(md)
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / f"audit_{a.label}.json").write_text(json.dumps(res, indent=1))
    if res.get("matchups"):
        (ROOT / "reference" / "MATCHUP_CHECKS.md").write_text(matchups_markdown(res["matchups"]))
    if a.write_md:
        before = json.loads(Path(a.write_md).read_text())
        md_txt = ["# Recommendation audit: before / after", "",
                  "_Generated by `audit.py --write-md`. Blind first pick (we pick first, no bans), top 10 "
                  "per map from the real draft engine. Flags: top-5 pick with < "
                  f"{MIN_GAMES_TOP5} games on the map; 0-game brawler with map strength > +{ZERO_GAME_PCT}% "
                  f"vs the typical pick; top 10 sharing < {MIN_OVERLAP} brawlers with the map's "
                  f"{TOP_PICKED} most picked. \"Picked-and-winning\" = among the 25 most picked with "
                  "raw win rate ≥ 50% (≥ 50 games)._", "",
                  "## Summary", "", "| | before (pre-fix model + engine) | after |", "|---|---|---|"]
        for k, lab in (("n_flagged", "maps flagged (of " + str(res["n_maps"]) + ")"), ("n_flags", "total flags"),
                       ("low_data_top5_total", "top-5 picks with < 30 games"),
                       ("mean_overlap_top15", "mean top-10 overlap with 15 most picked"),
                       ("mean_overlap_picked_winning", "mean top-10 overlap with picked-and-winning"),
                       ("mean_baseline_win_prob", "mean typical-pick baseline win%")):
            md_txt.append(f"| {lab} | {before.get(k)} | {res.get(k)} |")
        mb = before.get("matchups") or {}
        md_txt.append(f"| matchup checks passed | {str(mb.get('n_pass')) + '/' + str(mb.get('n')) if mb else 'n/a (pre-fix code)'} "
                      f"| {res['matchups']['n_pass']}/{res['matchups']['n']} |")
        md_txt += ["", "## Before", "", to_markdown(before, "Pre-fix model (models/model_legendary_before.pkl)"),
                   "## After", "", to_markdown(res, "Current model")]
        (ROOT / "reference" / "AUDIT.md").write_text("\n".join(md_txt))
    if a.json:
        print(json.dumps(res, indent=1))
        return
    print(summary_line(res))
    for x in res["maps"]:
        if a.map and x["map"] != a.map:
            continue
        print(f"\n{x['map']} ({x['mode']}, {x['games']} games) baseline {x['baseline_win_prob']}% "
              f"overlap {x['overlap_top15']}/10")
        for r in x["top"]:
            print(f"  {r['name']:<14} games={r['games']:<5} pr={r['pick_rate']:>5.1f}% wr={r['win_rate']} "
                  f"win={r['win_prob']} p25={r['win_prob_p25']} conf={r['confidence']} "
                  f"S={r['map_strength_centered']:+.1f} parts={r['strength_parts_pct']}")
        for f in x["flags"]:
            print("  FLAG", f)


if __name__ == "__main__":
    main()

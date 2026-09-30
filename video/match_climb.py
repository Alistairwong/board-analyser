"""Match the lit holds detected in a video against the climb database.

Each climb (and its left-right mirror) is compared with the detected holds by
position. The main score is the overlap between the two sets (Jaccard: holds
in both, divided by holds in either), so a perfect match scores 100% and a
missed or extra hold costs a little. Roles break ties; start and middle are
treated as one "hand" role because they look almost the same on camera.

Usage:
    python video/match_climb.py test1 [--top 5]

Reads data/detections/<name>.json (from detect_leds.py) and writes
data/detections/<name>_match.json.
"""
import argparse
import json
import math
import sqlite3
from pathlib import Path

from config import ANGLE, CLIMBS_PATH, DATA_DIR

DB_PATH = DATA_DIR / "tension.db"
DETECT_DIR = DATA_DIR / "detections"
HAND_ROLES = {"start", "middle"}
ROLE_MARGIN = 10          # hue units: a colour this close to a second kind of role is too unclear to name


def hue_distance(a, b):
    """Distance between two OpenCV hues (0-179), which wrap around at red."""
    d = abs(a - b) % 180
    return min(d, 180 - d)


def role_class(name):
    return "hand" if name in HAND_ROLES else (name or "?")


def load_climbs():
    data = json.loads(Path(CLIMBS_PATH).read_text())
    roles = {int(k): v["name"] for k, v in data["roles"].items()}
    climbs = []
    for c in data["climbs"]:
        holds = {(h["x"], h["y"]): role_class(roles.get(h["role"]))
                 for h in c["holds"]}
        climbs.append((c["uuid"], c["name"], holds))
    return climbs


def mirror(holds):
    return {(-x, y): r for (x, y), r in holds.items()}


def full_climb_holds(best, hole_by_pos):
    """The climb's own complete hold list from the database, for use once
    we're confident which climb this is. LED detection alone can miss a
    dim/occluded hold (e.g. a hand blocking it the whole clip); once the
    climb is identified, its official hold list is complete and correct.

    best: a result from match(), used for its uuid and mirrored flag.
    hole_by_pos: {(x, y): hole_id}, from calibrate.load_holes().
    """
    data = json.loads(Path(CLIMBS_PATH).read_text())
    roles = {int(k): v["name"] for k, v in data["roles"].items()}
    climb = next(c for c in data["climbs"] if c["uuid"] == best["uuid"])
    holds = []
    for h in climb["holds"]:
        x, y = (-h["x"], h["y"]) if best["mirrored"] else (h["x"], h["y"])
        hole_id = hole_by_pos.get((x, y))
        if hole_id is None:
            continue   # shouldn't happen for a climb that fits this board size
        holds.append({"hole_id": hole_id, "x": x, "y": y, "role_name": roles.get(h["role"])})
    return holds


def score(climb_holds, detected):
    common = climb_holds.keys() & detected.keys()
    union = len(climb_holds) + len(detected) - len(common)
    jaccard = len(common) / union if union else 0.0
    # Roles only break ties. A hold whose colour was unclear ("?") is left out of it, neither for nor against.
    roles_ok = sum(1 if climb_holds[p] == detected[p] else -1 for p in common if detected[p] != "?")
    return jaccard, len(common), roles_ok


def fmt(pos):
    return f"{pos[0]},{pos[1]}"


def climb_stats(uuids, angle=ANGLE):
    """Grade (V and Font), ascents and stars for climbs at the given angle."""
    uuids = list(uuids)
    if not uuids:
        return {}
    con = sqlite3.connect(DB_PATH)
    grades = dict(con.execute("SELECT difficulty, boulder_name FROM difficulty_grades"))
    placeholders = ",".join("?" * len(uuids))
    rows = con.execute(
        "SELECT climb_uuid, display_difficulty, benchmark_difficulty, "
        "ascensionist_count, quality_average FROM climb_stats "
        f"WHERE angle = ? AND climb_uuid IN ({placeholders})",
        [angle, *uuids],
    )
    stats = {}
    for uuid, diff, bench, ascents, quality in rows:
        name = grades.get(int(round(diff))) if diff is not None else None
        font, v = (name.split("/", 1) + [None])[:2] if name else (None, None)
        stats[uuid] = {
            "font": font, "v": v,
            "ascents": ascents,
            "stars": round(quality, 1) if quality is not None else None,
            "benchmark": bench is not None,
        }
    return stats


def grade_text(r):
    if not r.get("v"):
        return f"no ascents logged at {r.get('angle', ANGLE)}°"
    bench = ", benchmark" if r.get("benchmark") else ""
    stars = f"{r['stars']} stars" if r.get("stars") is not None else "no rating"
    return f"{r['v']} / {r['font']}, {r['ascents']} ascents, {stars}{bench}"


_HEIGHT_COUNTS = None
BOARD_HEIGHT = 144        # inches, y runs 0..144
HEIGHT_BINS = 10


def finish_odds(y, other):
    """How many finish holds there are per `other` hold ("hand" or "foot") at board height y, counted over every
    climb in the database. It's about 0.02 or less up to 100 in (a finish there is very unlikely) and rises to
    about 4 hand-holds' worth only in the top tenth (y >= 130); against foot holds finish wins from y = 100 up."""
    global _HEIGHT_COUNTS
    if _HEIGHT_COUNTS is None:
        counts = [{"finish": 0, "hand": 0, "foot": 0} for _ in range(HEIGHT_BINS)]
        data = json.loads(Path(CLIMBS_PATH).read_text())
        roles = {int(k): v["name"] for k, v in data["roles"].items()}
        for c in data["climbs"]:
            for h in c["holds"]:
                cls = role_class(roles.get(h["role"]))
                if cls in counts[0]:
                    counts[min(HEIGHT_BINS - 1, max(0, int(h["y"] / BOARD_HEIGHT * HEIGHT_BINS)))][cls] += 1
        _HEIGHT_COUNTS = counts
    c = _HEIGHT_COUNTS[min(HEIGHT_BINS - 1, max(0, int(y / BOARD_HEIGHT * HEIGHT_BINS)))]
    return (c["finish"] + 1) / (c[other] + 1)


_FINISH_PAIRS = None


def finish_pair_limit(pct=0.95):
    """How far apart (inches) the two finish holds of a climb can reasonably be: the `pct` point of the distances
    in every two-finish climb in the database (about 46 in at 0.95; the median pair is 16 in apart, the board is 88 wide)."""
    global _FINISH_PAIRS
    if _FINISH_PAIRS is None:
        data = json.loads(Path(CLIMBS_PATH).read_text())
        roles = {int(k): v["name"] for k, v in data["roles"].items()}
        dists = []
        for c in data["climbs"]:
            f = [h for h in c["holds"] if roles.get(h["role"]) == "finish"]
            if len(f) == 2:
                dists.append(math.dist((f[0]["x"], f[0]["y"]), (f[1]["x"], f[1]["y"])))
        _FINISH_PAIRS = sorted(dists)
    if not _FINISH_PAIRS:
        return float("inf")
    return _FINISH_PAIRS[min(len(_FINISH_PAIRS) - 1, int(pct * len(_FINISH_PAIRS)))]


def role_hues(roles):
    """{"hand": [hues], "foot": [...], "finish": [...]} from detect_leds.load_roles()."""
    out = {}
    for r in roles.values():
        out.setdefault(role_class(r["name"]), []).append(r["hue"])
    return out


def colour_agreement(climb_holds, lit, hues):
    """How well the LED colours fit a climb's roles: +1 for each lit hold (of the climb) whose colour clearly says the
    role the climb gives it, -1 where it clearly says another, 0 where the colour is too unclear to say."""
    score = 0
    for h in lit:
        cls = climb_holds.get((h["x"], h["y"]))
        if cls is None or h.get("hue") is None:
            continue
        near = sorted((min(hue_distance(h["hue"], x) for x in xs), c) for c, xs in hues.items())
        if len(near) > 1 and near[1][0] - near[0][0] < ROLE_MARGIN:
            continue
        score += 1 if near[0][1] == cls else -1
    return score


def break_ties(top, lit, hues):
    """Among candidates with the same position overlap, put the one whose roles fit the LED colours best first.
    This is the only place colours help pick a climb, and only after the positions alone couldn't decide.
    Sets roles_agree on the tied candidates to their colour agreement, so assess() sees a resolved tie."""
    if len(top) < 2:
        return top
    tied = [r for r in top if abs(r["jaccard"] - top[0]["jaccard"]) < 1e-9]
    if len(tied) < 2:
        return top
    holds_of = {u: h for u, _, h in get_climbs()}
    for r in tied:
        holds = holds_of[r["uuid"]]
        r["roles_agree"] = colour_agreement(mirror(holds) if r["mirrored"] else holds, lit, hues)
    tied.sort(key=lambda r: -r["roles_agree"])
    return tied + top[len(tied):]


_CLIMBS = None


def get_climbs():
    """Load the climb list once and reuse it (handy when matching many videos)."""
    global _CLIMBS
    if _CLIMBS is None:
        _CLIMBS = load_climbs()
    return _CLIMBS


def match(lit, top=5):
    """Rank climbs against a list of lit holds (from detect_leds). Best first."""
    detected = {(h["x"], h["y"]): role_class(h.get("role_name")) for h in lit}
    if not detected:
        return []

    results = []
    for uuid, name, holds in get_climbs():
        variants = [(False, holds)]
        mirrored = mirror(holds)
        if mirrored != holds:
            variants.append((True, mirrored))
        for is_mirror, hs in variants:
            jac, n_common, roles_ok = score(hs, detected)
            if n_common:
                results.append({
                    "uuid": uuid, "name": name, "mirrored": is_mirror,
                    "jaccard": jac, "matched": n_common, "climb_holds": len(hs),
                    "roles_agree": roles_ok,
                    "missing": sorted(fmt(p) for p in hs.keys() - detected.keys()),
                    "extra": sorted(fmt(p) for p in detected.keys() - hs.keys()),
                })

    results.sort(key=lambda r: (r["jaccard"], r["roles_agree"]), reverse=True)
    top = results[:top]

    stats = climb_stats({r["uuid"] for r in top})
    empty = {"font": None, "v": None, "ascents": 0, "stars": None, "benchmark": False}
    for r in top:
        r.update(stats.get(r["uuid"], empty))
        r["angle"] = ANGLE
    return top


def print_matches(top, n_detected):
    print(f"{'#':>2}  {'match':>6}  {'holds':>7}  {'grade':>11}  {'ascents':>7}  {'stars':>5}  climb")
    for i, r in enumerate(top, 1):
        tag = " (mirrored)" if r["mirrored"] else ""
        grade = f"{r['v']} / {r['font']}" if r.get("v") else "-"
        stars = f"{r['stars']}" if r.get("stars") is not None else "-"
        print(f"{i:>2}  {r['jaccard']:6.0%}  {r['matched']:>3}/{r['climb_holds']:<3}  "
              f"{grade:>11}  {r.get('ascents') or 0:>7}  {stars:>5}  {r['name']}{tag}")
    if top:
        best = top[0]
        runner_up = top[1]["jaccard"] if len(top) > 1 else 0.0
        print(f"\nBest: {best['name']}{' (mirrored)' if best['mirrored'] else ''}")
        print(f"  {grade_text(best)}")
        print(f"  margin over #2: {best['jaccard'] - runner_up:.0%}")
        if best["missing"]:
            print(f"  in the climb but not detected: {', '.join(best['missing'])}")
        if best["extra"]:
            print(f"  detected but not in the climb: {', '.join(best['extra'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="detection name, usually the video's file name without extension")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    det = json.loads((DETECT_DIR / f"{args.name}.json").read_text())
    if not det["lit"]:
        raise SystemExit("No lit holds in the detection file.")
    top = match(det["lit"], args.top)
    print(f"{len(det['lit'])} detected holds\n")
    print_matches(top, len(det["lit"]))

    out = DETECT_DIR / f"{args.name}_match.json"
    out.write_text(json.dumps({"detection": args.name, "top": top}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

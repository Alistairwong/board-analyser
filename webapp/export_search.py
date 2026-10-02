"""Export the climb database into one compact file for the search page.

Reads the climbs that fit a board (from load_climbs.py), adds setter, date,
grade, ascents and stars at the chosen angle, and writes
data/search/climbs_<board>_<angle>.json. search.py runs this automatically when the
database has changed, once per board and angle that someone opens.
"""
import json
import sqlite3

from config import BOARDS, DATA_DIR
from load_climbs import load_climbs

ROLE_CODES = {"start": 0, "middle": 1, "finish": 2, "foot": 3}


def db_path(key):
    return DATA_DIR / BOARDS[key]["db"]


def out_path(key, angle):
    return DATA_DIR / "search" / f"climbs_{key}_{angle}.json"


def board_name(con, b):
    try:
        layout = con.execute("SELECT name FROM layouts WHERE id = ?", (b["layout_id"],)).fetchone()
        size = con.execute("SELECT name FROM product_sizes WHERE id = ?", (b["size_id"],)).fetchone()
        return f"{layout[0]}, {size[0]}"
    except (sqlite3.Error, TypeError):
        return b["name"]


def size_edges(con, b):
    return con.execute("SELECT edge_left, edge_right, edge_bottom, edge_top FROM product_sizes WHERE id = ?",
                       (b["size_id"],)).fetchone()


def load_holes(con, b):
    """Holes on the board size: same rule as calibrate.py, without needing OpenCV."""
    left, right, bottom, top = size_edges(con, b)
    return con.execute(
        "SELECT DISTINCT h.id, h.name, h.x, h.y FROM holes h "
        "JOIN placements p ON p.hole_id = h.id "
        "WHERE p.layout_id = ? "
        "AND h.x > ? AND h.x < ? AND h.y > ? AND h.y < ?",
        (b["layout_id"], left, right, bottom, top),
    ).fetchall()


def led_positions(con, b):
    """hole id -> LED position on the board size (what the Bluetooth kit expects)."""
    try:
        return dict(con.execute(
            "SELECT hole_id, position FROM leds WHERE product_size_id = ?", (b["size_id"],)))
    except sqlite3.Error:
        return {}


def hole_sets(con, b):
    """hole id -> index of its hold set (e.g. wood or plastic), plus the set names."""
    try:
        rows = con.execute(
            "SELECT p.hole_id, s.name FROM placements p JOIN sets s ON s.id = p.set_id "
            "WHERE p.layout_id = ?", (b["layout_id"],)).fetchall()
    except sqlite3.Error:
        return {}, []
    names = sorted({n for _, n in rows if n})
    return {h: names.index(n) for h, n in rows if n}, names


def role_colours(con, b):
    """Role name -> the colour the board's app lights it in."""
    cols = {r[1] for r in con.execute("PRAGMA table_info(placement_roles)")}
    colour = "led_color" if "led_color" in cols else "screen_color"
    return {name: hexcol for name, hexcol in con.execute(
        f"SELECT name, {colour} FROM placement_roles WHERE product_id = ?", (b["product_id"],)) if hexcol}


def export(key="tension", angle=None):
    b = BOARDS[key]
    angle = angle or b["angle"]
    con = sqlite3.connect(db_path(key))
    holes = load_holes(con, b)
    index = {(x, y): i for i, (_, _, x, y) in enumerate(holes)}
    leds = led_positions(con, b)
    sets, set_names = hole_sets(con, b)

    climbs_in, roles, _, _ = load_climbs(con, b["layout_id"], b["size_id"], b["product_id"])
    role_names = {rid: r["name"] for rid, r in roles.items()}

    cols = {r[1] for r in con.execute("PRAGMA table_info(climbs)")}
    setter = "setter_username" if "setter_username" in cols else "NULL"
    created = "created_at" if "created_at" in cols else "NULL"
    meta = {u: (s, c) for u, s, c in con.execute(
        f"SELECT uuid, {setter}, {created} FROM climbs WHERE layout_id = ?", (b["layout_id"],))}

    stats = {u: (d, bm, a, q) for u, d, bm, a, q in con.execute(
        "SELECT climb_uuid, display_difficulty, benchmark_difficulty, ascensionist_count, "
        "quality_average FROM climb_stats WHERE angle = ?", (angle,))}
    grades = {int(d): n for d, n in con.execute(
        "SELECT difficulty, boulder_name FROM difficulty_grades") if n}

    climbs, skipped = [], 0
    for c in climbs_in:
        if b["only_climbed"] and c["uuid"] not in stats:
            continue
        packed = []
        for h in c["holds"]:
            i = index.get((h["x"], h["y"]))
            if i is None:
                break
            packed += [i, ROLE_CODES.get(role_names.get(h["role"]), 1)]
        else:
            s, when = meta.get(c["uuid"], (None, None))
            d, bm, a, q = stats.get(c["uuid"], (None, None, 0, None))
            climbs.append([
                c["uuid"], c["name"], s or "", (when or "")[:10],
                round(d, 2) if d is not None else None,
                1 if bm is not None else 0,
                a or 0,
                round(q, 2) if q is not None else None,
                packed,
            ])
            continue
        skipped += 1

    left, right, bottom, top = size_edges(con, b)
    out = out_path(key, angle)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "board": board_name(con, b),
        "key": key,
        "angle": angle,
        "boards": {k: {"name": v["name"], "angles": v["angles"]} for k, v in BOARDS.items()},
        "bounds": [left, right, bottom, top],
        "image": b["image"],
        "holes": [[x, y, leds.get(hid), sets.get(hid)] for hid, _, x, y in holes],
        "sets": set_names,
        "role_colours": role_colours(con, b),
        "grades": {str(k): v for k, v in grades.items()},
        "climbs": climbs,
    }, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"{key} {angle}°: exported {len(climbs)} climbs ({skipped} skipped) to {out}")
    lit = sum(1 for hid, *_ in holes if hid in leds)
    print(f"LED positions found for {lit} of {len(holes)} holes")


if __name__ == "__main__":
    import sys
    export(*(sys.argv[1:2] + [int(a) for a in sys.argv[2:3]]))

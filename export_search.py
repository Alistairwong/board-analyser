"""Export the climb database into one compact file for the search page.

Reads the climbs that fit your board (from load_climbs.py), adds setter, date,
grade, ascents and stars at your board angle, and writes data/search/climbs.json.
search.py runs this automatically when the database has changed.
"""
import json
import sqlite3
from pathlib import Path

from config import ANGLE, CLIMBS_PATH, LAYOUT_ID, PRODUCT_ID, PRODUCT_SIZE_ID

DB_PATH = Path("data/tension.db")
OUT = Path("data/search/climbs.json")
ROLE_CODES = {"start": 0, "middle": 1, "finish": 2, "foot": 3}


def board_name(con):
    try:
        layout = con.execute("SELECT name FROM layouts WHERE id = ?", (LAYOUT_ID,)).fetchone()
        size = con.execute("SELECT name FROM product_sizes WHERE id = ?", (PRODUCT_SIZE_ID,)).fetchone()
        return f"{layout[0]}, {size[0]}"
    except (sqlite3.Error, TypeError):
        return "Tension Board"


def load_holes(con):
    """Holes on your board size: same rule as calibrate.py, without needing OpenCV."""
    left, right, bottom, top = con.execute(
        "SELECT edge_left, edge_right, edge_bottom, edge_top FROM product_sizes WHERE id = ?",
        (PRODUCT_SIZE_ID,),
    ).fetchone()
    return con.execute(
        "SELECT DISTINCT h.id, h.name, h.x, h.y FROM holes h "
        "JOIN placements p ON p.hole_id = h.id "
        "WHERE p.layout_id = ? "
        "AND h.x > ? AND h.x < ? AND h.y > ? AND h.y < ?",
        (LAYOUT_ID, left, right, bottom, top),
    ).fetchall()


def led_positions(con):
    """hole id -> LED position on your board size (what the Bluetooth kit expects)."""
    try:
        return dict(con.execute(
            "SELECT hole_id, position FROM leds WHERE product_size_id = ?", (PRODUCT_SIZE_ID,)))
    except sqlite3.Error:
        return {}


def role_colours(con):
    """Role name -> the colour the Tension app lights it in."""
    cols = {r[1] for r in con.execute("PRAGMA table_info(placement_roles)")}
    colour = "led_color" if "led_color" in cols else "screen_color"
    return {name: hexcol for name, hexcol in con.execute(
        f"SELECT name, {colour} FROM placement_roles WHERE product_id = ?", (PRODUCT_ID,)) if hexcol}


def export():
    con = sqlite3.connect(DB_PATH)
    holes = load_holes(con)
    index = {(x, y): i for i, (_, _, x, y) in enumerate(holes)}
    leds = led_positions(con)

    data = json.loads(Path(CLIMBS_PATH).read_text())
    role_names = {int(k): v["name"] for k, v in data["roles"].items()}

    cols = {r[1] for r in con.execute("PRAGMA table_info(climbs)")}
    setter = "setter_username" if "setter_username" in cols else "NULL"
    created = "created_at" if "created_at" in cols else "NULL"
    meta = {u: (s, c) for u, s, c in con.execute(
        f"SELECT uuid, {setter}, {created} FROM climbs WHERE layout_id = ?", (LAYOUT_ID,))}

    stats = {u: (d, b, a, q) for u, d, b, a, q in con.execute(
        "SELECT climb_uuid, display_difficulty, benchmark_difficulty, ascensionist_count, "
        "quality_average FROM climb_stats WHERE angle = ?", (ANGLE,))}
    grades = {int(d): n for d, n in con.execute(
        "SELECT difficulty, boulder_name FROM difficulty_grades") if n}

    climbs, skipped = [], 0
    for c in data["climbs"]:
        packed = []
        for h in c["holds"]:
            i = index.get((h["x"], h["y"]))
            if i is None:
                break
            packed += [i, ROLE_CODES.get(role_names.get(h["role"]), 1)]
        else:
            s, when = meta.get(c["uuid"], (None, None))
            d, b, a, q = stats.get(c["uuid"], (None, None, 0, None))
            climbs.append([
                c["uuid"], c["name"], s or "", (when or "")[:10],
                round(d, 2) if d is not None else None,
                1 if b is not None else 0,
                a or 0,
                round(q, 2) if q is not None else None,
                packed,
            ])
            continue
        skipped += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "board": board_name(con),
        "angle": ANGLE,
        "holes": [[x, y, leds.get(hid)] for hid, _, x, y in holes],
        "role_colours": role_colours(con),
        "grades": {str(k): v for k, v in grades.items()},
        "climbs": climbs,
    }, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(climbs)} climbs ({skipped} skipped) to {OUT}")
    lit = sum(1 for hid, *_ in holes if hid in leds)
    print(f"LED positions found for {lit} of {len(holes)} holes")


if __name__ == "__main__":
    export()

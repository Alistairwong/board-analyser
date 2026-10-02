"""Load Tension Board 2 climbs for one layout and board size as hold sets.

Each climb becomes a list of holds with board coordinates (inches, x=0 at the
centre line) and a role (start / hand / foot / finish). This is the reference
data the video matcher compares detected LEDs against.
"""
import json
import re
import sqlite3

from config import CLIMBS_PATH, DATA_DIR, LAYOUT_ID, PRODUCT_ID, PRODUCT_SIZE_ID

DB_PATH = DATA_DIR / "tension.db"

# A climb's "frames" string looks like p123r5p456r6... (placement id, role id)
TOKEN = re.compile(r"p(\d+)r(\d+)")


def load_climbs(con, layout_id=LAYOUT_ID, size_id=PRODUCT_SIZE_ID, product_id=PRODUCT_ID):
    cur = con.cursor()

    left, right, bottom, top = cur.execute(
        "SELECT edge_left, edge_right, edge_bottom, edge_top "
        "FROM product_sizes WHERE id = ?",
        (size_id,),
    ).fetchone()

    # placement id -> hole position on the board
    placements = {
        pid: (x, y)
        for pid, x, y in cur.execute(
            "SELECT p.id, h.x, h.y FROM placements p "
            "JOIN holes h ON h.id = p.hole_id "
            "WHERE p.layout_id = ?",
            (layout_id,),
        )
    }

    # role id -> name and display colour
    roles = {
        rid: {"name": name, "colour": colour}
        for rid, name, colour in cur.execute(
            "SELECT id, name, screen_color FROM placement_roles "
            "WHERE product_id = ?",
            (product_id,),
        )
    }

    # Only listed, published climbs whose bounding box fits this board size
    rows = cur.execute(
        "SELECT uuid, name, frames FROM climbs "
        "WHERE layout_id = ? AND is_listed = 1 AND is_draft = 0 "
        "AND edge_left > ? AND edge_right < ? "
        "AND edge_bottom > ? AND edge_top < ?",
        (layout_id, left, right, bottom, top),
    ).fetchall()

    climbs, skipped = [], 0
    for uuid, name, frames in rows:
        holds = []
        for pid, rid in TOKEN.findall(frames or ""):
            pid, rid = int(pid), int(rid)
            if pid not in placements:
                holds = None  # references a hold outside this layout
                break
            x, y = placements[pid]
            holds.append({"placement": pid, "x": x, "y": y, "role": rid})
        if not holds:
            skipped += 1
            continue
        climbs.append({"uuid": uuid, "name": name, "holds": holds})

    return climbs, roles, len(placements), skipped


def main():
    con = sqlite3.connect(DB_PATH)
    climbs, roles, n_placements, skipped = load_climbs(con)

    print(f"Hold placements on layout: {n_placements}")
    print(f"Climbs loaded: {len(climbs)}  (skipped {skipped})")
    print("\nRoles:")
    for rid, r in sorted(roles.items()):
        print(f"  {rid}: {r['name']} ({r['colour']})")

    if climbs:
        c = climbs[0]
        print(f"\nExample: {c['name']} ({len(c['holds'])} holds)")
        for h in c["holds"]:
            print(f"  placement {h['placement']}: x={h['x']}, y={h['y']}, "
                  f"role={roles.get(h['role'], {}).get('name', h['role'])}")

    CLIMBS_PATH.write_text(json.dumps({"roles": roles, "climbs": climbs}))
    print(f"\nSaved to {CLIMBS_PATH}")


if __name__ == "__main__":
    main()

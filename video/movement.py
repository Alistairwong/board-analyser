"""Movement analysis from a body-landmark track: which holds each limb used,
when moves happened, timing/effort metrics, and rule-based technique flags.

Pure logic over plain data structures (lists of dicts), so it's fully
testable with synthetic landmark tracks -- no video or MediaPipe needed.
Consumes the track produced by pose.to_board_inches() and the climb's own
hold list (detect_leds.detect()'s "lit", i.e. only this climb's ~10-20
holds, not the whole board).

Single side-on camera, so depth is unknown: "hips_out" is a rough 2D
approximation (hip x vs. the supporting holds' x), not a real measurement.
"""
from statistics import median

HAND_LIMBS = ("left_wrist", "right_wrist")
FOOT_LIMBS = ("left_foot", "right_foot")   # toe (foot_index), not the ankle -- see pose.py
LIMBS = HAND_LIMBS + FOOT_LIMBS

HOLD_TOLERANCE = 4.0       # inches: a limb within this of a hold counts as "on" it
MIN_RUN = 3                # samples an assignment must hold to count as stable (debounce)
HESITATION_FACTOR = 2.0    # a pause this many times the median gap counts as hesitation
DYNAMIC_HIP_SPEED = 20.0   # inches/second of hip vertical speed that counts as a dynamic move
HIP_WINDOW = 0.4           # seconds either side of a hand move to look at hip speed
HIPS_OUT_OFFSET = 10.0     # inches of hip/support x-offset that counts as "hips out"


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def nearest_hold(point, holds, tolerance=HOLD_TOLERANCE):
    """The closest hold to a board-inch point, or None if nothing's close enough."""
    best, best_d = None, tolerance
    for h in holds:
        d = _dist(point, (h["x"], h["y"]))
        if d <= best_d:
            best, best_d = h, d
    return best["hole_id"] if best else None


def assign_holds(track, holds, tolerance=HOLD_TOLERANCE):
    """Per frame, per limb: which hold (if any) that limb is on."""
    assignments = []
    for f in track:
        assign = {}
        for limb in LIMBS:
            pt = f["points"].get(limb)
            assign[limb] = nearest_hold(pt, holds, tolerance) if pt else None
        assignments.append({"t": f["t"], "assign": assign})
    return assignments


def _stable_runs(assignments, limb, min_run=MIN_RUN):
    """Collapse one limb's per-frame assignment into stable (t, hole_id) runs.

    A run must last at least min_run consecutive samples to count -- shorter
    blips (tracking jitter, a hand passing over a hold) are dropped by
    merging them into whichever run they interrupt.
    """
    raw = [(f["t"], f["assign"][limb]) for f in assignments]
    runs = []   # each: [start_t, hole_id, length]
    for t, hole_id in raw:
        if runs and runs[-1][1] == hole_id:
            runs[-1][2] += 1
        else:
            runs.append([t, hole_id, 1])
    stable = [r for r in runs if r[2] >= min_run]
    return [(t, hole_id) for t, hole_id, _ in stable]


def detect_moves(assignments, min_run=MIN_RUN):
    """All moves across every limb, in chronological order.

    A move is a limb's stable hold assignment changing to a different hold
    (including to/from "off the wall", i.e. None). The first stable
    assignment for a limb is its starting position, not a move.
    """
    moves = []
    for limb in LIMBS:
        runs = _stable_runs(assignments, limb, min_run)
        for i in range(1, len(runs)):
            t, to_hold = runs[i]
            _, from_hold = runs[i - 1]
            if to_hold == from_hold:
                continue
            moves.append({"limb": limb, "from_hold": from_hold, "to_hold": to_hold, "t": t})
    moves.sort(key=lambda m: m["t"])
    for i, m in enumerate(moves):
        m["duration_since_prev"] = round(m["t"] - moves[i - 1]["t"], 2) if i else None
    return moves


def compute_metrics(moves, track):
    """Timing/effort metrics for the attempt."""
    if not track:
        return {"duration_s": 0.0, "move_count": 0, "moves_per_min": 0.0,
                "rest_time_s": 0.0, "longest_pause_s": 0.0, "longest_pause_hold": None}
    duration = track[-1]["t"] - track[0]["t"]
    gaps = [m["duration_since_prev"] for m in moves if m["duration_since_prev"] is not None]
    typical = median(gaps) if gaps else 0.0
    rest_time = sum(g for g in gaps if g > typical * 1.5)
    if gaps:
        i_longest = max(range(len(gaps)), key=lambda i: gaps[i])
        longest_pause = gaps[i_longest]
        longest_pause_hold = moves[i_longest]["from_hold"]   # where they were resting
    else:
        longest_pause, longest_pause_hold = 0.0, None
    return {
        "duration_s": round(duration, 2),
        "move_count": len(moves),
        "moves_per_min": round(len(moves) / duration * 60, 1) if duration > 0 else 0.0,
        "rest_time_s": round(rest_time, 2),
        "longest_pause_s": round(longest_pause, 2),
        "longest_pause_hold": longest_pause_hold,
    }


def _hip_centre(points):
    hips = [points[n] for n in ("left_hip", "right_hip") if n in points]
    if not hips:
        return None
    return (sum(p[0] for p in hips) / len(hips), sum(p[1] for p in hips) / len(hips))


def _hip_vertical_speed(track, t, window=HIP_WINDOW):
    """Max abs hip vertical speed (inches/s) in a window around time t."""
    pts = sorted((f["t"], _hip_centre(f["points"])) for f in track if abs(f["t"] - t) <= window)
    pts = [(t2, c) for t2, c in pts if c is not None]
    speeds = []
    for (t1, c1), (t2, c2) in zip(pts, pts[1:]):
        dt = t2 - t1
        if dt > 0:
            speeds.append(abs(c2[1] - c1[1]) / dt)
    return max(speeds) if speeds else 0.0


def classify_styles(moves, track, dynamic_speed=DYNAMIC_HIP_SPEED):
    """Tag hand moves "static" or "dynamic" from hip vertical speed around the move."""
    for m in moves:
        if m["limb"] in HAND_LIMBS:
            speed = _hip_vertical_speed(track, m["t"])
            m["style"] = "dynamic" if speed >= dynamic_speed else "static"
        else:
            m["style"] = None
    return moves


def flag_hesitations(moves, factor=HESITATION_FACTOR):
    """Mark moves preceded by an unusually long pause, relative to this attempt's pace."""
    gaps = [m["duration_since_prev"] for m in moves if m["duration_since_prev"] is not None]
    typical = median(gaps) if gaps else 0.0
    for m in moves:
        gap = m["duration_since_prev"]
        m["hesitated"] = bool(gap is not None and typical > 0 and gap > typical * factor)
    return moves


def flag_hips(moves, assignments, track, holds_by_id, offset_threshold=HIPS_OUT_OFFSET):
    """Approximate hips-in/out: hip x vs. the other limbs' supporting holds.

    A single camera can't measure true 3D hip position, so this only
    compares hold x-positions -- a rough signal, not a biomechanical one.
    """
    idx_by_t = {a["t"]: i for i, a in enumerate(assignments)}
    for m in moves:
        i = idx_by_t.get(m["t"])
        hip = _hip_centre(track[i]["points"]) if i is not None else None
        offset = None
        if i is not None and hip is not None:
            support_xs = [holds_by_id[hid]["x"]
                          for limb, hid in assignments[i]["assign"].items()
                          if limb != m["limb"] and hid in holds_by_id]
            if support_xs:
                offset = hip[0] - sum(support_xs) / len(support_xs)
        m["hips_offset"] = round(offset, 1) if offset is not None else None
        m["hips_out"] = bool(offset is not None and abs(offset) > offset_threshold)
    return moves


def attempt_outcome(moves, holds):
    """"topped" if a hand ended on a finish hold, else "unknown"."""
    hold_role = {h["hole_id"]: h.get("role_name") for h in holds}
    final = {m["limb"]: m["to_hold"] for m in moves if m["limb"] in HAND_LIMBS}
    if any(hold_role.get(h) == "finish" for h in final.values() if h is not None):
        return "topped"
    return "unknown"


def build_flags(moves):
    """Human-readable one-line flags for the notable moves."""
    flags = []
    for m in moves:
        limb = m["limb"].replace("_", " ")
        if m.get("hesitated"):
            flags.append(f"hesitated for {m['duration_since_prev']:.1f}s before moving "
                         f"{limb} to hold {m['to_hold']}")
        if m.get("style") == "dynamic":
            flags.append(f"dynamic move: {limb} to hold {m['to_hold']}")
        if m.get("hips_out"):
            flags.append(f"hips out ({m['hips_offset']:+.0f}in) moving {limb} to hold {m['to_hold']}")
    return flags


def analyse(track, holds, tolerance=HOLD_TOLERANCE, min_run=MIN_RUN):
    """Run the full movement analysis on one attempt. Returns a dict ready to save."""
    holds_by_id = {h["hole_id"]: h for h in holds}
    assignments = assign_holds(track, holds, tolerance)
    moves = detect_moves(assignments, min_run)
    classify_styles(moves, track)
    flag_hesitations(moves)
    flag_hips(moves, assignments, track, holds_by_id)
    return {
        "attempt": {
            "start_s": track[0]["t"] if track else 0.0,
            "end_s": track[-1]["t"] if track else 0.0,
            "outcome": attempt_outcome(moves, holds),
        },
        "moves": moves,
        "metrics": compute_metrics(moves, track),
        "flags": build_flags(moves),
    }

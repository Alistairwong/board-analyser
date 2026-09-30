"""Photo/video -> which climb is lit on the board, with the climb's holds highlighted.

Runs in Docker (see docker-compose.yml). Reuses video/'s own recognition code (the
Dockerfile copies it in beside this file). Upload a photo or a video; the app aligns
it to a hand-made reference calibration, finds the lit holds, matches them against
the climb database and returns an annotated photo plus the climb's details.

A reference calibration is made once from the "Calibrate" tab (click 4 corner holes
on a photo of the board); after that any photo/video from any angle is aligned to it
automatically.
"""
import argparse
import json
import math
import re
import sqlite3
import subprocess
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from autocalibrate import auto_calibrate, board_mask, manual_references, reference_frame
from calibrate import CALIB_DIR, DB_PATH, corner_holes, grab_frame, load_holes
from config import DATA_DIR
from detect_leds import assign_lit_roles, detect, hole_patches, load_roles
from empty import find_empty, read_frames
from match_climb import break_ties, full_climb_holds, grade_text, match, role_class, role_hues
from movement import analyse as analyse_movement
import verify
from recognise import GOOD, assess

HERE = Path(__file__).resolve().parent
OUT = DATA_DIR / "recogniser"
MAX_UPLOAD = 1024 * 1024 * 1024
VIDEO_FPS = 3                     # frames per second analysed from a video
POSE_FPS = 5                      # frames per second the body checks see (the scan is a bit denser and is thinned to this)
MAX_SCANS = 600                   # the empty-board scan looks at every ~3rd frame, but at most this many frames in all
SHOT = 1000                       # per-check pictures are shrunk to this many pixels
POSE_SIDE = 640                   # frames are shrunk to this many pixels for the scan (MediaPipe resizes anyway)
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
VIDEO_EXT = {"video/mp4": "mp4", "video/quicktime": "mov", "video/webm": "webm"}

JOBS = {}                         # id -> {"status", "progress", "step", "result", "error"}
WORK = threading.Lock()           # one recognition at a time: OpenCV/SIFT is memory hungry


# ---- recognition ---------------------------------------------------------
def still_to_clip(img, clip):
    """Everything downstream reads videos, so a still image becomes a 2 s clip of that one frame."""
    png = clip.with_suffix(".png")
    cv2.imwrite(str(png), img)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png), "-t", "2", "-r", "5",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                    "-crf", "10", str(clip)], check=True)   # crf 10: keep LED colours intact
    png.unlink()


def photo_to_clip(photo_bytes, clip):
    """Decoding with OpenCV applies the phone's EXIF rotation, as the browser does."""
    img = cv2.imdecode(np.frombuffer(photo_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("That image couldn't be read.")
    still_to_clip(img, clip)


def clean_plate(clip, n=15, span=(0.0, 1.0)):
    """The board with the climber removed: the per-pixel median of n frames spread across the clip.
    The camera is fixed, so anything that moves (the climber) drops out and the board, the LEDs and
    every hold stay. span: only use this fraction of the video, e.g. (0, 0.5) for the first half. ponytail: fails if the climber covers the same pixels for over half the clip,
    and needs a fixed camera; a handheld clip would need the frames aligned first."""
    cap = cv2.VideoCapture(str(clip))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    first = int(span[0] * total)
    last = max(first, int(span[1] * total) - 1)
    want = set(np.linspace(first, last, min(n, last - first + 1)).astype(int).tolist())
    frames, i = [], 0
    while len(want) > len(frames) and cap.grab():
        if i in want:
            ok, f = cap.retrieve()
            if ok:
                frames.append(f)
        i += 1
    cap.release()
    if not frames:
        raise ValueError("Couldn't read any frames from the video.")
    if len(frames) == 1:
        return frames[0]
    out = np.empty_like(frames[0])
    for y in range(0, out.shape[0], 256):          # in bands, so the stack copy stays small
        out[y:y + 256] = np.median(np.stack([f[y:y + 256] for f in frames]), axis=0).astype(np.uint8)
    return out


def lit_holds(det):
    """A detection's lit holds in the shape draw() takes."""
    return [{"hole_id": h["hole_id"], "x": h["x"], "y": h["y"], "role": h.get("role_name"), "seen": True,
             "confidence": strength(h["score"], det["threshold"])} for h in det["lit"]]


def save_image(name, frame):
    """A picture for a check, shrunk to keep the page light. Returns its URL."""
    scale = SHOT / max(frame.shape[:2])
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    cv2.imwrite(str(OUT / f"{name}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return f"out/{name}.jpg"


def read_image(jid, tag, img, roles):
    """Read a picture on its own: line it up, find its lit holds, name the best climb, and draw what it found.
    Returns {"top": (uuid, mirrored, jaccard, name) | None, "lit": n, "frame": annotated picture}, or None if it won't line up."""
    clip = OUT / f"{jid}_{tag}.mp4"
    still_to_clip(img, clip)
    try:
        cal, calib = align(clip, f"auto-{jid}-{tag}", None)
    except ValueError:
        return None
    det = detect(clip, cal, verbose=False, images=False, assign_roles=False)
    top = match(det["lit"], top=1) if det["lit"] else []
    assign_lit_roles(det["lit"], roles)
    return {"top": (top[0]["uuid"], top[0]["mirrored"], top[0]["jaccard"], top[0]["name"]) if top else None,
            "lit": len(det["lit"]), "frame": draw(img.copy(), calib, lit_holds(det), [], roles)}


SHOW_CHECK_IMAGES = False   # set per job in recognise(); jobs run one at a time (WORK lock)


def reading_image(jid, tag, reading, label):
    """The [{"url", "caption"}] entry for a read_image() result."""
    if not reading or not SHOW_CHECK_IMAGES:
        return []
    said = f"{reading['top'][3]} ({reading['top'][2]:.0%})" if reading["top"] else "no climb"
    return [{"url": save_image(f"{jid}-{tag}", reading["frame"]), "caption": f"{label}: {reading['lit']} lit holds, best match {said}"}]


def body_image(jid, tag, clip, fps, frame_info, calib, holds, roles, caption):
    """A video frame with the recognised climb's holds circled and the tracked hands/feet marked."""
    if not SHOW_CHECK_IMAGES:
        return []
    idx = round(frame_info["t"] * fps)
    got = read_frames(clip, [idx])
    if idx not in got:
        return []
    frame = draw(got[idx], calib, [dict(h, role=h["role_name"], seen=True, confidence=0.0) for h in holds], [], roles, labels=False)
    r = max(6, min(frame.shape[:2]) // 90)
    for name, (x, y, vis) in frame_info["landmarks"].items():
        if vis >= 0.5 and name.endswith(("wrist", "foot")):
            colour = (0, 255, 255) if name.endswith("wrist") else (255, 255, 0)
            cv2.circle(frame, (int(x), int(y)), r, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), r - 2, colour, -1, cv2.LINE_AA)
    return [{"url": save_image(f"{jid}-{tag}", frame), "caption": caption}]


def judge(frames, H, holds):
    """Did the climber top out? Uses video/movement.py: "topped" means a hand ended on a finish hold.
    A miss can also be the pose tracker losing the climber, so the answer says how much it could track."""
    from pose import to_board_inches               # imports MediaPipe, so only when a video needs it
    track = to_board_inches(H, frames)
    tracked = sum(1 for f in track if any(k.endswith("wrist") for k in f["points"]))
    if tracked < 3:
        return {"outcome": "untracked", "reason": "Couldn't find a climber in the video.", "tracked_frames": tracked}
    result = analyse_movement(track, holds)
    m = result["metrics"]
    return {"outcome": "topped" if result["attempt"]["outcome"] == "topped" else "not topped",
            "reason": "", "tracked_frames": tracked, "frames": len(track),
            "moves": m["move_count"], "duration_s": round(m["duration_s"], 1)}


def climb_details(best):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT setter_username, description FROM climbs WHERE upper(uuid) = upper(?)",
                      (best["uuid"],)).fetchone()
    setter, description = row if row else (None, "")
    return {"name": best["name"], "uuid": best["uuid"], "setter": setter, "description": description or "",
            "grade_v": best.get("v"), "grade_font": best.get("font"), "ascents": best.get("ascents") or 0,
            "stars": best.get("stars"), "benchmark": bool(best.get("benchmark")),
            "angle": best.get("angle"), "mirrored": best["mirrored"], "summary": grade_text(best)}


def strength(score, threshold):
    """0-1 signal strength: 0 at the lit/unlit cut-off, 1 at three times it.
    This is how clearly the LED shows, not a probability."""
    return float(min(1.0, max(0.0, (score / threshold - 1) / 2))) if threshold else 0.0


def draw(frame, calib, holds, extras, roles, labels=True):
    """Circle the holds on the picture (role colours; labelled with confidence if `labels`). Returns the picture."""
    H = np.array(calib["homography"], dtype=np.float64)
    all_holes = load_holes(sqlite3.connect(DB_PATH))
    centres, radii, _ = hole_patches(H, all_holes, frame.shape)
    index = {h[0]: i for i, h in enumerate(all_holes)}
    by_name = {r["name"]: r["bgr"] for r in roles.values()}
    thick = max(2, int(min(frame.shape[:2]) / 400))
    font = max(0.5, min(frame.shape[:2]) / 1400)

    def label(c, text, colour):
        org = (c[0] + 4, c[1] - 4)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font, colour, thick, cv2.LINE_AA)

    for h in extras:                                  # lit, but not part of the recognised climb
        i = index.get(h["hole_id"])
        if i is not None:
            c = (int(centres[i][0]), int(centres[i][1]))
            cv2.circle(frame, c, int(radii[i] * 1.3), (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)
    for h in holds:
        i = index.get(h["hole_id"])
        if i is None:
            continue
        colour = by_name.get(h["role"], (255, 0, 255))
        c = (int(centres[i][0]), int(centres[i][1]))
        r = int(radii[i] * 1.7)
        cv2.circle(frame, c, r, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.circle(frame, c, r, colour, thick if h["seen"] else max(1, thick - 1), cv2.LINE_AA)
        if labels and h["seen"]:
            label(c, f"{round(h['confidence'] * 100)}%", colour)
    return frame


def render(frame, calib, holds, extras, climb, verdict, jid, roles, outcome=None):
    """The main picture (an empty-board frame, or the photo): the climb's holds circled, with a title bar."""
    draw(frame, calib, holds, extras, roles)
    thick = max(2, int(min(frame.shape[:2]) / 400))
    font = max(0.5, min(frame.shape[:2]) / 1400)
    if climb:
        title = f"{climb['name']}  {climb['grade_v'] or '?'}" + ("" if verdict in GOOD else f"  ({verdict})")
        if outcome and outcome["outcome"] in ("topped", "not topped"):
            title += "  - TOPPED" if outcome["outcome"] == "topped" else "  - not topped"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], int(48 * font * 2)), (0, 0, 0), -1)
        cv2.putText(frame, title, (12, int(34 * font * 2 * 0.75)), cv2.FONT_HERSHEY_SIMPLEX, font * 1.6,
                    (255, 255, 255), thick, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"{jid}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])


def align(clip, name, progress):
    """Line a clip up with the reference calibration. Returns (name, calibration dict), or raises a readable error."""
    info = {}
    found = auto_calibrate(clip, name=name, verbose=False, progress=progress, info=info)
    if not found:
        if not manual_references():
            raise ValueError("There's no reference calibration yet. Make one from the Calibrate tab first.")
        raise ValueError(f"Couldn't line this up with the board (only {info.get('features', 0)} matching features, need 40). "
                         "Try a sharper, brighter photo with the whole board in view.")
    return found, json.loads((CALIB_DIR / f"{found}.json").read_text())


def reference_for(calib, shape):
    """What empty.find_empty needs to compare video frames with the calibration photo, or None if that isn't available."""
    try:
        ref_calib, ref_img = reference_frame(calib["reference"])
        H_vid = np.array(calib["homography"], dtype=np.float64)
        H_ref = np.array(ref_calib["homography"], dtype=np.float64)
        holes = load_holes(sqlite3.connect(DB_PATH))
        mask = board_mask(ref_calib, ref_img.shape)
        centres, radii, _ = hole_patches(H_ref, holes, ref_img.shape)
        for (cx, cy), r in zip(centres, radii):        # an LED may be lit at any hole, so don't compare there
            cv2.circle(mask, (int(cx), int(cy)), max(3, int(r * 0.6)), 0, -1)   # r is 1.5 in, so this is ~0.9 in
        return {"image": ref_img, "M": H_ref @ np.linalg.inv(H_vid), "mask": mask}
    except (KeyError, FileNotFoundError, np.linalg.LinAlgError, cv2.error, SystemExit):
        return None


def video_info(clip):
    cap = cv2.VideoCapture(str(clip))
    fps, total = cap.get(cv2.CAP_PROP_FPS) or 30, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()
    return fps, total


def recognise(jid, clip, is_photo):
    job = JOBS[jid]

    def step(text, lo, hi):
        job["step"] = text
        return lambda f: job.__setitem__("progress", round(lo + (hi - lo) * f, 3))

    roles = load_roles()
    cal_name, calib = align(clip, f"auto-{jid}", step("Aligning to your calibration", 0, 0.1))

    # Which picture the lit holds are read from: for a video, an empty-board frame if the video has one.
    frames = plate = None
    fps = 30
    if is_photo:
        det = detect(clip, cal_name, fps=VIDEO_FPS, verbose=False, images=False, assign_roles=False, progress=step("Finding the lit holds", 0.1, 0.6))
        base, source = grab_frame(clip, calib["time"]), {"kind": "photo", "note": "Read from your photo.", "label": "Your photo"}
    else:
        from pose import estimate_landmarks
        fps, total = video_info(clip)
        frames = estimate_landmarks(clip, progress=step("Watching for the climber", 0.1, 0.5), step_frames=max(3, math.ceil(total / MAX_SCANS)),
                                    max_side=POSE_SIDE, keep_all=True)
        job["step"], job["progress"] = "Looking for a moment with nobody on the board", 0.52
        plate = clean_plate(clip)
        empty = find_empty(clip, frames, board_mask(calib, plate.shape), plate, fps, reference_for(calib, plate.shape))
        det, why = None, "Nobody ever left the board, so the lit holds were read from the whole video."
        if empty:
            eclip = OUT / f"{jid}_empty.mp4"
            still_to_clip(empty["image"], eclip)
            try:
                ecal, ecalib = align(eclip, f"auto-{jid}-empty", step("Aligning the empty board", 0.54, 0.58))
                edet = detect(eclip, ecal, verbose=False, images=False, assign_roles=False)
            except ValueError:                        # the empty frame wouldn't line up: use the whole video
                edet, why = None, "The empty-board frame wouldn't line up, so the lit holds were read from the whole video."
            if edet and edet["lit"]:
                det, calib, base = edet, ecalib, empty["image"]
                source = {"kind": "empty", "moments": empty["moments"], "at": empty["times"], "label": "Empty-board frame",
                          "note": f"Read from an empty-board frame ({empty['moments']} empty moment{'s' if empty['moments'] > 1 else ''} found, "
                                  f"e.g. at {empty['times'][0]} s; checked for anything in the way against {empty['checked_against']})."}
            elif edet is not None:
                why = "An empty-board frame showed no lit holds, so the lit holds were read from the whole video."
        if det is None:
            det = detect(clip, cal_name, fps=VIDEO_FPS, verbose=False, images=False, assign_roles=False, progress=step("Reading the whole video", 0.58, 0.68))
            base, source = plate, {"kind": "whole", "note": why, "label": "Median picture of the whole video"}
    job["step"], job["progress"] = "Matching against the climb database", 0.7
    # Lit or not first, then the climb by position alone, and only then roles: colours break a tie between climbs
    # with the same holds, and name the roles of holds no climb explains.
    lit = det["lit"]
    top = match(lit, top=5) if lit else []
    top = break_ties(top, lit, role_hues(roles))
    verdict, ties = assess(top)
    global SHOW_CHECK_IMAGES
    SHOW_CHECK_IMAGES = verdict not in GOOD       # evidence pictures only when not sure
    best = top[0] if top else None
    assign_lit_roles(lit, roles)

    thr = det["threshold"]
    lit_by_id = {h["hole_id"]: h for h in lit}
    if best and verdict in GOOD:
        wanted = full_climb_holds(best, {(x, y): hid for hid, _, x, y in load_holes(sqlite3.connect(DB_PATH))})
    else:
        wanted = [{"hole_id": h["hole_id"], "x": h["x"], "y": h["y"], "role_name": h.get("role_name")} for h in lit]
    holds = []
    for w in wanted:
        seen = lit_by_id.get(w["hole_id"])
        # A hold whose colour was too unclear to name took its role from the matched climb.
        observed = seen.get("role_name") if seen else None
        if not seen or not w["role_name"]:
            role_source = None
        elif observed is None:
            role_source = "climb"
        else:
            role_source = "colour" if role_class(observed) == role_class(w["role_name"]) else "conflict"
        holds.append({"hole_id": w["hole_id"], "x": w["x"], "y": w["y"], "role": w["role_name"],
                      "role_source": role_source, "colour_looked_like": (observed or (seen or {}).get("role_guess")) if role_source in ("climb", "conflict") else None,
                      "seen": bool(seen), "score": seen["score"] if seen else 0,
                      "confidence": strength(seen["score"], thr) if seen else 0.0})
    holds.sort(key=lambda h: (-h["y"], h["x"]))
    in_climb = {h["hole_id"] for h in holds}
    extras = [h for h in lit if h["hole_id"] not in in_climb]
    climb = climb_details(best) if best else None

    # Every check shows the picture it looked at. Lit-hole checks read their own picture from scratch.
    job["step"], job["progress"] = "Checking the answer", 0.72
    led = verify.led_check(best, verdict, ties)
    led["images"] = reading_image(jid, "c1", {"top": (best["uuid"], best["mirrored"], best["jaccard"], best["name"]) if best else None,
                                               "lit": len(lit), "frame": draw(base.copy(), calib, lit_holds(det), [], roles)}, source["label"])
    outcome, checks = None, [led]
    if not is_photo:
        H = np.array(calib["homography"], dtype=np.float64)
        if source["kind"] == "empty":
            reading = read_image(jid, "c2", plate, roles)
            c2 = verify.empty_vs_whole(reading["top"] if reading else None, best) if best else verify.check(verify.EMPTY_NAME, verify.NA, "No climb to compare.")
            c2["images"] = reading_image(jid, "c2", reading, "Median picture of the whole video")
        else:
            readings = [read_image(jid, f"c2{k}", clean_plate(clip, span=span), roles) for k, span in enumerate(((0.0, 0.5), (0.5, 1.0)))]
            halves = [r["top"] if r else None for r in readings] if best else None
            c2 = verify.split_half(halves, best) if halves else verify.check(verify.SPLIT_NAME, verify.NA, "No climb to compare.")
            c2["images"] = [im for r, label, k in zip(readings, ("First half", "Second half"), range(2)) for im in reading_image(jid, f"c2{k}", r, f"{label} (median picture)")]
        checks.append(c2)
        if verdict in GOOD:
            from pose import to_board_inches
            span = frames[-1]["t"] - frames[0]["t"] if len(frames) > 1 else 0
            thin = max(1, round(((len(frames) - 1) / span if span else POSE_FPS) / POSE_FPS))
            body = frames[::thin]                     # the body checks were tuned for ~5 samples a second
            outcome = judge(body, H, wanted)
            track = to_board_inches(H, body)
            by_pos = {(x, y): hid for hid, _, x, y in load_holes(sqlite3.connect(DB_PATH))}
            cands = [{"uuid": t["uuid"], "name": t["name"], "holds": full_climb_holds(t, by_pos)} for t in top]
            c3 = verify.body_on_route(track, cands, best["uuid"])
            peak = verify.best_route_frame(track, wanted)
            if peak and c3["status"] != verify.NA:
                i, on, n = peak
                c3["images"] = body_image(jid, "c3", clip, fps, body[i], calib, wanted, roles,
                                          f"Best moment, {body[i]['t']:.1f} s: {on} of {n} tracked hands/feet on this climb's holds")
            c4 = verify.start_and_finish(track, wanted, outcome["outcome"] == "topped")
            s_i, f_i = verify.start_finish_frames(track, wanted)
            for tag, i, text in (("c4a", s_i, "Start"), ("c4b", f_i, "Finish")):
                if i is not None and c4["status"] != verify.NA:
                    c4["images"] += body_image(jid, tag, clip, fps, body[i], calib, wanted, roles, f"{text}, {body[i]['t']:.1f} s: a hand on a {text.lower()} hold")
            checks += [c3, c4]
        else:
            outcome = {"outcome": "unknown", "reason": "The climb wasn't recognised confidently, so its finish hold isn't known."}
            for n in (verify.BODY_NAME, verify.START_NAME):
                checks.append(verify.check(n, verify.NA, "Needs a confident match first."))
    else:
        for n in (verify.SPLIT_NAME, verify.BODY_NAME, verify.START_NAME):
            checks.append(verify.check(n, verify.NA, "Needs a video, not a photo."))
    job["step"], job["progress"] = "Drawing the photo", 0.96
    render(base, calib, holds, extras, climb, verdict, jid, roles, outcome)

    margin = best["jaccard"] - (top[1]["jaccard"] if len(top) > 1 else 0) if best else 0
    job["result"] = {
        "id": jid, "image": f"out/{jid}.jpg", "kind": "photo" if is_photo else "video", "source": source,
        "verdict": verdict, "ties": ties, "detected": len(lit),
        "match": round(best["jaccard"], 3) if best else 0, "margin": round(margin, 3),
        "climb": climb, "holds": holds, "attempt": outcome, "verification": verify.summary(checks),
        "not_in_climb": [{"x": h["x"], "y": h["y"], "role": h.get("role_name")} for h in extras],
        "candidates": [{"name": t["name"], "match": round(t["jaccard"], 3), "grade": t.get("v"),
                        "mirrored": t["mirrored"]} for t in top],
        "alignment_features": calib.get("matched_features"),
    }
    (OUT / f"{jid}.json").write_text(json.dumps(job["result"], indent=2))
    for f in CALIB_DIR.glob(f"auto-{jid}*"):          # the per-photo alignments aren't worth keeping
        f.unlink()


def run_job(jid, clip, is_photo):
    with WORK:
        JOBS[jid].update(status="running", progress=0.0)
        try:
            recognise(jid, clip, is_photo)
            JOBS[jid]["status"] = "done"
        except Exception as e:                        # shown to the user, so keep it readable
            JOBS[jid].update(status="error", error=str(e) or type(e).__name__)
        finally:
            JOBS[jid]["progress"] = 1.0
            for f in [*OUT.glob(f"{jid}_*"), *(DATA_DIR / "detections").glob(f"{jid}_*")]:
                f.unlink()


# ---- calibration -----------------------------------------------------------
def save_calibration(name, photo_bytes, pts):
    img = cv2.imdecode(np.frombuffer(photo_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("That image couldn't be read.")
    holes = load_holes(sqlite3.connect(DB_PATH))
    corners = corner_holes(holes)
    board = np.float32([[h[2], h[3]] for _, h in corners])
    H = cv2.getPerspectiveTransform(board, np.float32(pts))
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    png = CALIB_DIR / f"{name}_frame.png"
    cv2.imwrite(str(png), img)
    (CALIB_DIR / f"{name}.json").write_text(json.dumps({
        "video": str(png), "time": 0, "frame_size": [img.shape[1], img.shape[0]],
        "corners": [{"label": l, "hole_id": h[0], "board": [h[2], h[3]], "pixel": list(p)}
                    for (l, h), p in zip(corners, pts)],
        "homography": H.tolist()}, indent=2))
    check = img.copy()
    proj = cv2.perspectiveTransform(np.float32([[h[2], h[3]] for h in holes]).reshape(-1, 1, 2), H).reshape(-1, 2)
    radius = max(4, int(min(img.shape[:2]) / 150))
    for x, y in proj:
        cv2.circle(check, (int(x), int(y)), radius, (0, 255, 255), 2)
    cv2.imwrite(str(CALIB_DIR / f"{name}_check.png"), check)


# ---- http --------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        length = self.headers.get("Content-Length", "")
        if not length.isdigit() or int(length) == 0:
            return self.reply(411, {"error": "empty upload"})
        if int(length) > MAX_UPLOAD:
            return self.reply(413, {"error": "file too large"})
        return self.rfile.read(int(length))

    def file(self, path, ctype):
        if path.is_file():
            self.reply(200, path.read_bytes(), ctype)
        else:
            self.reply(404, {"error": "not found"})

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self.file(HERE / "index.html", "text/html; charset=utf-8")
        if path == "/api/corners":
            corners = corner_holes(load_holes(sqlite3.connect(DB_PATH)))
            return self.reply(200, [{"label": l, "hole": h[1] or h[0]} for l, h in corners])
        if path == "/api/calibrations":
            return self.reply(200, [{"name": n, "check": f"cal/{n}_check.png"} for n in manual_references()])
        if m := re.fullmatch(r"/api/job/([0-9a-f]{12})", path):
            job = JOBS.get(m[1])
            return self.reply(200 if job else 404, job or {"error": "unknown job"})
        if m := re.fullmatch(r"/out/([0-9a-f]{12}(?:-[a-z0-9]+)?)\.jpg", path):
            return self.file(OUT / f"{m[1]}.jpg", "image/jpeg")
        if m := re.fullmatch(r"/cal/([A-Za-z0-9_-]+)_check\.png", path):
            return self.file(CALIB_DIR / f"{m[1]}_check.png", "image/png")
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if url.path == "/api/recognise":
            if ctype not in IMAGE_TYPES and ctype not in VIDEO_EXT:
                return self.reply(415, {"error": "Upload a jpg/png/webp photo or an mp4/mov/webm video."})
            data = self.read_body()
            if data is None:
                return
            jid = uuid.uuid4().hex[:12]
            OUT.mkdir(parents=True, exist_ok=True)
            clip = OUT / f"{jid}_in.mp4"
            is_photo = ctype in IMAGE_TYPES
            JOBS[jid] = {"status": "queued", "progress": 0.0, "step": "Waiting", "result": None, "error": None}
            try:
                if is_photo:
                    photo_to_clip(data, clip)
                else:
                    clip = OUT / f"{jid}_in.{VIDEO_EXT[ctype]}"
                    clip.write_bytes(data)
            except Exception as e:
                JOBS[jid].update(status="error", error=str(e))
                return self.reply(200, {"job": jid})
            threading.Thread(target=run_job, args=(jid, clip, is_photo), daemon=True).start()
            return self.reply(200, {"job": jid})
        if url.path == "/api/calibrate":
            name = (q.get("name") or ["ref1"])[0]
            try:
                pts = [float(v) for v in (q.get("pts") or [""])[0].split(",")]
                assert re.fullmatch(r"[A-Za-z0-9_-]{1,30}", name) and len(pts) == 8
            except (ValueError, AssertionError):
                return self.reply(400, {"error": "Need a name (letters, numbers, - _) and 4 clicked corners."})
            data = self.read_body()
            if data is None:
                return
            try:
                save_calibration(name, data, list(zip(pts[0::2], pts[1::2])))
            except Exception as e:
                return self.reply(400, {"error": str(e)})
            return self.reply(200, {"name": name, "check": f"cal/{name}_check.png"})
        self.reply(404, {"error": "not found"})

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Recogniser on :{args.port}", flush=True)
    ThreadingHTTPServer(("", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

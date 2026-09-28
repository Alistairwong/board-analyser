"""Detect which holds are lit in a board video.

Uses a saved calibration to find every hole in the frame, measures how
brightly coloured each one is across the video (LEDs are vivid; holds and
the wall are greys and beiges), and reports the holes that stay lit along
with their colour. Occlusion by the climber is handled by looking at many
frames and taking a high percentile rather than any single frame.

Usage:
    python video/detect_leds.py data/test1.mov [--calib test1] [--fps 5] [--percentile 75]

Writes data/detections/<video>.json and <video>_leds.png (an overlay to check).
"""
import argparse
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from config import CLIMBS_PATH, DATA_DIR
from calibrate import DB_PATH, CALIB_DIR, load_holes, grab_frame

DETECT_DIR = DATA_DIR / "detections"
PATCH_INCHES = 1.5      # radius around each hole to look for an LED
THRESHOLD_K = 4.5       # lit = this many spreads above a typical unlit hole
MIN_THRESHOLD = 20      # never call anything below this lit

# Hue ranges (OpenCV 0-179) that LEDs show up as on camera. Measured from
# test1: finish ~23 (orange), start/hand ~84-101 (teal/blue), foot ~143
# (purple). The green wall paint sits around 47, between the bands.
LED_HUE_BANDS = [(0, 35), (70, 115), (125, 180)]


def is_led_hue(hue):
    return any(lo <= hue <= hi for lo, hi in LED_HUE_BANDS)


def project(H, pts):
    pts = np.float32(pts).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def hole_patches(H, holes, shape):
    """Pixel centre and a square search window for every hole."""
    centres = project(H, [[h[2], h[3]] for h in holes])
    edges = project(H, [[h[2] + PATCH_INCHES, h[3]] for h in holes])
    radii = np.maximum(2, np.linalg.norm(edges - centres, axis=1)).astype(int)
    height, width = shape[:2]
    patches = []
    for (cx, cy), r in zip(centres, radii):
        x0, x1 = max(0, int(cx) - r), min(width, int(cx) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(height, int(cy) + r + 1)
        patches.append((y0, y1, x0, x1))
    return centres, radii, patches


def score_frame(frame, patches):
    """Per hole: how much its most colourful spot stands out, and its hue.

    The score is the peak colourfulness minus the patch's median, so a small
    bright LED scores highly but an evenly coloured surface doesn't.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)
    colourful = cv2.GaussianBlur(s * v / 255.0, (3, 3), 0)

    scores = np.zeros(len(patches), np.float32)
    hues = np.zeros(len(patches), np.float32)
    for i, (y0, y1, x0, x1) in enumerate(patches):
        p = colourful[y0:y1, x0:x1]
        if p.size == 0:
            continue
        iy, ix = np.unravel_index(np.argmax(p), p.shape)
        scores[i] = p[iy, ix] - np.median(p)
        hues[i] = hsv[y0 + iy, x0 + ix, 0]
    return scores, hues


def circular_mean_hue(hues):
    """Average OpenCV hues (0-179), which wrap around at red."""
    angles = np.asarray(hues) * 2 * np.pi / 180
    mean = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return float((mean * 180 / (2 * np.pi)) % 180)


def hue_distance(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def hex_to_bgr(hex_colour):
    h = str(hex_colour).lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def load_roles():
    """Role names and colours saved by load_climbs.py, with their hues."""
    roles = {}
    try:
        data = json.loads(Path(CLIMBS_PATH).read_text())
    except FileNotFoundError:
        return roles
    for rid, r in data.get("roles", {}).items():
        try:
            bgr = hex_to_bgr(r["colour"])
        except (ValueError, TypeError, KeyError):
            continue
        hue = int(cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
        roles[int(rid)] = {"name": r["name"], "bgr": bgr, "hue": hue}
    return roles


def guess_role(hue, roles):
    if not roles:
        return None
    return min(roles, key=lambda rid: hue_distance(hue, roles[rid]["hue"]))


def detect(video, calib_name=None, fps=5, percentile=75, threshold=None,
           verbose=True, images=True, progress=None):
    """Find the lit holds in a video. Returns (and saves) the detection."""
    video = Path(video)
    say = print if verbose else (lambda *a, **k: None)

    calib_name = calib_name or video.stem
    calib = json.loads((CALIB_DIR / f"{calib_name}.json").read_text())
    H = np.array(calib["homography"], dtype=np.float64)

    holes = load_holes(sqlite3.connect(DB_PATH))
    roles = load_roles()
    if roles:
        say("Role colours: " + ", ".join(
            f"{r['name']} (hue {r['hue']})" for r in roles.values()))
    else:
        say(f"No role colours loaded from {CLIMBS_PATH}; roles won't be guessed.")

    cap = cv2.VideoCapture(str(video))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    step = max(1, round(video_fps / fps))

    all_scores, all_hues = [], []
    patches = centres = radii = None
    size_warning = None
    frame_idx = 0
    while cap.grab():
        if frame_idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if patches is None:
                size = [frame.shape[1], frame.shape[0]]
                if size != calib["frame_size"]:
                    size_warning = f"video is {size}, calibration was {calib['frame_size']}"
                    print(f"Warning: {size_warning}")
                centres, radii, patches = hole_patches(H, holes, frame.shape)
            s, h = score_frame(frame, patches)
            all_scores.append(s)
            all_hues.append(h)
            if progress and total_frames > 0:
                progress(min(1.0, frame_idx / total_frames))
        frame_idx += 1
    cap.release()

    if not all_scores:
        raise RuntimeError(f"No frames read from {video}")

    scores = np.array(all_scores)      # frames x holes
    hues = np.array(all_hues)
    hole_scores = np.percentile(scores, percentile, axis=0)

    # Almost every hole is unlit, so the typical score and its spread describe
    # "unlit"; anything far above that is an LED, bright hand or dim foot alike.
    if threshold is None:
        typical = np.median(hole_scores)
        spread = 1.4826 * np.median(np.abs(hole_scores - typical))
        threshold = max(MIN_THRESHOLD, typical + THRESHOLD_K * spread)

    # Each hole's colour: average hue over its strongest quarter of frames
    n_top = max(1, len(scores) // 4)
    top_frames = np.argsort(-scores, axis=0)[:n_top]
    hole_hues = np.array([circular_mean_hue(hues[top_frames[:, i], i])
                          for i in range(len(holes))])

    lit = []
    for i in np.where((hole_scores > threshold)
                      & np.array([is_led_hue(h) for h in hole_hues]))[0]:
        hue = float(hole_hues[i])
        rid = guess_role(hue, roles)
        hid, name, x, y = holes[i]
        lit.append({
            "hole_id": hid, "name": name, "x": x, "y": y,
            "score": round(float(hole_scores[i]), 1),
            "hue": round(hue, 1),
            "role": rid,
            "role_name": roles[rid]["name"] if rid is not None else None,
        })
    lit.sort(key=lambda h: (-h["y"], h["x"]))
    lit_ids = {h["hole_id"] for h in lit}

    say(f"Analysed {len(scores)} frames, {len(holes)} holes.")
    say(f"Lit threshold: {threshold:.0f}")
    say("Top scores (* = lit, x = above threshold but not an LED colour):")
    for i in np.argsort(-hole_scores)[:30]:
        if holes[i][0] in lit_ids:
            mark = "*"
        elif hole_scores[i] > threshold:
            mark = "x"
        else:
            mark = " "
        say(f"  {mark} hole {holes[i][1]:>8}  score {hole_scores[i]:6.1f}  hue {hole_hues[i]:5.1f}")
    say(f"\n{len(lit)} lit holds:")
    for h in lit:
        say(f"  {h['name']:>8}  score {h['score']:6.1f}  hue {h['hue']:5.1f}  -> {h['role_name'] or '?'}")

    result = {
        "video": str(video),
        "calibration": calib_name,
        "frames": len(scores),
        "threshold": float(threshold),
        "size_warning": size_warning,
        "lit": lit,
    }
    DETECT_DIR.mkdir(parents=True, exist_ok=True)
    out = DETECT_DIR / f"{video.stem}.json"
    out.write_text(json.dumps(result, indent=2))

    if images:
        # Overlay: lit holds circled in their guessed role colour
        frame = grab_frame(video, calib["time"])
        index = {h[0]: i for i, h in enumerate(holes)}
        for h in lit:
            i = index[h["hole_id"]]
            colour = roles[h["role"]]["bgr"] if h["role"] is not None else (255, 0, 255)
            c = (int(centres[i][0]), int(centres[i][1]))
            cv2.circle(frame, c, int(radii[i] * 1.6), (0, 0, 0), 5)
            cv2.circle(frame, c, int(radii[i] * 1.6), colour, 3)
        cv2.imwrite(str(DETECT_DIR / f"{video.stem}_leds.png"), frame)

        # Tuning aid: every hole's score written on the frame (lit ones in green)
        scores_img = grab_frame(video, calib["time"])
        for i, (cx, cy) in enumerate(centres):
            text = f"{hole_scores[i]:.0f}"
            colour = (0, 255, 0) if holes[i][0] in lit_ids else (0, 255, 255)
            org = (int(cx) - 10, int(cy) + 5)
            cv2.putText(scores_img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(scores_img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        cv2.imwrite(str(DETECT_DIR / f"{video.stem}_scores.png"), scores_img)

    say(f"\nSaved {out}" + (" and overlay images." if images else "."))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--calib", default=None,
                    help="calibration name in data/calibrations (defaults to video name)")
    ap.add_argument("--fps", type=float, default=5,
                    help="frames per second of video to analyse")
    ap.add_argument("--percentile", type=float, default=75,
                    help="higher = more tolerant of the climber blocking a hold")
    ap.add_argument("--threshold", type=float, default=None,
                    help="set the lit/unlit cut-off by hand instead of automatically")
    args = ap.parse_args()
    detect(args.video, args.calib, args.fps, args.percentile, args.threshold)


if __name__ == "__main__":
    main()

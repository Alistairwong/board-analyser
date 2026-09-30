"""Render an annotated video: pose skeleton, board holds, and live hold
assignment overlaid on the original footage. A visual sanity check for the
numbers analyse_movement.py prints -- especially useful while tuning
HOLD_TOLERANCE/MIN_RUN in movement.py against what actually happened.

Usage:
    python video/render_movement.py data/clip.mov [--calib name] [--fps 10]

Writes data/results/<video>_movement.mp4 (same fps as the pose sampling,
so the clip plays back at roughly the right speed but at reduced smoothness).
"""
import argparse
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from calibrate import CALIB_DIR, DB_PATH, load_holes
from detect_leds import load_roles, project
from movement import assign_holds
from pose import estimate_landmarks, to_board_inches
from recognise import RESULTS_DIR, resolved_holds

SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
    ("left_wrist", "left_hand"), ("right_wrist", "right_hand"),
    ("left_hip", "left_knee"), ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_foot"), ("right_ankle", "right_foot"),
]
LIMB_COLOUR = {   # BGR
    "left_hand": (255, 80, 0), "right_hand": (0, 80, 255),
    "left_wrist": (255, 80, 0), "right_wrist": (0, 80, 255),
    "left_elbow": (255, 150, 60), "right_elbow": (60, 150, 255),
    "left_knee": (255, 230, 80), "right_knee": (80, 230, 255),
    "left_shoulder": (255, 255, 255), "right_shoulder": (255, 255, 255),
    "left_hip": (255, 255, 255), "right_hip": (255, 255, 255),
    "left_foot": (255, 200, 0), "right_foot": (0, 200, 255),
}


def render(video, calib_name=None, fps=10, verbose=True, holds=None, frames=None):
    video = Path(video)
    say = print if verbose else (lambda *a, **k: None)
    calib_name = calib_name or video.stem
    calib = json.loads((CALIB_DIR / f"{calib_name}.json").read_text())
    H = np.array(calib["homography"], dtype=np.float64)

    holds = holds or resolved_holds(video, calib_name, verbose=verbose)   # holds/frames: pass them in to skip the repeat work
    # This climb's holds, coloured by their role (same colours as the board's
    # own LEDs); everything else is just a small reference dot.
    bgr_by_role_name = {r["name"]: r["bgr"] for r in load_roles().values()}
    role_colour = {h["hole_id"]: bgr_by_role_name[h["role_name"]]
                   for h in holds if h.get("role_name") in bgr_by_role_name}

    # Draw every hole on the board for reference, not just this climb's lit
    # ones -- otherwise most of the board never shows up in the overlay.
    all_holes = load_holes(sqlite3.connect(DB_PATH))
    hole_pixels = {hid: project(H, [[x, y]])[0] for hid, name, x, y in all_holes}

    say(f"Estimating pose at {fps} fps...")
    frames = frames or estimate_landmarks(video, fps=fps)
    track = to_board_inches(H, frames)
    assignments = assign_holds(track, holds)

    cap = cv2.VideoCapture(str(video))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(video_fps / fps))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{video.stem}_movement.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx, sample_i = 0, 0
    while cap.grab():
        if frame_idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if sample_i < len(frames):
                pix = frames[sample_i]["landmarks"]
                assign = assignments[sample_i]["assign"] if sample_i < len(assignments) else {}
                held_now = {v for v in assign.values() if v is not None}

                for hid, (px, py) in hole_pixels.items():
                    if hid in role_colour:
                        thickness = -1 if hid in held_now else 2   # filled = currently held
                        cv2.circle(frame, (int(px), int(py)), 10, role_colour[hid], thickness)
                    else:
                        cv2.circle(frame, (int(px), int(py)), 4, (150, 150, 150), 1)

                for a, b in SKELETON_EDGES:
                    if a in pix and b in pix:
                        pa = (int(pix[a][0]), int(pix[a][1]))
                        pb = (int(pix[b][0]), int(pix[b][1]))
                        cv2.line(frame, pa, pb, (255, 255, 255), 2)
                for name, (x, y, vis) in pix.items():
                    if vis < 0.5 or name.endswith(("_pinky", "_index")):   # those only feed the hand centre
                        continue
                    cv2.circle(frame, (int(x), int(y)), 6, LIMB_COLOUR.get(name, (200, 200, 200)), -1)

                t = frames[sample_i]["t"]
                for text, colour, thick in ((f"t={t:.1f}s", (0, 0, 0), 3), (f"t={t:.1f}s", (255, 255, 255), 1)):
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, thick)
                sample_i += 1
            writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    say(f"Saved {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--calib", default=None,
                    help="calibration name in data/calibrations (defaults to video name)")
    ap.add_argument("--fps", type=float, default=10,
                    help="frames per second to sample and render")
    args = ap.parse_args()
    render(args.video, args.calib, args.fps)


if __name__ == "__main__":
    main()

"""Analyse body movement in a board video: which holds each limb used, when
moves happened, timing/effort metrics, and rule-based technique flags.

Uses the video's calibration and the climb's hold list -- the climb's own
full hold list from the database when we're confident which climb it is
(recognise.resolved_holds()), since LED detection alone can miss a dim or
occluded hold.

Usage:
    python video/analyse_movement.py data/clip.mov
    python video/analyse_movement.py data/clip.mov --calib tripod-left --fps 15

Writes data/results/<video>_movement.json (kept separate from
recognise.py's <video>.json match result).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from calibrate import CALIB_DIR
from movement import analyse
from pose import estimate_landmarks, to_board_inches
from recognise import RESULTS_DIR, resolved_holds


def analyse_movement(video, calib_name=None, fps=10, verbose=True):
    video = Path(video)
    say = print if verbose else (lambda *a, **k: None)

    calib_name = calib_name or video.stem
    calib = json.loads((CALIB_DIR / f"{calib_name}.json").read_text())
    H = np.array(calib["homography"], dtype=np.float64)

    holds = resolved_holds(video, calib_name, verbose=verbose)

    say(f"Estimating pose at {fps} fps (this can take a while)...")
    frames = estimate_landmarks(video, fps=fps)
    track = to_board_inches(H, frames)

    result = analyse(track, holds)
    result["video"] = str(video)
    result["calibration"] = calib_name

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{video.stem}_movement.json"
    out.write_text(json.dumps(result, indent=2))

    m = result["metrics"]
    say(f"\n{m['move_count']} moves, {m['duration_s']:.1f}s, "
        f"{m['moves_per_min']:.1f} moves/min, outcome: {result['attempt']['outcome']}")
    if m["longest_pause_s"]:
        say(f"  longest pause: {m['longest_pause_s']:.1f}s at hold {m['longest_pause_hold']}")
    for flag in result["flags"]:
        say(f"  - {flag}")
    say(f"\nSaved {out}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--calib", default=None,
                    help="calibration name in data/calibrations (defaults to video name)")
    ap.add_argument("--fps", type=float, default=10,
                    help="frames per second of video to analyse for pose")
    args = ap.parse_args()
    analyse_movement(args.video, args.calib, args.fps)


if __name__ == "__main__":
    main()

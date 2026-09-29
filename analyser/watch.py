"""Sort each new recorded clip into a folder for the climb it shows.

Runs on the NAS (Docker, see docker-compose.yml). Watches data/recordings/, which
webapp/search.py's /upload fills, and for every finished clip:
  1. (webm only) re-encode to mp4 at a fixed frame rate, so frame timing is reliable
  2. calibrate the camera automatically against a hand-made calibration
  3. find the lit holds and match them against the climb database
  4. move the clip and a small result file to data/climbs/<climb>/ (or _unidentified/)

Only light work happens here. Movement analysis, pose tracking and overlays are
done on request from the PC with video/analyse_movement.py.

The recognition code is video/'s own: the Dockerfile copies those files in beside
this one, so nothing is duplicated in the repo.
"""
import json
import re
import shutil
import subprocess
import time
import traceback
from pathlib import Path

from autocalibrate import auto_calibrate
from config import CLIMBS_PATH, DATA_DIR
from detect_leds import detect
from match_climb import grade_text, match
from recognise import GOOD, assess

INBOX = DATA_DIR / "recordings"
OUT = DATA_DIR / "climbs"
CLIP_TYPES = {".mp4", ".webm"}
POLL_SECONDS = 5


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "climb"


def normalise(clip):
    """Browsers' webm has no reliable frame rate; re-encode to a constant-rate mp4."""
    if clip.suffix != ".webm":
        return clip
    mp4 = clip.with_suffix(".mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-r", "30", "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(mp4)], check=True)
    clip.unlink()
    return mp4


def identify(clip):
    """Return the result dict for one clip (never moves anything)."""
    result = {"video": clip.name, "verdict": "no match", "ties": 0, "best": None, "candidates": []}
    calib = auto_calibrate(clip, name=clip.stem, verbose=False)
    if not calib:
        result["verdict"] = "not calibrated"
        return result
    det = detect(clip, calib, images=False, verbose=False)
    top = match(det["lit"], top=5) if det["lit"] else []
    result["verdict"], result["ties"] = assess(top)
    result.update(calibration=calib, detected_holds=len(det["lit"]), size_warning=det.get("size_warning"),
                  best=top[0] if top else None, candidates=top)
    return result


def file_away(clip, result):
    best = result["best"]
    if best and result["verdict"] in GOOD:
        folder = OUT / f"{slug(best['name'])}-{best['uuid'][:8]}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "climb.json").write_text(json.dumps(
            {"name": best["name"], "uuid": best["uuid"], "grade": grade_text(best)}, indent=2))
    else:
        folder = OUT / "_unidentified"
        folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{clip.stem}.json").write_text(json.dumps(result, indent=2))
    shutil.move(str(clip), folder / clip.name)
    return folder


def process(clip):
    clip = normalise(clip)
    folder = file_away(clip, identify(clip))
    print(f"{clip.name} -> {folder.name}", flush=True)


def main():
    INBOX.mkdir(parents=True, exist_ok=True)
    print(f"Watching {INBOX}", flush=True)
    while True:
        clips = sorted(f for f in INBOX.iterdir() if f.suffix in CLIP_TYPES)
        if clips and not CLIMBS_PATH.exists():
            print(f"{CLIMBS_PATH.name} is missing (the web app rebuilds it); waiting.", flush=True)
            clips = []
        for clip in clips:
            try:
                process(clip)
            except Exception:
                failed = OUT / "_failed"
                failed.mkdir(parents=True, exist_ok=True)
                (failed / f"{clip.stem}.error.txt").write_text(traceback.format_exc())
                if clip.exists():
                    shutil.move(str(clip), failed / clip.name)
                print(f"{clip.name} failed, moved to _failed/", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

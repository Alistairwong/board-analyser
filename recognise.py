"""Recognise the climb in one or more board videos with a single command.

For each video: pick a calibration, detect the lit holds, match them against
the climb database, and save the result to data/results/<video>.json.

Usage:
    python recognise.py data/clip.mov
    python recognise.py data/*.mov                 # several at once (works on Windows too)
    python recognise.py data/clip.mov --calib tripod-left
    python recognise.py data/clip.mov --calibrate  # click corners for this video's camera spot
    python recognise.py data/clip.mov --verbose    # show the full detection printout

Calibration used, in order: --calib if given, then one named after the video,
then DEFAULT_CALIB from config.py, otherwise you're asked to click corners.
"""
import argparse
import glob
import json
from pathlib import Path

from config import DEFAULT_CALIB
from calibrate import CALIB_DIR, calibrate
from autocalibrate import auto_calibrate
from detect_leds import detect
from match_climb import match, print_matches, grade_text

RESULTS_DIR = Path("data/results")


def pick_calibration(video, calib=None, force=False):
    if force:
        return calibrate(video)
    if calib:
        return calib
    if (CALIB_DIR / f"{video.stem}.json").exists():
        return video.stem
    if DEFAULT_CALIB and (CALIB_DIR / f"{DEFAULT_CALIB}.json").exists():
        return DEFAULT_CALIB
    print(f"No calibration found for {video.name}, trying automatic calibration.")
    return auto_calibrate(video) or calibrate(video)


def assess(top):
    """How much to trust the best match, plus how many climbs tie with it."""
    if not top:
        return "no match", 0
    best = top[0]
    ties = sum(1 for r in top[1:]
               if r["jaccard"] == best["jaccard"] and r["roles_agree"] == best["roles_agree"])
    runner_up = top[1]["jaccard"] if len(top) > 1 else 0.0
    margin = best["jaccard"] - runner_up
    if ties and best["jaccard"] >= 0.8:
        return "tied", ties
    if best["jaccard"] >= 0.8 and margin >= 0.15:
        return "confident", 0
    if best["jaccard"] >= 0.6 and margin >= 0.05:
        return "likely", 0
    return "uncertain", 0


def recognise(video, calib=None, force_calibrate=False, verbose=False):
    video = Path(video)
    calib_name = pick_calibration(video, calib, force_calibrate)
    det = detect(video, calib_name, verbose=verbose)
    return finish(video, calib_name, det)


def finish(video, calib_name, det):
    """Match a detection against the climbs, then save and return the result."""
    video = Path(video)
    top = match(det["lit"], top=10)
    verdict, ties = assess(top)

    result = {
        "video": str(video),
        "calibration": calib_name,
        "detected_holds": len(det["lit"]),
        "verdict": verdict,
        "ties": ties,
        "best": top[0] if top else None,
        "candidates": top,
        "size_warning": det.get("size_warning"),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{video.stem}.json").write_text(json.dumps(result, indent=2))
    return result


def one_line(r):
    best = r["best"]
    if not best:
        return f"{Path(r['video']).name}: no match ({r['detected_holds']} holds detected)"
    tag = " (mirrored)" if best["mirrored"] else ""
    extra = f", tied with {r['ties']} other(s)" if r["ties"] else ""
    return (f"{Path(r['video']).name}: {best['name']}{tag}  ({grade_text(best)})\n"
            f"  {best['jaccard']:.0%} match, {r['verdict']}{extra}  [calibration: {r['calibration']}]")


def expand(patterns):
    """Expand wildcards ourselves, since Windows PowerShell doesn't."""
    videos = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        videos.extend(matches if matches else [p])
    return [Path(v) for v in videos]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="video file(s); wildcards like data/*.mov are fine")
    ap.add_argument("--calib", default=None, help="calibration name to use for every video")
    ap.add_argument("--calibrate", action="store_true",
                    help="click corners for each video instead of reusing a calibration")
    ap.add_argument("--verbose", action="store_true", help="show the full detection printout")
    args = ap.parse_args()

    videos = expand(args.videos)
    results = []
    for video in videos:
        if not video.exists():
            print(f"{video}: file not found, skipping")
            continue
        try:
            r = recognise(video, args.calib, args.calibrate, args.verbose)
        except Exception as e:  # keep going through a batch if one video fails
            print(f"{video.name}: failed ({e})")
            continue
        results.append(r)
        if r["size_warning"]:
            print(f"  warning: {r['size_warning']}; this video may need its own calibration "
                  f"(--calibrate)")
        if len(videos) == 1:
            print(f"{r['detected_holds']} lit holds detected\n")
            print_matches(r["candidates"][:5], r["detected_holds"])
            print()
        print(one_line(r))

    if len(results) > 1:
        counts = {}
        for r in results:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        print(f"\n{len(results)} videos: " + ", ".join(f"{n} {v}" for v, n in counts.items()))
    if results:
        print(f"Results saved in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()

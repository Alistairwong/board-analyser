"""Calibrate a new camera position automatically, with no clicking.

The holds never move, so any two photos of the board are related by a single
perspective transform. This finds hundreds of matching features (hold edges,
bolt holes, textures) between a frame of the new video and a frame from a
calibration you clicked by hand, fits that transform, and chains it with the
hand calibration to get the new one.

Usage:
    python video/autocalibrate.py data/backlog/clip.mov [--name gym-left] [--ref test1]
    python video/autocalibrate.py "data/backlog/*.mov"

It needs at least one hand-made calibration to use as a reference.
"""
import argparse
import glob
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from config import PRODUCT_SIZE_ID
from calibrate import CALIB_DIR, DB_PATH, draw_check, grab_frame, load_holes

MAX_SIDE = 1600          # downscale frames to this for feature matching
MIN_INLIERS = 40         # fewer matched features than this = don't trust it
PLENTY = 300             # this many matched features is clearly good: stop looking
RATIO = 0.75             # Lowe's ratio test for feature matches
TRY_TIMES = (1.0, 0.5, 2.0, 3.0)   # seconds into the new video to try
SIFT = cv2.SIFT_create(nfeatures=4000)


def manual_references():
    refs = []
    for f in sorted(CALIB_DIR.glob("*.json")):
        try:
            if not json.loads(f.read_text()).get("auto"):
                refs.append(f.stem)
        except (json.JSONDecodeError, OSError):
            continue
    return refs


def reference_frame(ref):
    calib = json.loads((CALIB_DIR / f"{ref}.json").read_text())
    png = CALIB_DIR / f"{ref}_frame.png"
    frame = cv2.imread(str(png)) if png.exists() else grab_frame(calib["video"], calib["time"])
    return calib, frame


def board_mask(calib, shape):
    """Only use features on the board itself, not the room around it."""
    con = sqlite3.connect(DB_PATH)
    l, r, b, t = con.execute(
        "SELECT edge_left, edge_right, edge_bottom, edge_top FROM product_sizes WHERE id = ?",
        (PRODUCT_SIZE_ID,),
    ).fetchone()
    H = np.array(calib["homography"], dtype=np.float64)
    corners = np.float32([[l, b], [r, b], [r, t], [l, t]]).reshape(-1, 1, 2)
    poly = cv2.perspectiveTransform(corners, H).reshape(-1, 2).astype(np.int32)
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    return mask


def features(img, mask=None):
    scale = min(1.0, MAX_SIDE / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if mask is not None and scale < 1:
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    kp, des = SIFT.detectAndCompute(gray, mask)
    pts = np.float32([k.pt for k in kp]) / scale
    return pts, des


def fit(ref_feats, new_feats):
    """Transform from reference pixels to new pixels, and how many features agree."""
    (ref_pts, ref_des), (new_pts, new_des) = ref_feats, new_feats
    if ref_des is None or new_des is None or len(ref_des) < 4 or len(new_des) < 4:
        return None, 0
    pairs = cv2.BFMatcher().knnMatch(ref_des, new_des, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < MIN_INLIERS:
        return None, len(good)
    src = ref_pts[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = new_pts[[m.trainIdx for m in good]].reshape(-1, 1, 2)
    G, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if G is None:
        return None, 0
    return G, int(inliers.sum())


def auto_calibrate(video, name=None, refs=None, verbose=True, progress=None, info=None):
    """Try to calibrate automatically. Returns the calibration name, or None."""
    video = Path(video)
    refs = refs or manual_references()
    if not refs:
        if verbose:
            print("No hand-made calibration to use as a reference yet.")
        return None

    if not video.exists():
        if verbose:
            print(f"{video}: file not found")
        return None

    best = None   # (inliers, ref, t, G, frame)
    frames_read = 0
    attempts, done = len(refs) * len(TRY_TIMES), 0
    for ref in refs:
        if best and best[0] >= PLENTY:
            break
        try:
            ref_calib, ref_frame = reference_frame(ref)
        except (SystemExit, FileNotFoundError, cv2.error):
            continue
        ref_feats = features(ref_frame, board_mask(ref_calib, ref_frame.shape))
        for t in TRY_TIMES:
            done += 1
            if progress:
                progress(done / attempts)
            try:
                frame = grab_frame(video, t)
            except SystemExit:
                continue
            frames_read += 1
            G, n = fit(ref_feats, features(frame))
            if G is not None and (best is None or n > best[0]):
                best = (n, ref, t, G, frame, ref_calib)
            if best and best[0] >= PLENTY:
                break

    if info is not None:
        info["features"] = best[0] if best else 0
        info["frames_read"] = frames_read
    if frames_read == 0:
        if verbose:
            print(f"{video.name}: couldn't read any frames from the video")
        return None
    if best is None or best[0] < MIN_INLIERS:
        if verbose:
            found = best[0] if best else 0
            print(f"{video.name}: auto-calibration failed, only {found} matching features "
                  f"(need {MIN_INLIERS})")
        return None

    n, ref, t, G, frame, ref_calib = best
    H = G @ np.array(ref_calib["homography"], dtype=np.float64)
    H /= H[2, 2]

    name = name or f"auto-{video.stem}"
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    (CALIB_DIR / f"{name}.json").write_text(json.dumps({
        "video": str(video),
        "time": t,
        "frame_size": [frame.shape[1], frame.shape[0]],
        "auto": True,
        "reference": ref,
        "matched_features": n,
        "homography": H.tolist(),
    }, indent=2))
    cv2.imwrite(str(CALIB_DIR / f"{name}_frame.png"), frame)
    holes = load_holes(sqlite3.connect(DB_PATH))
    cv2.imwrite(str(CALIB_DIR / f"{name}_check.png"), draw_check(frame, holes, H))
    if verbose:
        print(f"Auto-calibrated {video.name} from '{ref}' ({n} matching features). "
              f"Saved as '{name}'.")
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="video file(s); wildcards like data/backlog/*.mov are fine")
    ap.add_argument("--name", default=None,
                    help="name for this camera position (only with a single video)")
    ap.add_argument("--ref", default=None, help="hand-made calibration to use as the reference")
    args = ap.parse_args()

    videos = []
    for pattern in args.videos:   # expand wildcards ourselves, since PowerShell doesn't
        videos.extend(sorted(glob.glob(pattern)) or [pattern])
    if args.name and len(videos) > 1:
        raise SystemExit("--name only works with a single video")

    refs = [args.ref] if args.ref else None
    for v in videos:
        name = auto_calibrate(v, args.name, refs)
        if name:
            print(f"  check the overlay: {CALIB_DIR / (name + '_check.png')}")


if __name__ == "__main__":
    main()

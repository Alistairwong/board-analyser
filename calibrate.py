"""Calibrate one camera position: map board coordinates to video pixels.

You click four corner holes on a frame from the video. The script works out
the perspective transform (homography) from board inches to image pixels,
then draws every hole on the frame so you can check the alignment by eye.

Usage:
    python calibrate.py path/to/video.mp4 [--time 1.0] [--name tripod-left]

The calibration is saved to data/calibrations/<name>.json and can be reused
for every clip filmed from the same camera position.
"""
import argparse
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from config import LAYOUT_ID, PRODUCT_SIZE_ID

DB_PATH = Path("data/tension.db")
CALIB_DIR = Path("data/calibrations")
MAX_DISPLAY = 1200      # longest side of the on-screen window, in pixels
WIN = "calibrate"


def load_holes(con):
    """All holes used by the layout that sit within this board size."""
    cur = con.cursor()
    left, right, bottom, top = cur.execute(
        "SELECT edge_left, edge_right, edge_bottom, edge_top "
        "FROM product_sizes WHERE id = ?",
        (PRODUCT_SIZE_ID,),
    ).fetchone()
    return cur.execute(
        "SELECT DISTINCT h.id, h.name, h.x, h.y FROM holes h "
        "JOIN placements p ON p.hole_id = h.id "
        "WHERE p.layout_id = ? "
        "AND h.x > ? AND h.x < ? AND h.y > ? AND h.y < ?",
        (LAYOUT_ID, left, right, bottom, top),
    ).fetchall()


def corner_holes(holes):
    """The holes at each end of the bottom and top rows, in click order.

    The board is a staggered grid, so "furthest into the corner" can tie
    between neighbouring rows. Using the ends of the lowest and highest rows
    gives an unambiguous rectangle that's easy to spot on the wall.
    """
    bottom_y = min(h[3] for h in holes)
    top_y = max(h[3] for h in holes)
    bottom = [h for h in holes if h[3] == bottom_y]
    top = [h for h in holes if h[3] == top_y]
    return [
        ("left end of the bottom row", min(bottom, key=lambda h: h[2])),
        ("right end of the bottom row", max(bottom, key=lambda h: h[2])),
        ("right end of the top row", max(top, key=lambda h: h[2])),
        ("left end of the top row", min(top, key=lambda h: h[2])),
    ]


def grab_frame(video, t):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Couldn't read a frame at {t}s from {video}")
    return frame


def collect_clicks(frame, corners):
    scale = min(1.0, MAX_DISPLAY / max(frame.shape[:2]))
    display = cv2.resize(frame, None, fx=scale, fy=scale)
    clicks = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x / scale, y / scale))

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_click)

    while len(clicks) < 4:
        img = display.copy()
        for cx, cy in clicks:
            cv2.circle(img, (int(cx * scale), int(cy * scale)), 6, (0, 0, 255), -1)
        label, hole = corners[len(clicks)]
        text = f"{len(clicks) + 1}/4: click the {label}  (u = undo, q = quit)"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WIN, img)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and clicks:
            clicks.pop()
        elif key == ord("q"):
            raise SystemExit("Quit without saving.")

    return clicks


def draw_check(frame, holes, H):
    pts = np.float32([[h[2], h[3]] for h in holes]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    radius = max(4, int(min(frame.shape[:2]) / 150))
    check = frame.copy()
    for x, y in proj:
        cv2.circle(check, (int(x), int(y)), radius, (0, 255, 255), 2)
    return check


def calibrate(video, t=1.0, name=None):
    """Interactively calibrate a camera position. Returns the calibration name."""
    video = Path(video)
    con = sqlite3.connect(DB_PATH)
    holes = load_holes(con)
    corners = corner_holes(holes)

    print(f"{len(holes)} holes on the board. Corner holes to click, in order:")
    for label, (hid, hname, x, y) in corners:
        print(f"  {label:28s} hole {hname or hid}  (x={x}, y={y})")

    frame = grab_frame(video, t)
    clicks = collect_clicks(frame, corners)

    board_pts = np.float32([[h[2], h[3]] for _, h in corners])
    img_pts = np.float32(clicks)
    H = cv2.getPerspectiveTransform(board_pts, img_pts)

    name = name or video.stem
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    (CALIB_DIR / f"{name}.json").write_text(json.dumps({
        "video": str(video),
        "time": t,
        "frame_size": [frame.shape[1], frame.shape[0]],
        "corners": [{"label": l, "hole_id": h[0], "board": [h[2], h[3]], "pixel": list(p)}
                    for (l, h), p in zip(corners, clicks)],
        "homography": H.tolist(),
    }, indent=2))

    cv2.imwrite(str(CALIB_DIR / f"{name}_frame.png"), frame)   # reference for autocalibrate
    check = draw_check(frame, holes, H)
    cv2.imwrite(str(CALIB_DIR / f"{name}_check.png"), check)
    print(f"\nSaved calibration and overlay to {CALIB_DIR}/{name}.*")

    scale = min(1.0, MAX_DISPLAY / max(check.shape[:2]))
    cv2.imshow(WIN, cv2.resize(check, None, fx=scale, fy=scale))
    print("Check the overlay, then press any key on the image window (or close it).")
    while True:
        key = cv2.waitKey(100)
        if key != -1 or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--time", type=float, default=1.0,
                    help="seconds into the video to grab the frame from")
    ap.add_argument("--name", default=None,
                    help="name for this camera position (defaults to video name)")
    args = ap.parse_args()
    calibrate(args.video, args.time, args.name)


if __name__ == "__main__":
    main()

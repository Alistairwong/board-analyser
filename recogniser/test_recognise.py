"""End-to-end check with a synthetic board: no real footage needed.

Renders a textured board with the real hole layout, "photographs" it from two camera
positions, then drives the running server: calibrate from photo 1 (no LEDs), recognise
photo 2 (a known climb lit, from a different angle). Run inside the container:

    docker run --rm -v <scratch data>:/data -v $PWD/recogniser:/t --entrypoint sh <image> \
        -c 'python server.py & sleep 3; python /t/test_recognise.py'
"""
import json
import sqlite3
import time
import urllib.request

import cv2
import numpy as np

from calibrate import corner_holes, load_holes
from calibrate import DB_PATH
from config import CLIMBS_PATH

URL = "http://localhost:8000"
PX = 12                        # base image pixels per inch
PAD = 6


def base_px(x, y):
    return (x + 44 + PAD) * PX, (144 - y + PAD) * PX


def board_image(holes, lit=()):
    rng = np.random.default_rng(1)
    w, h = int((88 + 2 * PAD) * PX), int((144 + 2 * PAD) * PX)
    tex = cv2.GaussianBlur(rng.integers(90, 170, (h // 4, w // 4), dtype=np.uint8), (0, 0), 1.2)
    img = cv2.cvtColor(cv2.resize(tex, (w, h), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    img[..., 0] = (img[..., 0] * 0.8).astype(np.uint8)              # a beige wall, not grey
    for _, _, x, y in holes:
        cv2.circle(img, tuple(int(v) for v in base_px(x, y)), int(0.6 * PX), (40, 40, 40), 2)
    for x, y, bgr in lit:
        c = tuple(int(v) for v in base_px(x, y))
        cv2.circle(img, c, int(0.9 * PX), tuple(int(v * 0.5) for v in bgr), -1)
        cv2.circle(img, c, int(0.5 * PX), bgr, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def camera(img, quad):
    """Photograph the board image: map its rectangle onto `quad` in a 1400x1800 frame."""
    h, w = img.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(quad))
    return cv2.warpPerspective(img, M, (1400, 1800)), M


def post(path, body, ctype):
    req = urllib.request.Request(URL + path, data=body, headers={"Content-Type": ctype}, method="POST")
    return json.load(urllib.request.urlopen(req))


def main():
    con = sqlite3.connect(DB_PATH)
    holes = load_holes(con)
    data = json.loads(CLIMBS_PATH.read_text())
    roles = {int(k): v["name"] for k, v in data["roles"].items()}
    pos = {(x, y): hid for hid, _, x, y in holes}
    # An asymmetric climb (so its mirror image isn't a tie) with a decent number of holds
    climb = next(c for c in data["climbs"] if 10 <= len(c["holds"]) <= 25
                 and {(h["x"], h["y"]) for h in c["holds"]} != {(-h["x"], h["y"]) for h in c["holds"]})
    colours = {"start": (200, 200, 0), "middle": (200, 200, 0), "finish": (0, 140, 255), "foot": (255, 60, 200)}
    lit = [(h["x"], h["y"], colours[roles[h["role"]]]) for h in climb["holds"]]

    # Photo 1: reference, straight-on, no LEDs. Click the four corner holes.
    ref, M1 = camera(board_image(holes), [[150, 150], [1250, 130], [1270, 1650], [130, 1670]])
    clicks = []
    for _, h in corner_holes(holes):
        clicks += list(cv2.perspectiveTransform(np.float32([[base_px(h[2], h[3])]]), M1)[0][0])
    ok, png = cv2.imencode(".png", ref)
    r = post("/api/calibrate?name=ref1&pts=" + ",".join(f"{v:.1f}" for v in clicks), png.tobytes(), "image/png")
    print("calibrated:", r)

    # Photo 2: different camera position, climb lit
    ok, jpg = cv2.imencode(".jpg", camera(board_image(holes, lit), [[260, 90], [1180, 210], [1330, 1560], [90, 1720]])[0])
    job = post("/api/recognise", jpg.tobytes(), "image/jpeg")["job"]
    for _ in range(300):
        j = json.load(urllib.request.urlopen(f"{URL}/api/job/{job}"))
        if j["status"] in ("done", "error"):
            break
        time.sleep(1)
    assert j["status"] == "done", j
    res = j["result"]
    print(json.dumps({k: res[k] for k in ("verdict", "match", "margin", "detected", "climb")}, indent=1)[:900])
    assert res["climb"]["uuid"] == climb["uuid"], f"expected {climb['name']}, got {res['climb']}"
    assert res["verdict"] in ("confident", "likely", "tied"), res["verdict"]
    seen = sum(h["seen"] for h in res["holds"])
    assert len(res["holds"]) == len(climb["holds"]) and seen >= 0.8 * len(climb["holds"]), (seen, len(res["holds"]))
    assert all(0 <= h["confidence"] <= 1 for h in res["holds"])
    img = urllib.request.urlopen(URL + "/" + res["image"]).read()
    assert img[:2] == b"\xff\xd8", "result isn't a jpeg"
    open("/tmp/last_result.jpg", "wb").write(img)
    print(f"OK: {climb['name']} recognised, {seen}/{len(res['holds'])} holds seen")

    # Video: same lit climb, but a light-grey "climber" sweeps across the board (over some LEDs).
    # The result photo must be the clean plate: no climber in it.
    quad = [[260, 90], [1180, 210], [1330, 1560], [90, 1720]]
    board, _ = camera(board_image(holes, lit), quad)
    inside = cv2.warpPerspective(np.full(board_image(holes).shape[:2], 255, np.uint8),
                                 camera(board_image(holes), quad)[1], (1400, 1800)) > 0
    vw = cv2.VideoWriter("/tmp/v.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (1400, 1800))
    frames_n = 60
    for i in range(frames_n):
        f = board.copy()
        x = 100 + int(900 * i / frames_n)
        f[600:1200, x:x + 250][inside[600:1200, x:x + 250]] = 210
        vw.write(f)
    vw.release()
    job = post("/api/recognise", open("/tmp/v.mp4", "rb").read(), "video/mp4")["job"]
    for _ in range(300):
        j = json.load(urllib.request.urlopen(f"{URL}/api/job/{job}"))
        if j["status"] in ("done", "error"):
            break
        time.sleep(1)
    assert j["status"] == "done", j
    res = j["result"]
    assert res["climb"]["uuid"] == climb["uuid"], res["climb"]
    out = cv2.imdecode(np.frombuffer(urllib.request.urlopen(URL + "/" + res["image"]).read(), np.uint8), cv2.IMREAD_COLOR)
    mid = cv2.imdecode(np.frombuffer(cv2.imencode(".png", board)[1].tobytes(), np.uint8), cv2.IMREAD_COLOR)
    blob = lambda im: int(((im.min(axis=2) > 195) & inside)[120:].sum())   # light-grey pixels = climber
    single = board.copy(); single[600:1200, 500:750][inside[600:1200, 500:750]] = 210
    print(f"climber pixels: clean plate {blob(out)}, one raw frame would have {blob(single)}")
    assert blob(out) < 2000 < blob(single)
    cv2.imwrite("/tmp/last_video_result.jpg", out)
    print("OK: video result photo is climber-free")


if __name__ == "__main__":
    main()

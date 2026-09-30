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

import verify

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
    # Two hand holds get colours the app can't name from hue alone:
    #  - hue 30 (halfway between finish and hand): settled by height, this low on the board it can't be a finish
    #  - hue 129 (halfway between hand and foot): no height rule for that, so it stays undefined and takes the climb's role
    middles = [h for h in climb["holds"] if roles[h["role"]] == "middle"]
    low = next(h for h in middles if h["y"] < 100)
    odd = next(h for h in middles if h is not low)
    bgr129 = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[129, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
    special = {(low["x"], low["y"]): (0, 255, 255), (odd["x"], odd["y"]): bgr129}
    lit = [(x, y, special.get((x, y), c)) for x, y, c in lit]

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
    find = lambda p: next(h for h in res["holds"] if (h["x"], h["y"]) == (p["x"], p["y"]))
    assert find(odd)["role"] == "middle" and find(odd)["role_source"] == "climb", find(odd)   # role taken from the climb
    assert find(low)["role"] == "middle" and find(low)["role_source"] == "colour", find(low)  # settled by height, no conflict
    assert sum(h["role_source"] == "colour" for h in res["holds"]) >= 0.7 * len(res["holds"])
    print("OK: unclear colours: one settled by height, one taken from the climb")
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
    assert res["source"]["kind"] == "whole", res["source"]                       # nobody ever leaves, so no empty frame
    print("OK: climber never leaves -> falls back to the whole video:", res["source"]["note"])
    cv2.imwrite("/tmp/last_video_result.jpg", out)
    print("OK: video result photo is climber-free")

    # Video where the climber leaves the board for the last 30%: the lit holds must be read from an empty frame
    vw = cv2.VideoWriter("/tmp/v2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (1400, 1800))
    for i in range(frames_n):
        f = board.copy()
        if i < int(frames_n * 0.7):
            x = 100 + int(900 * i / frames_n)
            f[600:1200, x:x + 250][inside[600:1200, x:x + 250]] = 210
        vw.write(f)
    vw.release()
    job = post("/api/recognise", open("/tmp/v2.mp4", "rb").read(), "video/mp4")["job"]
    for _ in range(300):
        j = json.load(urllib.request.urlopen(f"{URL}/api/job/{job}"))
        if j["status"] in ("done", "error"):
            break
        time.sleep(1)
    assert j["status"] == "done", j
    res2 = j["result"]
    assert res2["source"]["kind"] == "empty", res2["source"]
    assert res2["climb"]["uuid"] == climb["uuid"], res2["climb"]
    assert all(t >= 0.42 * frames_n / 10 for t in res2["source"]["at"]), res2["source"]      # chosen after the climber left
    out2 = cv2.imdecode(np.frombuffer(urllib.request.urlopen(URL + "/" + res2["image"]).read(), np.uint8), cv2.IMREAD_COLOR)
    assert blob(out2) < 2000, blob(out2)
    names = {c["name"]: c for c in res2["verification"]["checks"]}
    assert names[verify.EMPTY_NAME]["status"] == "agrees", names[verify.EMPTY_NAME]
    assert names[verify.EMPTY_NAME]["images"] and names["Lit holds match the climb"]["images"]
    for c in res2["verification"]["checks"]:
        for im in c["images"]:
            assert urllib.request.urlopen(URL + "/" + im["url"]).read()[:2] == b"\xff\xd8", im
    assert "your calibration photo" in res2["source"]["note"], res2["source"]                # the photo gate was usable and used
    print("OK: read from an empty-board frame:", res2["source"]["note"])
    # No person in the synthetic video, so the real pose tracker should report it couldn't track anyone
    assert res["attempt"]["outcome"] in ("untracked", "not topped"), res["attempt"]
    print("pose tracker ran on the video:", res["attempt"])

    # Top-out judgement on synthetic wrist tracks (identity homography: pixels are inches)
    from server import judge
    from empty import body_on_board, empty_runs, spread_pick
    from detect_leds import assign_role, cap_finish, load_roles
    from match_climb import finish_odds, match

    # Role logic: clear colours are named, in-between ones are left undefined, at most two finish holds
    rl = load_roles()
    name = lambda hue: (lambda rid, clear: rl[rid]["name"] if clear else None)(*assign_role(hue, rl))
    assert name(5) == "finish" and name(150) == "foot" and name(90) in ("start", "middle")
    assert name(30) is None                                                   # halfway between finish and the hand colours
    lit3 = [{"hole_id": i, "role": 7, "role_name": "finish", "hue": h, "x": 0, "y": 140} for i, h in enumerate((20, 2, 8), 1)]
    kept = [h["hole_id"] for h in cap_finish(lit3, rl) if h["role_name"] == "finish"]
    assert sorted(kept) == [2, 3], kept                                       # the two closest to the finish colour
    assert lit3[0]["role_name"] is None and lit3[0]["role_uncertain"]
    # Search with two roles undefined: still finds the climb, and undefined roles don't count against it
    role_of = {int(k): v["name"] for k, v in data["roles"].items()}
    cl = [{"x": h["x"], "y": h["y"], "role_name": role_of[h["role"]]} for h in climb["holds"]]
    for h in cl[:2]:
        h["role_name"] = None
    top = match(cl, top=3)
    assert top[0]["uuid"] == climb["uuid"] and top[0]["roles_agree"] == len(cl) - 2, top[0]
    print("OK: role logic (undefined colours, finish cap, search with undefined roles)")

    # Where finish holds are: the database says they beat hand holds only in the top tenth and foot holds above ~100 in
    assert finish_odds(135, "hand") > 3 and finish_odds(60, "hand") < 0.05 and finish_odds(118, "hand") < 0.2
    assert finish_odds(120, "foot") > 3 and finish_odds(40, "foot") < 0.05
    at = lambda hue, y: (lambda rid, clear: rl[rid]["name"] if clear else None)(*assign_role(hue, rl, y))
    assert at(30, 138) == "finish"                          # finish/hand colour undecided, very top of the board
    assert at(30, 60) in ("start", "middle") and at(30, 118) in ("start", "middle")
    assert at(165, 120) == "finish" and at(165, 40) == "foot" and at(165, 95) is None   # finish/foot: settled at both ends, open between
    assert at(90, 5) in ("start", "middle") and at(5, 20) == "finish"                   # a clear colour is trusted at any height
    low = [{"hole_id": 1, "role": 7, "role_name": "finish", "hue": 0, "x": 0, "y": 20}, {"hole_id": 2, "role": 7, "role_name": "finish", "hue": 6, "x": 0, "y": 135},
           {"hole_id": 3, "role": 7, "role_name": "finish", "hue": 8, "x": 8, "y": 138}]
    assert [h["hole_id"] for h in cap_finish(low, rl) if h["role_name"] == "finish"] == [2, 3]   # the low one goes, despite the best hue
    print("OK: finish holds are judged by height as well as colour")

    # Two finish holds can't be far apart: the limit comes from the database (about 46 in)
    from match_climb import break_ties, colour_agreement, finish_pair_limit, role_class, role_hues
    lim = finish_pair_limit()
    assert 30 < lim < 70, lim
    far = [{"hole_id": 1, "role": 7, "role_name": "finish", "hue": 2, "x": -40, "y": 138}, {"hole_id": 2, "role": 7, "role_name": "finish", "hue": 3, "x": 40, "y": 138}]
    assert [h["hole_id"] for h in cap_finish(far, rl) if h["role_name"] == "finish"] == [1]        # 80 in apart: only one stays
    near = [{"hole_id": 1, "role": 7, "role_name": "finish", "hue": 2, "x": 0, "y": 138}, {"hole_id": 2, "role": 7, "role_name": "finish", "hue": 3, "x": 16, "y": 138}]
    assert len([h for h in cap_finish(near, rl) if h["role_name"] == "finish"]) == 2               # 16 in apart: both stay
    print("OK: two finish holds must be close together (limit %.0f in)" % lim)

    # Climbs with identical holds but different roles: positions can't separate them, colours then do
    groups = {}
    for c in data["climbs"]:
        groups.setdefault(frozenset((h["x"], h["y"]) for h in c["holds"]), []).append(c)
    rolemap = lambda c: frozenset((h["x"], h["y"], role_class(role_of[h["role"]])) for h in c["holds"])
    pair = next(((g[0], k) for g in groups.values() if len(g) > 1 for k in g[1:] if rolemap(k) != rolemap(g[0]) and len(g[0]["holds"]) >= 8), None)
    if pair:
        a_c, b_c = pair
        hue_of = {"hand": rl[next(r for r in rl if rl[r]["name"] == "middle")]["hue"], "foot": rl[next(r for r in rl if rl[r]["name"] == "foot")]["hue"],
                  "finish": rl[next(r for r in rl if rl[r]["name"] == "finish")]["hue"]}
        lit_a = [{"x": h["x"], "y": h["y"], "hue": hue_of[role_class(role_of[h["role"]])], "role_name": None} for h in a_c["holds"]]
        top = match(lit_a, top=10)
        assert top[0]["jaccard"] == 1.0 and top[1]["jaccard"] == 1.0                                # positions alone can't separate them
        top = break_ties(top, lit_a, role_hues(rl))
        assert rolemap(next(c for c in data["climbs"] if c["uuid"] == top[0]["uuid"])) == rolemap(a_c), "colours should pick the climb whose roles fit"
        print("OK: colours only break ties between climbs with the same holds")

    # Empty-moment selection and the pose gate
    assert empty_runs([1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1], 3) == [[3, 4, 5], [7, 8, 9, 10]]      # short run dropped
    assert empty_runs([0, 0, 1, 1], 3) == []
    assert spread_pick([[0, 1, 2], [10, 11, 12, 13]], 3) == [0, 11, 13] or len(spread_pick([[0, 1, 2], [10, 11, 12, 13]], 3)) == 3
    picks = spread_pick([list(range(0, 50)), list(range(100, 150))], 7)
    assert len(picks) == 7 and picks[0] == 0 and picks[-1] == 149 and any(p >= 100 for p in picks) and any(p < 50 for p in picks)
    outline = np.zeros((100, 100), np.uint8)
    outline[20:80, 20:80] = 255
    assert body_on_board([(50, 50, 0.9)], outline)                  # a hand on the board
    assert not body_on_board([(50, 50, 0.1)], outline)              # too faint to count
    assert not body_on_board([(5, 5, 0.9)], outline)                # someone standing beside it
    assert not body_on_board([], outline)                           # nobody in shot
    print("OK: empty-moment selection and pose gate")
    holds = [{"hole_id": i, "x": x, "y": y, "role_name": r} for i, (x, y, r) in
             enumerate([(0, 0, "start"), (1, 10, "middle"), (0, 20, "middle"), (1, 30, "finish")], 1)]

    def climb_to(n):
        pts = [(0, 0), (1, 10), (0, 20), (1, 30)][:n]
        return [{"t": 0.1 * (4 * k + j), "landmarks": {"right_wrist": (x, y, 0.99)}}
                for k, (x, y) in enumerate(pts) for j in range(4)]
    I = np.eye(3)
    assert judge(climb_to(4), I, holds)["outcome"] == "topped"
    assert judge(climb_to(3), I, holds)["outcome"] == "not topped"
    assert judge([], I, holds)["outcome"] == "untracked"
    print("OK: judge distinguishes topped / not topped / untracked")
    ver = res["verification"]
    by_name = {c["name"]: c for c in ver["checks"]}
    assert by_name[verify.SPLIT_NAME]["status"] == "agrees", by_name[verify.SPLIT_NAME]   # LEDs are constant
    assert len(by_name[verify.SPLIT_NAME]["images"]) == 2 and len(by_name["Lit holds match the climb"]["images"]) == 1
    for c in ver["checks"]:
        for im in c["images"]:
            assert urllib.request.urlopen(URL + "/" + im["url"]).read()[:2] == b"\xff\xd8", im
    assert by_name[verify.BODY_NAME]["status"] == "not available"                         # no climber to track
    assert by_name[verify.START_NAME]["status"] == "not available"
    assert ver["level"] == "unconfirmed", ver                                              # 2 of 2 available agree, but too few to confirm
    print("cross-check summary:", ver["level"], "-", ver["text"])
    for c in ver["checks"]:
        print("  ", c["status"], "|", c["name"], "|", c["detail"])

    # Body checks on synthetic tracks: one climb's route, and a rival that shares only some holds
    a_holds = holds
    b_holds = [dict(h, hole_id=h["hole_id"] + 10, x=h["x"] + 20) for h in holds]        # a different route
    cands = [{"uuid": "A", "name": "A", "holds": a_holds}, {"uuid": "B", "name": "B", "holds": b_holds}]
    on_a = [{"t": f["t"], "points": {"right_wrist": f["landmarks"]["right_wrist"][:2]}} for f in climb_to(4)]
    assert verify.body_on_route(on_a, cands, "A")["status"] == "agrees"
    assert verify.body_on_route(on_a, cands, "B")["status"] == "disagrees"              # body follows A, not B
    assert verify.body_on_route(on_a[:3], cands, "A")["status"] == "not available"      # too little tracking
    assert verify.start_and_finish(on_a, a_holds, True)["status"] == "agrees"
    i, on, n = verify.best_route_frame(on_a, cands[0]["holds"])
    assert (on, n) == (1, 1) and i == 0
    s_i, f_i = verify.start_finish_frames(on_a, a_holds)
    assert s_i == 0 and f_i == len(on_a) - 1 and verify.start_finish_frames(on_a[4:8], a_holds) == (None, None)
    assert verify.start_and_finish(on_a[8:], a_holds, False)["status"] == "disagrees"   # began mid-route, never finished
    assert verify.start_and_finish([], a_holds, False)["status"] == "not available"
    lost = [{"t": 0.1 * i, "points": {}} for i in range(10)]                            # tracker found nobody
    assert verify.start_and_finish(lost, a_holds, False)["status"] == "not available"
    led, half, body = (verify.check(n, "agrees", "") for n in (verify.SPLIT_NAME, verify.SPLIT_NAME, verify.BODY_NAME))
    assert verify.summary([led, half])["level"] == "unconfirmed"           # two LED-based checks aren't independent
    assert verify.summary([led, half, body])["level"] == "confirmed"       # lit holds + body agree
    assert verify.summary([led, verify.check(verify.BODY_NAME, "disagrees", "")])["level"] == "double-check"
    print("OK: cross-checks agree/disagree as expected on synthetic tracks")


if __name__ == "__main__":
    main()

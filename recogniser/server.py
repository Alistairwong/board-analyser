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

from autocalibrate import auto_calibrate, manual_references
from calibrate import CALIB_DIR, DB_PATH, corner_holes, grab_frame, load_holes
from config import DATA_DIR
from detect_leds import detect, hole_patches, load_roles
from match_climb import full_climb_holds, grade_text, match
from recognise import GOOD, assess

HERE = Path(__file__).resolve().parent
OUT = DATA_DIR / "recogniser"
MAX_UPLOAD = 1024 * 1024 * 1024
VIDEO_FPS = 3                     # frames per second analysed from a video
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
VIDEO_EXT = {"video/mp4": "mp4", "video/quicktime": "mov", "video/webm": "webm"}

JOBS = {}                         # id -> {"status", "progress", "step", "result", "error"}
WORK = threading.Lock()           # one recognition at a time: OpenCV/SIFT is memory hungry


# ---- recognition ---------------------------------------------------------
def photo_to_clip(photo_bytes, clip):
    """Everything downstream reads videos, so a photo becomes a 2 s clip of one frame.
    Decoding with OpenCV applies the phone's EXIF rotation, as the browser does."""
    img = cv2.imdecode(np.frombuffer(photo_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("That image couldn't be read.")
    png = clip.with_suffix(".png")
    cv2.imwrite(str(png), img)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png), "-t", "2", "-r", "5",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                    "-crf", "10", str(clip)], check=True)   # crf 10: keep LED colours intact
    png.unlink()


def clean_plate(clip, n=15):
    """The board with the climber removed: the per-pixel median of n frames spread across the clip.
    The camera is fixed, so anything that moves (the climber) drops out and the board, the LEDs and
    every hold stay. ponytail: fails if the climber covers the same pixels for over half the clip,
    and needs a fixed camera; a handheld clip would need the frames aligned first."""
    cap = cv2.VideoCapture(str(clip))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    want = set(np.linspace(0, total - 1, min(n, total)).astype(int).tolist())
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


def render(clip, is_photo, calib, holds, extras, climb, verdict, jid, roles):
    """The photo (or, for a video, the climber-free clean plate) with the climb's holds circled and labelled."""
    frame = grab_frame(clip, calib["time"]) if is_photo else clean_plate(clip)
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
        if h["seen"]:
            label(c, f"{round(h['confidence'] * 100)}%", colour)

    if climb:
        title = f"{climb['name']}  {climb['grade_v'] or '?'}" + ("" if verdict in GOOD else f"  ({verdict})")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], int(48 * font * 2)), (0, 0, 0), -1)
        cv2.putText(frame, title, (12, int(34 * font * 2 * 0.75)), cv2.FONT_HERSHEY_SIMPLEX, font * 1.6,
                    (255, 255, 255), thick, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"{jid}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])


def recognise(jid, clip, is_photo):
    job = JOBS[jid]

    def step(text, lo, hi):
        job["step"] = text
        return lambda f: job.__setitem__("progress", round(lo + (hi - lo) * f, 3))

    info = {}
    job["step"] = "Aligning to your calibration"
    cal_name = auto_calibrate(clip, name=f"auto-{jid}", verbose=False, progress=step("Aligning to your calibration", 0, 0.3), info=info)
    if not cal_name:
        if not manual_references():
            raise ValueError("There's no reference calibration yet. Make one from the Calibrate tab first.")
        raise ValueError(f"Couldn't line this up with the board (only {info.get('features', 0)} matching features, need 40). "
                         "Try a sharper, brighter photo with the whole board in view.")
    calib = json.loads((CALIB_DIR / f"{cal_name}.json").read_text())

    det = detect(clip, cal_name, fps=VIDEO_FPS, verbose=False, images=False,
                 progress=step("Finding the lit holds", 0.3, 0.9))
    job["step"], job["progress"] = "Matching against the climb database", 0.92
    lit = det["lit"]
    top = match(lit, top=5) if lit else []
    verdict, ties = assess(top)
    best = top[0] if top else None

    roles = load_roles()
    thr = det["threshold"]
    lit_by_id = {h["hole_id"]: h for h in lit}
    if best and verdict in GOOD:
        wanted = full_climb_holds(best, {(x, y): hid for hid, _, x, y in load_holes(sqlite3.connect(DB_PATH))})
    else:
        wanted = [{"hole_id": h["hole_id"], "x": h["x"], "y": h["y"], "role_name": h.get("role_name")} for h in lit]
    holds = []
    for w in wanted:
        seen = lit_by_id.get(w["hole_id"])
        holds.append({"hole_id": w["hole_id"], "x": w["x"], "y": w["y"], "role": w["role_name"],
                      "seen": bool(seen), "score": seen["score"] if seen else 0,
                      "confidence": strength(seen["score"], thr) if seen else 0.0})
    holds.sort(key=lambda h: (-h["y"], h["x"]))
    in_climb = {h["hole_id"] for h in holds}
    extras = [h for h in lit if h["hole_id"] not in in_climb]

    climb = climb_details(best) if best else None
    render(clip, is_photo, calib, holds, extras, climb, verdict, jid, roles)

    margin = best["jaccard"] - (top[1]["jaccard"] if len(top) > 1 else 0) if best else 0
    job["result"] = {
        "id": jid, "image": f"out/{jid}.jpg", "kind": "photo" if is_photo else "video",
        "verdict": verdict, "ties": ties, "detected": len(lit),
        "match": round(best["jaccard"], 3) if best else 0, "margin": round(margin, 3),
        "climb": climb, "holds": holds,
        "not_in_climb": [{"x": h["x"], "y": h["y"], "role": h.get("role_name")} for h in extras],
        "candidates": [{"name": t["name"], "match": round(t["jaccard"], 3), "grade": t.get("v"),
                        "mirrored": t["mirrored"]} for t in top],
        "alignment_features": calib.get("matched_features"),
    }
    (OUT / f"{jid}.json").write_text(json.dumps(job["result"], indent=2))
    for f in CALIB_DIR.glob(f"{cal_name}*"):          # the per-photo alignment isn't worth keeping
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
            for f in clip.parent.glob(clip.stem + ".*"):
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
        if m := re.fullmatch(r"/out/([0-9a-f]{12})\.jpg", path):
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

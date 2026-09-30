"""Video -> movement analysis: which holds each limb used, timing/effort metrics, technique flags,
and an annotated video (skeleton + holds). Runs in Docker beside the recogniser and reuses video/'s code
(the Dockerfile copies it in). Needs the reference calibration made in the recogniser's Calibrate tab.
"""
import json
import re
import sqlite3
import subprocess
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from autocalibrate import auto_calibrate, manual_references
from calibrate import CALIB_DIR, DB_PATH, load_holes
from config import DATA_DIR
from detect_leds import detect
from match_climb import full_climb_holds, grade_text, match
from movement import analyse
from recognise import GOOD, assess
import render_movement

HERE = Path(__file__).resolve().parent
OUT = DATA_DIR / "movement"
MAX_UPLOAD = 1024 * 1024 * 1024
CHUNK_MAX = 60 * 1024 * 1024        # pieces are 50 MB; the tunnel refuses requests over 100 MB
POSE_FPS = 10
VIDEO_EXT = {"video/mp4": "mp4", "video/quicktime": "mov", "video/webm": "webm"}
JOBS = {}                         # id -> {"status", "progress", "step", "result", "error"}
WORK = threading.Lock()           # one at a time: pose + SIFT are memory hungry


def analyse_video(jid, clip):
    from pose import estimate_landmarks, to_board_inches
    job = JOBS[jid]

    def step(text, lo, hi):
        job["step"] = text
        return lambda f: job.__setitem__("progress", round(lo + (hi - lo) * f, 3))

    name = f"auto-{jid}"
    info = {}
    if not auto_calibrate(clip, name=name, verbose=False, progress=step("Aligning to your calibration", 0, 0.1), info=info):
        raise ValueError("Make a reference calibration in the recogniser's Calibrate tab first." if not manual_references() else
                         f"Couldn't line this up with the board (only {info.get('features', 0)} matching features, need 40).")
    calib = json.loads((CALIB_DIR / f"{name}.json").read_text())
    H = calib["homography"]

    job["step"], job["progress"] = "Finding the climb", 0.12
    det = detect(clip, name, verbose=False)
    if not det["lit"]:
        raise ValueError("No lit holds found, so the climb can't be identified.")
    top = match(det["lit"], top=5)
    verdict, _ = assess(top)
    best = top[0] if top else None
    holds = full_climb_holds(best, {(x, y): hid for hid, _, x, y in load_holes(sqlite3.connect(DB_PATH))}) \
        if best and verdict in GOOD else det["lit"]     # ponytail: unsure match -> just the detected lit holds

    frames = estimate_landmarks(clip, fps=POSE_FPS, progress=step("Tracking the climber", 0.15, 0.6))
    import numpy as np
    track = to_board_inches(np.array(H, dtype=np.float64), frames)
    if sum(1 for f in track if any(k.endswith("wrist") for k in f["points"])) < 3:
        raise ValueError("Couldn't find a climber in the video.")
    result = analyse(track, holds)

    job["step"], job["progress"] = "Drawing the annotated video", 0.62
    raw = render_movement.render(clip, name, POSE_FPS, verbose=False, holds=holds, frames=frames)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(OUT / f"{jid}.mp4")], check=True)   # browsers can't play mp4v
    raw.unlink()

    at = {h["hole_id"]: f"{h['x']:g},{h['y']:g}" for h in holds}
    for m in result["moves"]:
        m["from"], m["to"] = at.get(m["from_hold"], "off"), at.get(m["to_hold"], "off")
    job["result"] = {
        "id": jid, "video": f"out/{jid}.mp4", "verdict": verdict,
        "climb": {"name": best["name"], "grade": grade_text(best), "mirrored": best["mirrored"],
                  "match": round(best["jaccard"], 3)} if best else None,
        "outcome": "topped" if result["attempt"]["outcome"] == "topped" else "not topped", "metrics": result["metrics"],
        "flags": result["flags"], "comments": result["comments"], "moves": result["moves"],
        "pause_at": at.get(result["metrics"]["longest_pause_hold"]),
    }
    (OUT / f"{jid}.json").write_text(json.dumps(job["result"], indent=2))


def run_job(jid, clip):
    with WORK:
        JOBS[jid].update(status="running", progress=0.0)
        try:
            analyse_video(jid, clip)
            JOBS[jid]["status"] = "done"
        except BaseException as e:                    # SystemExit too: shown to the user, so keep it readable
            JOBS[jid].update(status="error", error=str(e) or type(e).__name__)
        finally:
            JOBS[jid]["progress"] = 1.0
            clip.unlink(missing_ok=True)
            for f in CALIB_DIR.glob(f"auto-{jid}*"):
                f.unlink()


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, ctype="application/json", headers=()):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for h in headers:
            self.send_header(*h)
        self.end_headers()
        self.wfile.write(body)

    def video(self, path):
        """mp4 with Range support (Safari won't play a video without it). ponytail: reads the whole file, fine for short clips."""
        if not path.is_file():
            return self.reply(404, {"error": "not found"})
        data, total = path.read_bytes(), path.stat().st_size
        if m := re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "")):
            a = int(m[1]) if m[1] else max(0, total - int(m[2] or 0))
            b = min(int(m[2]), total - 1) if m[1] and m[2] else total - 1
            return self.reply(206, data[a:b + 1], "video/mp4", [("Content-Range", f"bytes {a}-{b}/{total}"), ("Accept-Ranges", "bytes")])
        self.reply(200, data, "video/mp4", [("Accept-Ranges", "bytes")])

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self.reply(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if m := re.fullmatch(r"/api/job/([0-9a-f]{12})", path):
            job = JOBS.get(m[1])
            return self.reply(200 if job else 404, job or {"error": "unknown job"})
        if m := re.fullmatch(r"/out/([0-9a-f]{12})\.mp4", path):
            return self.video(OUT / f"{m[1]}.mp4")
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        """Chunked upload: Cloudflare caps a request at 100 MB, so the page sends the file in 50 MB pieces
        to /api/chunk?id=<12 hex>&i=<n>&last=<0|1>. Piece 0 starts the file; the last piece starts the analysis."""
        url = urlparse(self.path)
        q = parse_qs(url.query)
        ext = VIDEO_EXT.get(self.headers.get("Content-Type", "").split(";")[0].strip().lower())
        length = self.headers.get("Content-Length", "")
        jid, i = (q.get("id") or [""])[0], (q.get("i") or [""])[0]
        if url.path != "/api/chunk" or not re.fullmatch(r"[0-9a-f]{12}", jid) or not i.isdigit():
            return self.reply(404, {"error": "not found"})
        if not ext:
            return self.reply(415, {"error": "Upload an mp4, mov or webm video."})
        if not length.isdigit() or not 0 < int(length) <= CHUNK_MAX:
            return self.reply(413, {"error": "empty or too large a piece"})
        clip = OUT / f"{jid}_in.{ext}"
        if int(i) == 0 and jid in JOBS:
            return self.reply(409, {"error": "id already used"})
        if int(i) and not clip.exists():
            return self.reply(400, {"error": "missing first piece"})
        if int(i) and clip.stat().st_size + int(length) > MAX_UPLOAD:
            return self.reply(413, {"error": "file too large"})
        with clip.open("wb" if int(i) == 0 else "ab") as f:
            f.write(self.rfile.read(int(length)))
        if (q.get("last") or ["0"])[0] == "1":
            JOBS[jid] = {"status": "queued", "progress": 0.0, "step": "Waiting", "result": None, "error": None}
            threading.Thread(target=run_job, args=(jid, clip), daemon=True).start()
        self.reply(200, {"job": jid})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("Movement on :8000", flush=True)
    ThreadingHTTPServer(("", 8000), Handler).serve_forever()

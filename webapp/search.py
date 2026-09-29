"""Open the climb search page in your browser.

Usage (from anywhere):
    python webapp/search.py                  # http://localhost:8000
    python webapp/search.py --port 8080
    python webapp/search.py --host 0.0.0.0   # let other devices on your network use it
    python webapp/search.py --reexport       # rebuild the climb data first

    python webapp/search.py --sync           # update the climb database from Tension first
    python webapp/search.py --sync-only      # update the database and exit (for a scheduled job)

The climb data is rebuilt automatically when tension.db or the climbs file
changes, even while the server is running. Press Ctrl+C to stop the server.
"""
import argparse
import gzip
import http.client
import http.server
import json
import os
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent

from config import CLIMBS_PATH, DATA_DIR
from export_search import DB_PATH, OUT, export

PAGE = HERE / "search" / "index.html"
RECORD_PAGE = HERE / "search" / "record.html"
RECORDINGS = DATA_DIR / "recordings"
UPLOAD_TYPES = {"video/mp4": "mp4", "video/webm": "webm", "video/quicktime": "mov"}
MAX_UPLOAD = 200 * 1024 * 1024
STATIC = HERE / "search"
# The photo/video recogniser is a separate container (it needs OpenCV); /recogniser/* is passed through to it.
RECOGNISER = urlparse(os.environ.get("RECOGNISER_URL", "http://host.docker.internal:8020"))
LOAD_CLIMBS = HERE / "load_climbs.py"
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".json": "application/json",
                ".webp": "image/webp", ".png": "image/png", ".jpg": "image/jpeg",
                ".svg": "image/svg+xml", ".js": "text/javascript; charset=utf-8",
                ".webmanifest": "application/manifest+json"}
_cache = {}
_export_lock = threading.Lock()


def sync_database():
    """Download or update tension.db with BoardLib, then rebuild the climb list.

    New climbs only come through when logged in, using TENSION_USERNAME and
    TENSION_PASSWORD from the environment (the server's .env file).
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["boardlib", "database", "tension", str(DB_PATH)]
    user = os.environ.get("TENSION_USERNAME", "").strip()
    password = os.environ.get("TENSION_PASSWORD", "")
    if user:
        cmd += ["--username", user]
        print(f"Syncing the Tension climb database as {user}...")
    else:
        print("No TENSION_USERNAME set, so only the climbs bundled with the app are available.")
    # BoardLib asks for the password at a prompt; answer it from the environment
    subprocess.run(cmd, check=True, text=True, input=(password + "\n") if user else None)
    subprocess.run([sys.executable, str(LOAD_CLIMBS)], check=True)


def ensure_data():
    if not DB_PATH.exists():
        sync_database()
    elif not CLIMBS_PATH.exists():
        subprocess.run([sys.executable, str(LOAD_CLIMBS)], check=True)


def export_if_needed():
    with _export_lock:
        ensure_data()             # rebuilds the climbs file if it has gone missing
        if needs_export():
            export()


def needs_export():
    if not OUT.exists():
        return True
    # rebuild when the data changes, or when the export itself has been updated
    newest_source = max(DB_PATH.stat().st_mtime, CLIMBS_PATH.stat().st_mtime,
                        (HERE / "export_search.py").stat().st_mtime)
    return OUT.stat().st_mtime < newest_source


def load(path):
    """File contents, plus a gzipped copy, cached until the file changes."""
    mtime = path.stat().st_mtime
    if path not in _cache or _cache[path][0] != mtime:
        raw = path.read_bytes()
        _cache[path] = (mtime, raw, gzip.compress(raw))
    return _cache[path][1], _cache[path][2]


class Handler(http.server.BaseHTTPRequestHandler):
    ROUTES = {
        "/": (PAGE, "text/html; charset=utf-8"),
        "/index.html": (PAGE, "text/html; charset=utf-8"),
        "/record": (RECORD_PAGE, "text/html; charset=utf-8"),
        "/climbs.json": (OUT, "application/json"),
    }

    def route(self, path):
        if path in self.ROUTES:
            return self.ROUTES[path]
        name = path.lstrip("/")
        f = STATIC / name
        if "/" not in name and ".." not in name and f.suffix in STATIC_TYPES:
            return f, STATIC_TYPES[f.suffix]
        return None

    def proxy(self):
        """Pass /recogniser/... through to the recogniser container, streaming both directions."""
        path = self.path[len("/recogniser"):]
        if not path:
            self.send_response(301)
            self.send_header("Location", "/recogniser/")
            self.send_header("Content-Length", "0")
            return self.end_headers()
        if path[0] not in "/?":
            return self.send_error(404)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            conn = http.client.HTTPConnection(RECOGNISER.hostname, RECOGNISER.port, timeout=600)
            conn.putrequest(self.command, path if path[0] == "/" else "/" + path)
            if self.headers.get("Content-Type"):
                conn.putheader("Content-Type", self.headers["Content-Type"])
            conn.putheader("Content-Length", str(length))
            conn.endheaders()
            left = length
            while left:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    break
                conn.send(chunk)
                left -= len(chunk)
            resp = conn.getresponse()
        except OSError:
            return self.send_error(502, "The recogniser isn't running")
        self.send_response(resp.status)
        for h in ("Content-Type", "Content-Length"):
            if resp.getheader(h):
                self.send_header(h, resp.getheader(h))
        self.end_headers()
        while chunk := resp.read(1 << 20):
            self.wfile.write(chunk)
        conn.close()

    def do_GET(self):
        if self.path.startswith("/recogniser"):
            return self.proxy()
        path = self.path.split("?")[0]
        if path == "/climbs.json":
            export_if_needed()        # picks up a database sync without a restart
        route = self.route(path)
        if not route or not route[0].exists():
            self.send_error(404)
            return
        raw, zipped = load(route[0])
        use_gzip = "gzip" in self.headers.get("Accept-Encoding", "")
        body = zipped if use_gzip else raw
        self.send_response(200)
        self.send_header("Content-Type", route[1])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        """Save an uploaded clip from /record into data/recordings/ (name is server-made)."""
        if self.path.startswith("/recogniser"):
            return self.proxy()
        if self.path.split("?")[0] != "/upload":
            return self.send_error(404)
        ext = UPLOAD_TYPES.get(self.headers.get("Content-Type", "").split(";")[0].strip().lower())
        length = self.headers.get("Content-Length", "")
        if not ext:
            return self.send_error(415)
        if not length.isdigit() or int(length) == 0:
            return self.send_error(411)
        length = int(length)
        if length > MAX_UPLOAD:
            return self.send_error(413)
        RECORDINGS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest, n = RECORDINGS / f"{stamp}.{ext}", 0
        while dest.exists():
            n += 1
            dest = RECORDINGS / f"{stamp}_{n}.{ext}"
        part = dest.with_name(dest.name + ".part")
        got = 0
        with part.open("wb") as f:
            while got < length:
                chunk = self.rfile.read(min(1 << 20, length - got))
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
        if got != length:                 # client dropped mid-upload
            part.unlink()
            return self.send_error(400)
        part.rename(dest)
        body = json.dumps({"name": dest.name, "bytes": got}).encode()
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--reexport", action="store_true", help="rebuild the climb data first")
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    ap.add_argument("--sync", action="store_true", help="update the climb database first")
    ap.add_argument("--sync-only", action="store_true", help="update the climb database and exit")
    args = ap.parse_args()

    if args.sync or args.sync_only:
        sync_database()
        if args.sync_only:
            export_if_needed()
            return
    ensure_data()

    if not PAGE.exists():
        print(f"Can't find the search page. It should be at:\n  {PAGE}")
        found = list(HERE.rglob("index*.htm*"))
        if found:
            print("Found these instead; move or rename one to the path above:")
            for f in found:
                print(f"  {f}")
        raise SystemExit(1)

    if args.reexport:
        export()
    else:
        export_if_needed()

    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://localhost:{args.port}"
    sys.stdout.flush()
    print(f"Climb search running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

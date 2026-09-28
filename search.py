"""Open the climb search page in your browser.

Usage:
    python search.py                  # http://localhost:8000
    python search.py --port 8080
    python search.py --host 0.0.0.0   # let other devices on your network use it
    python search.py --reexport       # rebuild the climb data first

    python search.py --sync           # update the climb database from Tension first
    python search.py --sync-only      # update the database and exit (for a scheduled job)

The climb data is rebuilt automatically when tension.db or the climbs file
changes, even while the server is running. Press Ctrl+C to stop the server.
"""
import argparse
import gzip
import http.server
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

# Work from the project folder, wherever the command is run from
PROJECT = Path(__file__).resolve().parent
os.chdir(PROJECT)

from config import CLIMBS_PATH
from export_search import DB_PATH, OUT, export

PAGE = Path("search/index.html")
STATIC = Path("search")
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
    subprocess.run([sys.executable, "load_climbs.py"], check=True)


def ensure_data():
    if not DB_PATH.exists():
        sync_database()
    elif not Path(CLIMBS_PATH).exists():
        subprocess.run([sys.executable, "load_climbs.py"], check=True)


def export_if_needed():
    with _export_lock:
        if needs_export():
            export()


def needs_export():
    if not OUT.exists():
        return True
    newest_source = max(DB_PATH.stat().st_mtime, Path(CLIMBS_PATH).stat().st_mtime)
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

    def do_GET(self):
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
        print(f"Can't find the search page. It should be at:\n  {PROJECT / PAGE}")
        found = [f for f in PROJECT.rglob("index*.htm*") if ".venv" not in f.parts]
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

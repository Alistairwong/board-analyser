"""Work through every clip in the backlog folder, one at a time.

For each clip:
  1. Auto-calibrate: find the board by matching holds against a hand-made
     calibration.
  2. Detect the lit LEDs.
  3. If calibration or detection fails, move the clip to data/manual/calibration
     or data/manual/detection for a manual look.
  4. Match the climb, report its details and confidence, and move the clip
     to data/processed/.

After all clips, a session check looks at clips in recording order. Repeated
attempts at the same climb usually come in a row, so an uncertain clip is
accepted if a confidently recognised clip right next to it in the same
session is the same climb (and that climb also fits the uncertain clip
reasonably well), or if 3+ uncertain clips in a row share the same best guess.

Usage:
    python backlog.py                          # processes data/backlog/
    python backlog.py --folder D:/old-clips    # a different folder
    python backlog.py --keep                   # report only, don't move files

Results go in data/results/, plus a spreadsheet of every recognised clip so
far: data/results/backlog.csv
"""
import argparse
import csv
import json
import os
import shutil
import struct
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autocalibrate import auto_calibrate
from detect_leds import detect
from recognise import RESULTS_DIR, finish, grade_text

VIDEO_TYPES = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
GOOD = {"confident", "likely", "tied"}
SESSION = {"session", "repeated"}        # accepted by the session check
ACCEPTED = GOOD | SESSION
MAX_GAP = timedelta(minutes=15)         # longer gap between clips = new session
MIN_SUPPORT = 0.5                       # a clip must overlap the climb at least this much
RUN_LENGTH = 3                          # uncertain clips in a row with the same guess
BACKLOG_DIR = Path("data/backlog")
PROCESSED_DIR = Path("data/processed")
MANUAL_DIR = Path("data/manual")
CSV_PATH = RESULTS_DIR / "backlog.csv"
BAR_WIDTH = 20


# ---------- progress display ----------

def bar(frac):
    filled = int(round(frac * BAR_WIDTH))
    return "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + "]"


def fmt_time(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


class Step:
    """One line per step, with a live bar while it runs."""

    def __init__(self, n, label):
        self.prefix = f"  {n}/4 {label:<18}"
        self.start = time.time()
        self.update(0)

    def update(self, frac):
        print(f"\r{self.prefix}{bar(frac)} {frac:4.0%}", end="", flush=True)

    def done(self, note=""):
        took = fmt_time(time.time() - self.start)
        print(f"\r{self.prefix}{bar(1.0)} done ({took}){'  ' + note if note else ''}   ")

    def fail(self, note):
        print(f"\r{self.prefix}failed: {note}" + " " * 30)


# ---------- file handling ----------

def move(video, folder):
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / video.name
    n = 1
    while dest.exists():                     # never overwrite another clip
        dest = folder / f"{video.stem}_{n}{video.suffix}"
        n += 1
    shutil.move(str(video), str(dest))
    return dest


def mov_creation_time(path):
    """Recording time stored inside .mov/.mp4 files (UTC), or None."""
    size_total = os.path.getsize(path)
    with open(path, "rb") as f:
        pos = 0
        while pos + 8 <= size_total:
            f.seek(pos)
            size, kind = struct.unpack(">I4s", f.read(8))
            header = 8
            if size == 1:
                size, header = struct.unpack(">Q", f.read(8))[0], 16
            elif size == 0:
                size = size_total - pos
            if size < 8:
                return None
            if kind == b"moov":
                p, end = pos + header, pos + size
                while p + 8 <= end:
                    f.seek(p)
                    s2, k2 = struct.unpack(">I4s", f.read(8))
                    if k2 == b"mvhd":
                        version = f.read(4)[0]
                        fmt, n = (">Q", 8) if version == 1 else (">I", 4)
                        secs = struct.unpack(fmt, f.read(n))[0]
                        return datetime(1904, 1, 1) + timedelta(seconds=secs) if secs else None
                    if s2 < 8:
                        return None
                    p += s2
                return None
            pos += size
    return None


def recorded_at(path):
    """When the clip was filmed: from the file's own metadata, else its modified time."""
    try:
        t = mov_creation_time(path)
    except (OSError, struct.error):
        t = None
    if t is None:
        t = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(tzinfo=None)
    return t


def save_result(video_stem, result):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{video_stem}.json").write_text(json.dumps(result, indent=2))


def write_csv():
    """Spreadsheet of every recognised clip across all runs."""
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if r.get("verdict") in ACCEPTED and r.get("best"):
            rows.append(r)
    fields = ["video", "recorded", "climb", "mirrored", "v_grade", "font_grade", "ascents", "stars",
              "benchmark", "match", "verdict", "ties", "calibration"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            b = r["best"]
            w.writerow({
                "video": Path(r["video"]).name,
                "recorded": r.get("recorded_at", ""),
                "climb": b.get("name", ""),
                "mirrored": b.get("mirrored", ""),
                "v_grade": b.get("v") or "",
                "font_grade": b.get("font") or "",
                "ascents": b.get("ascents", ""),
                "stars": b.get("stars") if b.get("stars") is not None else "",
                "benchmark": b.get("benchmark", ""),
                "match": f"{b['jaccard']:.0%}",
                "verdict": r.get("verdict", ""),
                "ties": r.get("ties", ""),
                "calibration": r.get("calibration", ""),
            })
    return len(rows)


# ---------- one clip ----------

def process(video, keep):
    """Run the four steps on one clip. Returns (outcome, result)."""
    # 1. auto-calibrate
    step = Step(1, "Auto-calibrating")
    info = {}
    try:
        calib = auto_calibrate(video, verbose=False, progress=step.update, info=info)
    except Exception as e:
        calib, info["error"] = None, str(e)
    if not calib:
        if info.get("error"):
            why = info["error"]
        elif info.get("frames_read") == 0:
            why = "couldn't read the video"
        else:
            why = f"only {info.get('features', 0)} matching features"
        step.fail(why)
        return "calibration", {"video": str(video), "verdict": "needs calibration", "reason": why}
    step.done(f"{info.get('features', '?')} matching features")

    # 2. detect LEDs
    step = Step(2, "Detecting LEDs")
    try:
        det = detect(video, calib, verbose=False, progress=step.update)
    except Exception as e:
        step.fail(str(e))
        return "detection", {"video": str(video), "verdict": "detection failed", "reason": str(e)}
    if not det["lit"]:
        step.fail("no lit holds found")
        return "detection", {"video": str(video), "verdict": "detection failed",
                             "reason": "no lit holds found", "calibration": calib}
    step.done(f"{len(det['lit'])} lit holds")

    # 3/4. match and report
    step = Step(3, "Matching climb")
    step.update(0.5)
    result = finish(video, calib, det)
    step.done()

    best = result["best"]
    if result["verdict"] not in GOOD or not best:
        print(f"  4/4 Result: {result['verdict']}"
              + (f", best guess {best['name']} at {best['jaccard']:.0%}" if best else ""))
        return "detection", result

    tag = " (mirrored)" if best["mirrored"] else ""
    ties = f", tied with {result['ties']} other(s)" if result["ties"] else ""
    print(f"  4/4 Result: {best['name']}{tag}")
    print(f"      {grade_text(best)}")
    print(f"      {best['jaccard']:.0%} match, {result['verdict']}{ties}")
    return "processed", result


# ---------- session check ----------

def climb_key(c):
    return (c["uuid"], c["mirrored"])


def supports(result, key):
    """This clip's candidate for the given climb, if it fits well enough."""
    for c in result.get("candidates") or []:
        if climb_key(c) == key and c["jaccard"] >= MIN_SUPPORT:
            return c
    return None


def sessions(entries):
    """Split clips (in recording order) wherever there's a long gap."""
    groups, current, last = [], [], None
    for e in entries:
        t = datetime.fromisoformat(e["result"]["recorded_at"])
        if last is not None and t - last > MAX_GAP:
            groups.append(current)
            current = []
        current.append(e)
        last = t
    if current:
        groups.append(current)
    return groups


def session_check(entries):
    """Accept uncertain clips that sit in a run of attempts at the same climb.

    entries: dicts with "result" and "outcome", for every clip this run.
    Returns the entries that were promoted, each with the reason.
    """
    entries = sorted(entries, key=lambda e: e["result"]["recorded_at"])
    promoted = []

    def promote(e, cand, verdict, reason):
        r = e["result"]
        r["promoted_from"] = r["verdict"]
        r["verdict"], r["best"], r["session_reason"] = verdict, cand, reason
        e["outcome"] = "processed"
        promoted.append(e)

    def neutral(e):   # nothing detected (e.g. LEDs off): says nothing either way, skip over it
        return not e["result"].get("candidates") and e["result"].get("verdict") not in ACCEPTED

    def is_open(e):   # uncertain but detected, and not already promoted
        return e["outcome"] == "detection" and e["result"].get("candidates") \
            and e["result"]["verdict"] not in ACCEPTED

    for group in sessions(entries):
        # 1. spread out from each confidently recognised clip
        for i, anchor in enumerate(group):
            if anchor["result"].get("verdict") not in GOOD:
                continue
            key = climb_key(anchor["result"]["best"])
            for step in (-1, 1):
                j = i + step
                while 0 <= j < len(group):
                    e = group[j]
                    r = e["result"]
                    if neutral(e):
                        j += step
                        continue
                    if r.get("verdict") in ACCEPTED and r.get("best") and climb_key(r["best"]) == key:
                        j += step
                        continue
                    cand = supports(r, key) if is_open(e) else None
                    if not cand:
                        break
                    promote(e, cand, "session",
                            f"next to a confident clip of the same climb ({Path(anchor['result']['video']).name})")
                    j += step

        # 2. runs of uncertain clips that agree with each other
        i = 0
        while i < len(group):
            e = group[i]
            if not is_open(e) or e["result"]["best"]["jaccard"] < MIN_SUPPORT:
                i += 1
                continue
            key = climb_key(e["result"]["best"])
            run = [e]
            j = i + 1
            while j < len(group):
                g = group[j]
                if neutral(g):
                    j += 1
                    continue
                if not (is_open(g) and g["result"]["best"]["jaccard"] >= MIN_SUPPORT
                        and climb_key(g["result"]["best"]) == key):
                    break
                run.append(g)
                j += 1
            if len(run) >= RUN_LENGTH:
                for r_e in run:
                    promote(r_e, r_e["result"]["best"], "repeated",
                            f"same best guess in {len(run)} clips in a row")
            i = j
    return promoted


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, default=BACKLOG_DIR, help="folder of clips to process")
    ap.add_argument("--keep", action="store_true", help="don't move any files")
    args = ap.parse_args()

    videos = sorted(p for p in args.folder.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_TYPES) if args.folder.exists() else []
    if not videos:
        raise SystemExit(f"No videos found in {args.folder}")

    total = len(videos)
    print(f"{total} clips in {args.folder}\n")
    started = time.time()
    counts = {"processed": 0, "calibration": 0, "detection": 0}
    destinations = {"processed": PROCESSED_DIR,
                    "calibration": MANUAL_DIR / "calibration",
                    "detection": MANUAL_DIR / "detection"}

    entries = []
    for i, video in enumerate(videos):
        elapsed = time.time() - started
        eta = fmt_time(elapsed / i * (total - i)) if i else "--"
        print(f"[{i + 1}/{total}] {bar(i / total)} {i / total:4.0%}  "
              f"elapsed {fmt_time(elapsed)}, ETA {eta}  {video.name}")

        try:
            outcome, result = process(video, args.keep)
        except Exception as e:   # anything unexpected: park it for a manual look
            print(f"  error: {e}")
            outcome, result = "detection", {"video": str(video), "verdict": "error", "reason": str(e)}
        counts[outcome] += 1
        result["recorded_at"] = recorded_at(video).isoformat(timespec="seconds")

        if not args.keep:
            dest = move(video, destinations[outcome])
            result["video"] = str(dest)
            print(f"      -> moved to {dest}")
        save_result(video.stem, result)
        entries.append({"stem": video.stem, "result": result, "outcome": outcome})
        print()

    promoted = session_check(entries)
    if promoted:
        print(f"Session check: {len(promoted)} uncertain clip(s) accepted as repeat attempts")
        for e in promoted:
            r, b = e["result"], e["result"]["best"]
            tag = " (mirrored)" if b["mirrored"] else ""
            print(f"  {Path(r['video']).name}: {b['name']}{tag}  ({grade_text(b)})")
            print(f"      {b['jaccard']:.0%} match, {r['session_reason']}")
            if not args.keep:
                dest = move(Path(r["video"]), PROCESSED_DIR)
                r["video"] = str(dest)
                print(f"      -> moved to {dest}")
            save_result(e["stem"], r)
            counts["detection"] -= 1
            counts["processed"] += 1
        print()

    n_rows = write_csv()
    took = fmt_time(time.time() - started)
    print(f"[{total}/{total}] {bar(1.0)} 100%  done in {took}\n")
    print("Summary")
    print(f"  recognised:               {counts['processed']}")
    print(f"  need manual calibration:  {counts['calibration']}"
          + (f"  ({MANUAL_DIR / 'calibration'})" if counts['calibration'] and not args.keep else ""))
    print(f"  need a detection check:   {counts['detection']}"
          + (f"  ({MANUAL_DIR / 'detection'})" if counts['detection'] and not args.keep else ""))
    print(f"\nSpreadsheet of all {n_rows} recognised clips: {CSV_PATH}")


if __name__ == "__main__":
    main()

# Board climb analyser

Personal project for a Tension Board 2 **Mirror** layout, **12 high x 8 wide**, fixed at **40°**.
Used by me and a few friends; not public. Developed on Windows (PowerShell, Python venv in `.venv`),
deployed with Docker on an Ubuntu home server.

Two **deliberately decoupled** projects in one repo, sharing only `data/` (gitignored) at the repo
root -- neither project imports the other's code:
1. **`video/`**: recognise which climb is in a board video (LEDs are lit for the climb), plus body
   movement analysis from a body-pose track.
2. **`webapp/`**: search/generate climbs, light them on the board over Bluetooth. The only one
   that's deployed (Docker); `video/` is local-only, run by hand on the PC.

Run every command from the repo root, e.g. `python video/recognise.py data/clip.mov` or
`python webapp/search.py`. Paths resolve via each project's own `config.py` (see below), not the
process's working directory, so this isn't strictly required, but it's the convention used
throughout and in every example below.

## Board settings (config.py)

`PRODUCT_ID = 5` (Tension Board 2), `LAYOUT_ID = 10` (Mirror), `PRODUCT_SIZE_ID = 8` (12 high x 8
wide), `ANGLE = 40`. Because the two projects are decoupled, this tiny file is **duplicated** as
`video/config.py` and `webapp/config.py` -- update both by hand if the board ever changes. Each
copy also defines `DATA_DIR`, resolved from `__file__` (not cwd): `video/config.py` always points
one level up to the shared `data/`; `webapp/config.py` does the same *unless* a `data/` folder
already sits next to it, which is what happens inside the Docker image (see Deployment) -- so the
same code works both locally and flattened into the container.

- Board coordinates are inches, x = 0 on the centre line (x from -44 to 44), y from 0 at the bottom to 144.
  Holes sitting exactly on the size edge (x = ±44) are NOT on this board: always use strict `>` / `<` edge filters.
- The layout is left-right symmetric; mirroring a climb is `x -> -x`.
- `data/tension.db` is the Aurora app database, downloaded with BoardLib (`boardlib database tension data/tension.db`).
  Frames strings look like `p123r5p456r6` (placement id, role id). Roles: 5 start, 6 middle (hand), 7 finish, 8 foot.

## video/ -- the video pipeline

Local-only (Windows PC): needs OpenCV/MediaPipe, never deployed. Own venv-friendly
`video/requirements.txt` (can share the one root `.venv` with `webapp/` for convenience; they don't
conflict, just don't overlap in what each actually needs).

- `load_climbs.py` (**webapp/**, not here -- see below): climbs that fit the board -> `data/climbs_mirror_12x8.json`.
  `video/` only *reads* that file (via `CLIMBS_PATH` in `video/config.py`); it never runs the loader itself.
- `calibrate.py`: click 4 corner holds to map board inches -> video pixels (homography), saved in `data/calibrations/`
- `autocalibrate.py`: calibrate a new camera position automatically by SIFT-matching against a hand calibration
- `detect_leds.py`: find lit holds (peak colourfulness minus patch median, 75th percentile over frames,
  robust threshold, LED hue bands so green wall paint is rejected)
- `match_climb.py`: Jaccard match of detected holds against all climbs and their mirrors; adds grade (V/Font),
  ascents, stars at ANGLE. `full_climb_holds()` returns a matched climb's *complete* official hold list
  (by hole id, from the database) -- used once we're confident which climb it is, since LED detection alone
  can miss a dim/occluded hold.
- `recognise.py`: one command per video (pick calibration, detect, match, save to `data/results/`).
  `resolved_holds()` is the shared entry point for anything that needs "the real hold set for this video":
  it detects + matches, and if the match is confident (`GOOD = {"confident", "likely", "tied"}`), swaps in
  `full_climb_holds()` instead of just what LED detection happened to catch.
- `backlog.py`: processes `data/backlog/` with progress bars: auto-calibrate -> detect -> match; moves clips to
  `data/processed/` or `data/manual/calibration|detection/`; then a session check accepts uncertain clips that are
  repeat attempts next to a confident clip of the same climb. Writes `data/results/backlog.csv`.
- `pose.py`: thin MediaPipe wrapper (Tasks API, `PoseLandmarker`). Downloads a small pose model to
  `data/models/` on first use. `estimate_landmarks()` samples a video for wrist/ankle/hip/shoulder/toe
  (`foot_index`, not ankle -- the actual contact point on a hold) pixel positions; `to_board_inches()`
  projects them to board inches via a calibration's inverse homography.
- `movement.py`: pure logic (no MediaPipe/video dependency, unit-tested with synthetic tracks in
  `test_movement.py`) that turns a landmark track + a climb's hold list (from `resolved_holds()`) into:
  per-limb hold assignments (debounced against tracking jitter), a move sequence, timing/effort metrics
  (duration, pace, rest time, longest pause), and rule-based technique flags (static vs. dynamic from hip
  vertical speed, hesitation, approximate hips-in/out). Single side-on camera, so depth is unknown --
  flags are heuristic, not biomechanical measurements.
- `analyse_movement.py`: one command per video, mirrors `recognise.py`. Resolves the hold set, estimates
  pose, runs `movement.analyse()`, saves `data/results/<video>_movement.json` (kept separate from the
  match result `<video>.json` so existing consumers like `backlog.py`'s CSV are untouched).
- `render_movement.py`: renders `data/results/<video>_movement.mp4` -- the skeleton, every board hole
  (small reference dot), this climb's holds (coloured by role, filled solid while a limb is on them), and
  a timestamp overlaid on the original footage. A visual sanity check for what the tracker actually saw.
- `mediapipe` conflicts with `opencv-python` (both provide `cv2`) -- `video/requirements.txt` uses
  `opencv-contrib-python` instead (a superset) so both can be installed together.

## webapp/ -- the climb search web app

The only project that's deployed. Own `webapp/requirements.txt`; the Docker image installs
`boardlib`/`pillow` directly instead (see Dockerfile), so that file is mainly for running it locally
outside Docker.

- `load_climbs.py`: climbs that fit the board -> `data/climbs_mirror_12x8.json` (`CLIMBS_PATH`). Lives
  here (not in `video/`) because `search.py` runs it directly as a subprocess whenever the database
  changes; `video/` only ever reads its output file.
- `search.py`: small stdlib HTTP server. Serves `webapp/search/` files plus `/climbs.json`. Auto-runs the
  export when the database, climbs file or `export_search.py` changes. `--sync` / `--sync-only` update the
  database with BoardLib using `TENSION_USERNAME` / `TENSION_PASSWORD` from the environment (`webapp/.env`
  on the server, never committed). All paths are built from `__file__`, not cwd or `os.chdir`.
- `export_search.py`: compact JSON for the page: holes `[x, y, led_position, set_index]`, grades, role colours, climbs.
  Must not import OpenCV (the Docker image doesn't have it).
- `search/index.html`: the whole front end in one file (no build step). Features: text/grade/ascents/stars filters,
  hold filters by role (must use / start / hand / finish / foot / avoid), mirrored matching, "find similar",
  layout image background (`board.webp` + `board.json` homography), colour presets incl. colour-blind friendly,
  pinned current climb + swipe on mobile, Web Bluetooth lighting (Aurora protocol, API v2/v3 framing),
  climb generator (learned from real climbs at the target grade, kNN grade estimate, options for hold set,
  hold size, move style, feet, start height, marked holds), saved generated climbs in localStorage.
- `search/sw.js` + `manifest.webmanifest` + icons: referenced by `index.html` for an installable PWA with
  offline cache, but **these files don't actually exist yet** -- the feature is currently broken (silent,
  since the service-worker registration swallows the failure). Still outstanding.
- Web Bluetooth only works over HTTPS or localhost, and not in iPhone browsers except Bluefy (I use an iPhone 16).

## Deployment (home server)

- Repo is private on GitHub. Server clone: `~/board-analyser`; deploy with
  `cd ~/board-analyser/webapp && docker compose up -d --build` (container `climb-search`, host port
  8010 -> 8000). `webapp/Dockerfile` builds with `webapp/` as the build context (its `COPY` lines are
  relative to that), and `webapp/docker-compose.yml` mounts the shared `../data` (one level up) to
  `/app/data` -- inside the container this lands data/ directly alongside `search.py`, which is exactly
  the "flattened" layout `webapp/config.py`'s `DATA_DIR` fallback expects.
- Public at `https://tension.aqhw.org` via Cloudflare Tunnel (service must be `http://192.168.50.144:8010`, not https)
  behind Cloudflare Access (email login).
- `.env` for `TENSION_USERNAME`/`TENSION_PASSWORD` lives at `webapp/.env` on the server (never committed).
- Update: commit + push on the PC, then on the server `git pull && cd webapp && docker compose up -d --build`.
- Weekly cron on the server: `cd ~/board-analyser/webapp && docker compose exec -T climb-search python search.py --sync-only`.

## Conventions

- `data/`, videos, `.venv/`, `__pycache__/` and `.env` are gitignored. Never commit them.
- Keep each script runnable on its own and importable as functions.
- `video/` and `webapp/` must not import each other's code. The only sharing is the `data/` folder on
  disk and the tiny duplicated `config.py` (board settings) -- if a change seems to need more sharing
  than that, it's a sign the change belongs in just one of the two projects, or that the shared piece
  should move to `data/` as a file rather than code.
- Test changes before calling them done. There is no real board or video in CI: use synthetic data
  (e.g. a synthetic staggered hole grid, rendered videos, or a mock climbs.json) and jsdom for the front end.
- I'm on Windows: give PowerShell commands for the PC and bash for the server. Watch for Windows adding
  `.txt` to dotfiles or unusual extensions.
- Plain, clear explanations; I'm a statistics/business analytics student, comfortable with Python and data,
  newer to web and Docker.

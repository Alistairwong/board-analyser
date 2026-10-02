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
- **Role colours** (`detect_leds.assign_role`, `cap_finish`): a lit hold whose hue is nearly as close to a role of
  another kind (hand / foot / finish) as to its best one (within `ROLE_MARGIN` = 10 hue units) is left **undefined**
  (`role_name` None, `role_uncertain`, `role_guess`) instead of guessed. Climb search ignores undefined roles
  (they neither help nor hurt the tie-break), and once the climb is matched the hold takes the climb's own role
  (the recogniser tags each hold `role_source`: colour / climb / conflict). At most **2 finish holds** are kept
  (checked against the database: 92% of climbs have 1, 7.5% have 2, none more); extras become undefined.
  **Height rule** (`match_climb.finish_odds(y, other)`, measured from the database at import): finish holds are
  98.9% in the upper half and 92% in the top 20%, but that alone doesn't separate them from hand holds; counting
  finish per hand hold by height, finish only beats hand in the top tenth (y >= 130, about 4:1; 0.02 or less below
  y = 100) and beats foot holds from y = 100 up. So an unclear finish-vs-hand/foot colour is settled by height when
  the odds are lopsided (>= 3 finish, <= 0.2 other), else stays undefined; a clear colour is trusted at any height;
  when more than 2 holds read as finish, low ones are dropped first.
  **Order in the recogniser: lit or not -> climb by position -> roles.** Detection runs with `assign_roles=False`
  (positions and hues only); `match()` compares positions alone; `match_climb.break_ties()` uses colours only to order
  climbs that tie on position overlap (`colour_agreement`, ignoring unclear colours); only then does
  `detect_leds.assign_lit_roles()` name roles (colour + height rules), used for holds no climb explains and to
  flag colour/climb conflicts. Two finish holds must also be close together: `match_climb.finish_pair_limit()` is the
  95th percentile of real two-finish pairs (about 46 in; median 16 in, so it only rejects far-apart pairs); `cap_finish`
  drops a second finish farther than that. Detection defaults (`assign_roles=True`) are unchanged for the PC scripts and analyser.
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
- **Boards (`webapp/config.py` `BOARDS`)**: the app serves Tension (40 only) and a Kilter Board Original 12x12 with
  kickboard (any 0-70° in 5° steps); the page has Board/Angle dropdowns (reload with `?board=&angle=`).
  `/climbs.json?board=&angle=` exports `data/search/climbs_<board>_<angle>.json` on first request (Kilter keeps only
  climbs with ascents at that angle, ~84k at 40°). `data/kilter.db` comes from `boardlib database kilter` (works without
  a login; `KILTER_USERNAME`/`KILTER_PASSWORD` in `webapp/.env` add newer climbs). Kilter's background is the hold-layout sheet
  `webapp/search/kilter.webp` (homography in `kilter.json`, fitted to the hole positions, ~3 px median error); it is only partly left-right symmetric, so "mirrored" matches ~70% of holes. The video/ tools and
  recogniser/movement/analyser stay Tension-only.
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
- `search/record.html` (served at `/record`): unattended recorder. Camera via `getUserMedia` (an iPhone
  works as a Mac camera via Continuity Camera; iPad/iPhone can also just open the page), MediaPipe pose in
  the browser, board region tapped once (stored in localStorage as 0-1 fractions). A wrist or toe inside
  the region starts a `MediaRecorder` clip *immediately* (no pre-roll: later MediaRecorder chunks can't be
  decoded on their own); it stops after N seconds clear, and clips with under `min-on` seconds on the board
  are discarded (walk-bys). Clips are capped at 40 s (one attempt each; a longer effort continues in the next clip).
  Finished clips are `POST`ed to `/upload` (`search.py` `do_POST`: mp4/webm/mov only, 200 MB cap, server-made
  filename) and saved in `data/recordings/`; failed uploads stay listed with a Retry button. One flat camera,
  so someone walking in front of the board also counts. iOS stops the camera if the screen locks or the tab
  is backgrounded, so keep the tab in front and the device awake (page uses the Wake Lock API).
- `analyser/` (Docker, `docker compose up -d`): watches `data/recordings/`, auto-calibrates, matches the lit holds
  against the climb DB (reusing `video/`'s code) and files each clip under `data/climbs/<climb>/` (or
  `_unidentified/`, `_failed/`). Light work only; pose/movement analysis stays on request via `video/`.
  Needs a hand-made calibration in `data/calibrations/` for auto-calibration to work.
- `recogniser/` (Docker, `cd recogniser && docker compose up -d --build`, host port 8020): web app for
  one-off recognition. Upload a photo or video; it auto-aligns to a hand-made reference calibration
  (`autocalibrate.py`), finds the lit holds, matches the climb (reusing `video/`'s code, copied in by the
  Dockerfile) and returns the photo (for a video, a climber-free "clean plate": the per-pixel median of 15 frames, so it assumes a fixed camera) with the climb's holds circled by role plus name, grade, setter, stars,
  match verdict and a per-hold "confidence" (LED signal strength: 0% at the lit/unlit cut-off, 100% at 3x it;
  not a probability). Its "Calibrate" tab makes the reference calibration in the browser (click the 4 corner
  holes on a photo), so no desktop window is needed. Results go to `data/recogniser/`. `test_recognise.py`
  is a synthetic end-to-end check (see its docstring). Untested on real footage.
  For videos it also says whether the climb was **topped**: after recognising the climb, MediaPipe pose
  (`video/pose.py`, model downloaded to `data/models/` on first use) tracks the climber and
  `movement.analyse()` reports "topped" if a hand ended on a finish hold; otherwise "not topped", or
  "untracked" when no climber was found. Only judged when the climb was confidently recognised, and a miss
  can be the tracker losing the climber. Photos can't be judged. Needs `mediapipe` (so opencv-contrib, not
  headless) and `libegl1`/`libgles2` in the image; the container's memory cap is 3 GB.
  **Empty-board frame** (`recogniser/empty.py`): for a video, one pose pass (every ~3rd frame, at most 600 scans,
  frames shrunk to 640 px, `pose.estimate_landmarks(step_frames, max_side, keep_all)`) marks scans where nobody is
  on the board: no landmark on/around the board outline (pose gate) AND nothing is in the way
  (difference gate: the frame is warped into the calibration photo's view and compared with it, ignoring hole positions, using
  local contrast, texture and brightness after allowing for a change in overall light; if that photo can't be compared with the
  video, the video's own median picture is used instead). Runs of 3+ empty scans count; up to 7 frames are median-merged into the **empty-board image**,
  which is re-aligned and used exactly like an uploaded photo for the lit-hole detection and as the output picture.
  If nobody ever leaves the board (or the empty frame shows no lit holds, e.g. LEDs switched off) it falls back to the
  whole-video detection and clean plate and says so (`result.source`). Live-recorded clips normally end with ~5 s of
  empty board. The same pose pass feeds the body checks (thinned to ~5 samples/s).
  **Cross-checks** (`recogniser/verify.py`, pure functions, tested with synthetic tracks): the result page lists
  independent checks that the climb is right: (1) LED match verdict, (2) split-half: detection re-run on each half
  of the video must pick the same climb (when an empty-board image was used this becomes "empty image and whole video agree") (`detect_leds.detect(span=...)`), (3) body on route: hands (on hand holds)
  and feet (on any hold) sit on the recognised climb's holds at least as much as on the runner-up candidates',
  (4) hands start on a start hold / end on a finish hold. Checks 1-2 come from the LEDs and 3-4 from body pose;
  the overall line is "confirmed" only when both sources agree and nothing disagrees, else "unconfirmed" or
  "double-check". Photos only get check 1. Every check shows the picture it used (`check.images`, files `data/recogniser/<id>-cN.jpg`): lit-hole
  checks (1, 2) re-read their own picture from scratch (empty-board frame; median picture of the video, or of each half
  in the fallback), body checks show the best frame (most hands/feet on the climb) and the start/finish frames with
  the tracked hands/feet marked. Planned: logbook (`boardlib logbook`) and the gym's "recently
  displayed" list as time-based checks (not built; the latter has no known API).
  Also reachable as `/recogniser/` on the main site (`search.py` proxies it to the container via `RECOGNISER_URL`,
  default `http://host.docker.internal:8020`), so it sits behind the same Cloudflare Access login. The page uses
  relative URLs so it works at both addresses. Through the tunnel, uploads over ~100 MB (long videos) will fail.
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

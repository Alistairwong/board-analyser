# Board climb analyser

Personal project for a Tension Board 2 **Mirror** layout, **12 high x 8 wide**, fixed at **40°**.
Used by me and a few friends; not public. Developed on Windows (PowerShell, Python venv in `.venv`),
deployed with Docker on an Ubuntu home server.

Two halves:
1. **Video pipeline**: recognise which climb is in a board video (LEDs are lit for the climb).
2. **Climb search web app**: search/generate climbs, light them on the board over Bluetooth.

## Board settings (config.py)

- `PRODUCT_ID = 5` (Tension Board 2), `LAYOUT_ID = 10` (Mirror), `PRODUCT_SIZE_ID = 8` (12 high x 8 wide), `ANGLE = 40`
- Board coordinates are inches, x = 0 on the centre line (x from -44 to 44), y from 0 at the bottom to 144.
  Holes sitting exactly on the size edge (x = ±44) are NOT on this board: always use strict `>` / `<` edge filters.
- The layout is left-right symmetric; mirroring a climb is `x -> -x`.
- `data/tension.db` is the Aurora app database, downloaded with BoardLib (`boardlib database tension data/tension.db`).
  Frames strings look like `p123r5p456r6` (placement id, role id). Roles: 5 start, 6 middle (hand), 7 finish, 8 foot.

## Video pipeline

- `load_climbs.py`: climbs that fit the board -> `data/climbs_mirror_12x8.json` (`CLIMBS_PATH`)
- `calibrate.py`: click 4 corner holds to map board inches -> video pixels (homography), saved in `data/calibrations/`
- `autocalibrate.py`: calibrate a new camera position automatically by SIFT-matching against a hand calibration
- `detect_leds.py`: find lit holds (peak colourfulness minus patch median, 75th percentile over frames,
  robust threshold, LED hue bands so green wall paint is rejected)
- `match_climb.py`: Jaccard match of detected holds against all climbs and their mirrors; adds grade (V/Font), ascents, stars at ANGLE
- `recognise.py`: one command per video (pick calibration, detect, match, save to `data/results/`)
- `backlog.py`: processes `data/backlog/` with progress bars: auto-calibrate -> detect -> match; moves clips to
  `data/processed/` or `data/manual/calibration|detection/`; then a session check accepts uncertain clips that are
  repeat attempts next to a confident clip of the same climb. Writes `data/results/backlog.csv`.

## Climb search web app

- `search.py`: small stdlib HTTP server. Serves `search/` files plus `/climbs.json`. Auto-runs the export when
  the database, climbs file or `export_search.py` changes. `--sync` / `--sync-only` update the database with
  BoardLib using `TENSION_USERNAME` / `TENSION_PASSWORD` from the environment (`.env` on the server, never committed).
- `export_search.py`: compact JSON for the page: holes `[x, y, led_position, set_index]`, grades, role colours, climbs.
  Must not import OpenCV (the Docker image doesn't have it).
- `search/index.html`: the whole front end in one file (no build step). Features: text/grade/ascents/stars filters,
  hold filters by role (must use / start / hand / finish / foot / avoid), mirrored matching, "find similar",
  layout image background (`board.webp` + `board.json` homography), colour presets incl. colour-blind friendly,
  pinned current climb + swipe on mobile, Web Bluetooth lighting (Aurora protocol, API v2/v3 framing),
  climb generator (learned from real climbs at the target grade, kNN grade estimate, options for hold set,
  hold size, move style, feet, start height, marked holds), saved generated climbs in localStorage.
- `search/sw.js` + `manifest.webmanifest` + icons: installable PWA with offline cache.
- Web Bluetooth only works over HTTPS or localhost, and not in iPhone browsers except Bluefy (I use an iPhone 16).

## Deployment (home server)

- Repo is private on GitHub. Server clone: `~/board-analyser`, run with `docker compose up -d --build`
  (container `climb-search`, host port 8010 -> 8000).
- Public at `https://tension.aqhw.org` via Cloudflare Tunnel (service must be `http://192.168.50.144:8010`, not https)
  behind Cloudflare Access (email login).
- Update: commit + push on the PC, then on the server `git pull && docker compose up -d --build`.
- Weekly cron on the server: `docker compose exec -T climb-search python search.py --sync-only`.

## Conventions

- `data/`, videos, `.venv/`, `__pycache__/` and `.env` are gitignored. Never commit them.
- Keep each script runnable on its own and importable as functions.
- Test changes before calling them done. There is no real board or video in CI: use synthetic data
  (e.g. a synthetic staggered hole grid, rendered videos, or a mock climbs.json) and jsdom for the front end.
- I'm on Windows: give PowerShell commands for the PC and bash for the server. Watch for Windows adding
  `.txt` to dotfiles or unusual extensions.
- Plain, clear explanations; I'm a statistics/business analytics student, comfortable with Python and data,
  newer to web and Docker.

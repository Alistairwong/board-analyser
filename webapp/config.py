"""Board settings shared by all scripts. Change these if the board changes.

This is a duplicate of video/config.py -- the video pipeline and the web
app are deliberately decoupled (neither imports the other's code), but both
need the same board settings, so this tiny file is kept in sync by hand.
Update both copies together when the board changes.
"""
from pathlib import Path

PRODUCT_ID = 5         # Tension Board 2
LAYOUT_ID = 10         # Tension Board 2 Mirror
PRODUCT_SIZE_ID = 8    # 12 high x 8 wide
ANGLE = 40             # board angle, for grades and ascent stats

# Boards the web app can switch between. The scalar settings above are the Tension board and are what the
# video/ project's duplicate of this file has; the app itself reads BOARDS. `angles` are the angles the
# board's app offers; only_climbed drops climbs nobody has logged at the chosen angle (keeps Kilter's ~220k
# climbs down to a size a phone can hold).
BOARDS = {
    "tension": dict(name="Tension Board 2", db="tension.db", product_id=PRODUCT_ID, layout_id=LAYOUT_ID,
                    size_id=PRODUCT_SIZE_ID, angle=ANGLE, angles=[ANGLE], only_climbed=False, image="board.json"),
    "kilter": dict(name="Kilter Board Original 12x12", db="kilter.db", product_id=1, layout_id=1,
                   size_id=10, angle=40, angles=list(range(0, 75, 5)), only_climbed=True, image="kilter.json"),
}

# data/ is shared with the video/ project, normally one level up from this
# folder. Inside the Docker image the layout is flattened (WORKDIR /app has
# data/ mounted directly alongside this file), so fall back to that instead.
_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data" if (_HERE / "data").exists() else _HERE.parent / "data"
CLIMBS_PATH = DATA_DIR / "climbs_mirror_12x8.json"

# Calibration used for videos that don't have their own (your usual tripod spot)
DEFAULT_CALIB = "test1"

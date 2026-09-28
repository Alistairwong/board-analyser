"""Board settings shared by all scripts. Change these if the board changes.

This is a duplicate of webapp/config.py -- the video pipeline and the web
app are deliberately decoupled (neither imports the other's code), but both
need the same board settings, so this tiny file is kept in sync by hand.
Update both copies together when the board changes.
"""
from pathlib import Path

PRODUCT_ID = 5         # Tension Board 2
LAYOUT_ID = 10         # Tension Board 2 Mirror
PRODUCT_SIZE_ID = 8    # 12 high x 8 wide
ANGLE = 40             # board angle, for grades and ascent stats

# data/ is shared with the webapp/ project, one level up from this folder.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLIMBS_PATH = DATA_DIR / "climbs_mirror_12x8.json"

# Calibration used for videos that don't have their own (your usual tripod spot)
DEFAULT_CALIB = "test1"

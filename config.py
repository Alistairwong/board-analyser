"""Board settings shared by all scripts. Change these if the board changes."""
from pathlib import Path

PRODUCT_ID = 5         # Tension Board 2
LAYOUT_ID = 10         # Tension Board 2 Mirror
PRODUCT_SIZE_ID = 8    # 12 high x 8 wide
CLIMBS_PATH = Path("data/climbs_mirror_12x8.json")
ANGLE = 40             # board angle, for grades and ascent stats

# Calibration used for videos that don't have their own (your usual tripod spot)
DEFAULT_CALIB = "test1"

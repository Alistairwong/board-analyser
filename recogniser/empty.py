"""Find a moment in a video when nobody is on the board, and build an image of the empty board from it.

Detecting the lit holds on an empty board can't be spoiled by a climber blocking an LED. A frame counts
as empty when BOTH hold:
  * pose gate: the pose tracker sees no body landmark on (or just around) the board, and
  * difference gate: nothing is in the way, judged by comparing the frame with the calibration photo
    (warped into the photo's view, ignoring the hole positions where an LED may be lit). If that photo
    can't be compared with this video (very different light), the video's own median picture is used
    instead, which has no climber in it because the camera is fixed and the climber moves.
Only runs of several empty checks in a row count, so a one-frame flicker can't pick a bad frame.
"""
import cv2
import numpy as np

MIN_RUN = 3               # consecutive empty checks that make an empty moment
MAX_PICK = 7              # frames averaged (median) into the empty-board image
MIN_VISIBILITY = 0.3      # a landmark this visible counts as a body part being there
BOARD_MARGIN = 0.03       # pose gate: grow the board outline by this share of the frame's longer side
CHANGED_LEVEL = 25        # median-picture gate: a pixel this many grey levels off has changed
MAX_CHANGED = 0.005       # ...and at most this share of the board may have changed
SMALL = 320               # median-picture gate works on frames shrunk to this many pixels
REF_SIDE = 640            # calibration-photo gate works on pictures shrunk to this many pixels
REF_LEVEL = 1.5           # ...a pixel whose local-contrast pattern differs by more than this has changed
TEX_LEVEL = 0.8           # ...or whose amount of texture differs by more than this (log ratio), beyond the overall change
BRIGHT_LEVEL = 1.5        # ...or whose brightness differs by more than this many spreads, beyond the overall change
REF_MAX_CHANGED = 0.01    # ...and at most this share of the board surface may have changed
REF_USABLE = 0.15         # if even the cleanest frame differs from the photo by more than this, the photo can't be used


def empty_runs(flags, min_run=MIN_RUN):
    """Lists of consecutive indices where flags are True, keeping runs of at least min_run."""
    runs, cur = [], []
    for i, f in enumerate(flags):
        if f:
            cur.append(i)
        else:
            if len(cur) >= min_run:
                runs.append(cur)
            cur = []
    if len(cur) >= min_run:
        runs.append(cur)
    return runs


def spread_pick(runs, n=MAX_PICK):
    """Up to n indices from all the runs, spread evenly across them."""
    flat = [i for r in runs for i in r]
    if len(flat) <= n:
        return flat
    return [flat[k] for k in np.linspace(0, len(flat) - 1, n).astype(int)]


def body_on_board(landmarks, outline):
    """True if any well-seen landmark (x, y, visibility) falls inside the (grown) board outline mask."""
    h, w = outline.shape
    return any(v >= MIN_VISIBILITY and 0 <= int(y) < h and 0 <= int(x) < w and outline[int(y), int(x)]
               for x, y, v in landmarks)


def read_frames(clip, indices, max_side=None):
    """{frame index: image} for the wanted frame indices, optionally shrunk. One sequential pass."""
    want, out = set(indices), {}
    cap = cv2.VideoCapture(str(clip))
    i = 0
    while len(out) < len(want) and cap.grab():
        if i in want:
            ok, f = cap.retrieve()
            if ok:
                if max_side and max(f.shape[:2]) > max_side:
                    s = max_side / max(f.shape[:2])
                    f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                out[i] = f
        i += 1
    cap.release()
    return out


def local_stats(gray, sigma=4.0):
    """(local contrast pattern, local spread). The pattern is each pixel minus its neighbourhood mean, over the spread:
    two photos of the same surface under different light give nearly the same pattern. The spread says how much
    texture there is, which a flat object (a person's shirt) in front of a textured board changes."""
    g = gray.astype(np.float32)
    mean = cv2.GaussianBlur(g, (0, 0), sigma)
    sd = np.sqrt(cv2.GaussianBlur((g - mean) ** 2, (0, 0), sigma))
    return (g - mean) / np.sqrt(sd ** 2 + 25.0), sd      # +25: flat, noisy patches don't blow up the pattern


def robust_z(gray, mask):
    """Brightness in units of the board's own spread, so a different exposure doesn't count as a change."""
    g = gray.astype(np.float32)
    vals = g[mask]
    med = np.median(vals)
    return cv2.GaussianBlur((g - med) / (1.4826 * np.median(np.abs(vals - med)) + 1.0), (0, 0), 3)


class ReferenceGate:
    """Compares video frames with the calibration photo to see if anything is in the way.

    reference: {"image": the photo (BGR), "M": 3x3 mapping video pixels to photo pixels, "mask": 0/255 board
    surface in photo pixels with the hole positions blanked out (an LED may be lit there)}.
    A pixel counts as changed if its contrast pattern, amount of texture or brightness differs from the photo's,
    after allowing for a change in light that affects the whole board."""

    def __init__(self, reference, frame_width):
        img = reference["image"]
        self.s = REF_SIDE / max(img.shape[:2])
        self.size = (max(1, int(img.shape[1] * self.s)), max(1, int(img.shape[0] * self.s)))
        gray = cv2.cvtColor(cv2.resize(img, self.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(reference["mask"], self.size, interpolation=cv2.INTER_NEAREST)
        self.mask = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0     # keep off the board's edge
        self.pattern, self.sd = local_stats(gray)
        self.z = robust_z(gray, self.mask) if self.mask.any() else None
        self.M = np.diag([self.s, self.s, 1.0]) @ reference["M"]
        self.frame_width = frame_width

    def changed(self, frame_small):
        """Share of the board surface where this (shrunk) frame differs from the calibration photo."""
        if self.z is None:
            return 1.0
        f = frame_small.shape[1] / self.frame_width
        warped = cv2.warpPerspective(frame_small, self.M @ np.diag([1 / f, 1 / f, 1.0]), self.size)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        pattern, sd = local_stats(gray)
        tex = np.log((sd + 3.0) / (self.sd + 3.0))
        tex = tex - np.median(tex[self.mask])                       # a change in overall sharpness isn't an obstruction
        bright = robust_z(gray, self.mask) - self.z
        moved = (np.abs(pattern - self.pattern) > REF_LEVEL) | (np.abs(tex) > TEX_LEVEL) | (np.abs(bright) > BRIGHT_LEVEL)
        return float(moved[self.mask].mean())


def find_empty(clip, pose_frames, board_mask, plate, video_fps, reference=None):
    """The empty-board image, or None if nobody ever leaves the board.

    pose_frames: pose.estimate_landmarks(..., keep_all=True) output; board_mask: 0/255 board outline at
    full frame size; plate: the video's median picture; reference: see ReferenceGate (optional).
    Returns {"image", "times", "moments", "checked_against"}.
    """
    if len(pose_frames) < MIN_RUN + 2:
        return None
    grown = cv2.dilate(board_mask, np.ones((max(3, int(BOARD_MARGIN * max(board_mask.shape))),) * 2, np.uint8))
    clear = [not body_on_board(f.get("all", []), grown) for f in pose_frames]      # pose gate
    idx = [round(f["t"] * video_fps) for f in pose_frames]

    # difference gate, only for frames the pose gate let through
    frames = read_frames(clip, [i for i, c in zip(idx, clear) if c], max_side=REF_SIDE)
    ok = [c and i in frames for i, c in zip(idx, clear)]
    checked_against = "the video's median picture"
    changed = None
    if reference is not None:
        gate = ReferenceGate(reference, board_mask.shape[1])
        by_ref = [gate.changed(frames[i]) if o else 1.0 for i, o in zip(idx, ok)]
        if min(by_ref, default=1.0) <= REF_USABLE:
            changed, checked_against = [c <= REF_MAX_CHANGED for c in by_ref], "your calibration photo"
    if changed is None:
        s = SMALL / max(board_mask.shape)
        small_mask = cv2.resize(board_mask, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST) > 0
        ref = cv2.cvtColor(cv2.resize(plate, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        changed = []
        for i, o in zip(idx, ok):
            if not o:
                changed.append(False)
                continue
            g = cv2.cvtColor(cv2.resize(frames[i], (ref.shape[1], ref.shape[0])), cv2.COLOR_BGR2GRAY)
            changed.append(bool((cv2.absdiff(g, ref) > CHANGED_LEVEL)[small_mask].mean() <= MAX_CHANGED))
    flags = [o and c for o, c in zip(ok, changed)]
    del frames

    runs = empty_runs(flags)
    if not runs:
        return None
    picked = spread_pick(runs)
    full = read_frames(clip, [idx[k] for k in picked])
    imgs = [full[idx[k]] for k in picked if idx[k] in full]
    if not imgs:
        return None
    image = np.empty_like(imgs[0])
    for y in range(0, image.shape[0], 256):                # in bands, so the stack copy stays small
        image[y:y + 256] = np.median(np.stack([f[y:y + 256] for f in imgs]), axis=0).astype(np.uint8)
    return {"image": image, "times": [round(pose_frames[k]["t"], 1) for k in picked], "moments": len(runs),
            "checked_against": checked_against}

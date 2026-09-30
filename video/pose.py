"""Body pose estimation for a board video, using MediaPipe.

Thin wrapper only: samples a video at a given fps, runs MediaPipe's
PoseLandmarker, and returns the few landmarks movement.py cares about
(wrists, ankles, hips, shoulders) in pixel space, or projected to board
inches via a calibration homography. All the actual move/metric/technique
logic lives in movement.py, which works on plain data and is unit-tested
with synthetic tracks; this file just does MediaPipe I/O and isn't tested
directly.

Downloads a small pose-landmark model (~5MB) to data/models/ on first use.
"""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import requests
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from config import DATA_DIR

MODEL_DIR = DATA_DIR / "models"
MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")

# Landmark indices from MediaPipe's 33-point pose model that movement.py needs.
# Feet use the toe (foot_index), not the ankle: climbers place the front/inside
# edge of the shoe on a hold, and the ankle sits several inches away from that
# contact point, so ankle-based assignment often missed real foot holds.
LANDMARKS = {
    "left_wrist": 15, "right_wrist": 16,
    "left_ankle": 27, "right_ankle": 28,
    "left_foot": 31, "right_foot": 32,
    "left_hip": 23, "right_hip": 24,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_knee": 25, "right_knee": 26,
    # hand points, only used to place the hand (see hand_centre)
    "left_pinky": 17, "right_pinky": 18, "left_index": 19, "right_index": 20,
}


def hand_centre(landmarks, side):
    """The middle of the hand (between the index and pinky knuckles) if both are visible, else None.
    The wrist sits a few inches behind where the fingers actually grip, so this places the hand better.
    ponytail: MediaPipe's pose model only guesses finger points and they get hidden when gripping; a dedicated
    hand-landmark model on a crop around the wrist would be sharper but is slower and needs a second model."""
    a, b = landmarks.get(f"{side}_index"), landmarks.get(f"{side}_pinky")
    if a and b and min(a[2], b[2]) >= 0.5:
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, min(a[2], b[2]))


def ensure_model():
    """Download the pose model on first use; reuse it after that."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model to {MODEL_PATH} ...")
    r = requests.get(MODEL_URL, timeout=60)
    r.raise_for_status()
    MODEL_PATH.write_bytes(r.content)
    return MODEL_PATH


def estimate_landmarks(video, fps=10, progress=None, step_frames=None, max_side=None, keep_all=False):
    """Sample a video and return per-frame pixel landmarks.

    Returns a list of {"t": seconds, "landmarks": {name: (x, y, visibility)}}.
    A landmark is left out of a frame's dict when MediaPipe doesn't detect a
    person at all for that frame; low-visibility landmarks are still included
    (to_board_inches filters those) so gaps are visible to movement.py rather
    than silently interpolated here.

    step_frames: look at every Nth frame instead of `fps` samples a second.
    max_side: shrink frames so their longest side is this many pixels before
    pose estimation (MediaPipe resizes internally anyway; the landmarks still
    come back in the original frame's pixels).
    keep_all: also return every landmark, as frame["all"] = [(x, y, visibility) x 33]
    (empty when no person is found), e.g. to tell whether anyone is in shot.
    """
    video = Path(video)
    ensure_model()
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
    )
    cap = cv2.VideoCapture(str(video))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    step = step_frames or max(1, round(video_fps / fps))

    out = []
    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.grab():
            if frame_idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                t_ms = int(frame_idx / video_fps * 1000)
                h, w = frame.shape[:2]
                small = frame
                if max_side and max(h, w) > max_side:
                    small = cv2.resize(frame, None, fx=max_side / max(h, w), fy=max_side / max(h, w),
                                       interpolation=cv2.INTER_AREA)
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                result = landmarker.detect_for_video(image, t_ms)
                landmarks, every = {}, []
                if result.pose_landmarks:
                    pose = result.pose_landmarks[0]
                    for name, idx in LANDMARKS.items():
                        lm = pose[idx]
                        landmarks[name] = (lm.x * w, lm.y * h, lm.visibility)
                    for side in ("left", "right"):
                        if c := hand_centre(landmarks, side):
                            landmarks[f"{side}_hand"] = c
                    if keep_all:
                        every = [(lm.x * w, lm.y * h, lm.visibility) for lm in pose]
                entry = {"t": t_ms / 1000, "landmarks": landmarks}
                if keep_all:
                    entry["all"] = every
                out.append(entry)
                if progress and total_frames > 0:
                    progress(min(1.0, frame_idx / total_frames))
            frame_idx += 1
    cap.release()
    return out


def to_board_inches(H, frames, min_visibility=0.5):
    """Project pixel landmarks to board inches using the inverse homography.

    Returns a list of {"t": seconds, "points": {name: (x_inches, y_inches)}},
    dropping any landmark below min_visibility for that frame.
    """
    H_inv = np.linalg.inv(H)
    track = []
    for f in frames:
        pts = {}
        for name, (x, y, vis) in f["landmarks"].items():
            if vis < min_visibility:
                continue
            proj = cv2.perspectiveTransform(np.float32([[[x, y]]]), H_inv).reshape(2)
            pts[name] = (float(proj[0]), float(proj[1]))
        track.append({"t": f["t"], "points": pts})
    return track

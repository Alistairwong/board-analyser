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
}


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


def estimate_landmarks(video, fps=10, progress=None):
    """Sample a video and return per-frame pixel landmarks.

    Returns a list of {"t": seconds, "landmarks": {name: (x, y, visibility)}}.
    A landmark is left out of a frame's dict when MediaPipe doesn't detect a
    person at all for that frame; low-visibility landmarks are still included
    (to_board_inches filters those) so gaps are visible to movement.py rather
    than silently interpolated here.
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
    step = max(1, round(video_fps / fps))

    out = []
    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.grab():
            if frame_idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                t_ms = int(frame_idx / video_fps * 1000)
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect_for_video(image, t_ms)
                h, w = frame.shape[:2]
                landmarks = {}
                if result.pose_landmarks:
                    pose = result.pose_landmarks[0]
                    for name, idx in LANDMARKS.items():
                        lm = pose[idx]
                        landmarks[name] = (lm.x * w, lm.y * h, lm.visibility)
                out.append({"t": t_ms / 1000, "landmarks": landmarks})
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

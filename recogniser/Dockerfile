# Climb recogniser web app: photo/video -> climb, with the holds highlighted.
# Build context is the repo root (see docker-compose.yml) so it can reuse video/'s recognition code.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*
# MediaPipe (pose, to judge whether the climb was topped) pulls in opencv-contrib-python, so no -headless build
RUN pip install --no-cache-dir mediapipe requests numpy

WORKDIR /app
# data/ is mounted at /data, which is where video/config.py looks when it sits in /app
COPY video/config.py video/calibrate.py video/autocalibrate.py video/detect_leds.py \
     video/match_climb.py video/recognise.py video/pose.py video/movement.py ./
COPY recogniser/server.py recogniser/verify.py recogniser/empty.py recogniser/index.html ./

EXPOSE 8000
CMD ["python", "server.py"]

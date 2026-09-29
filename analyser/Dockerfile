# Clip analyser: sorts recordings into per-climb folders. Light work only (no pose, no MediaPipe).
# Build context is the repo root (see docker-compose.yml) so it can reuse video/'s recognition code.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir opencv-python-headless numpy

WORKDIR /app
# data/ is mounted at /data, which is where video/config.py looks when it sits in /app
COPY video/config.py video/calibrate.py video/autocalibrate.py video/detect_leds.py \
     video/match_climb.py video/recognise.py ./
COPY analyser/watch.py ./

CMD ["python", "watch.py"]

# Movement analysis web app: board video -> moves, timing, technique flags and an annotated video.
# Build context is the repo root (see docker-compose.yml) so it can reuse video/'s code.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mediapipe requests numpy

WORKDIR /app
COPY video/config.py video/calibrate.py video/autocalibrate.py video/detect_leds.py \
     video/match_climb.py video/recognise.py video/pose.py video/movement.py video/render_movement.py ./
COPY movement/server.py movement/index.html ./

EXPOSE 8000
CMD ["python", "server.py"]

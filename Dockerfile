# Climb search web app. Only needs Python and BoardLib (no OpenCV).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir boardlib

COPY config.py load_climbs.py export_search.py search.py ./
COPY search/ search/

EXPOSE 8000
CMD ["python", "search.py", "--host", "0.0.0.0", "--no-browser"]

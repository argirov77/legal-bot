# syntax=docker/dockerfile:1.4
FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app/src
WORKDIR /app

# system deps for ocrmypdf & pdf processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    poppler-utils \
    qpdf \
    unpaper \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-bul \
    ghostscript \
    libjpeg62-turbo-dev \
    libtiff5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt constraints.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip setuptools wheel

# Prepare a light requirements file so dev builds can skip heavy dependencies.
RUN python - <<'PY'
from pathlib import Path
import re
req_path = Path('/app/requirements.txt')
heavy = re.compile(r"\s*(torch|transformers|bitsandbytes|accelerate)(\b|==)")
light = [line for line in req_path.read_text().splitlines() if not heavy.match(line)]
Path('/app/requirements-light.txt').write_text('\n'.join(light) + ('\n' if light else ''))
PY

FROM base AS dev
ARG INSTALL_HEAVY=false
RUN --mount=type=cache,target=/root/.cache/pip bash -c '\
    if [ "$INSTALL_HEAVY" = "true" ]; then \
        pip install --no-cache-dir -c /app/constraints.txt -r /app/requirements.txt; \
    else \
        pip install --no-cache-dir -c /app/constraints.txt -r /app/requirements-light.txt; \
    fi'
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir -c /app/constraints.txt -r /app/requirements-dev.txt
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
COPY src/ /app/src/
RUN mkdir -p /models /chroma_db /data
RUN chmod +x /app/docker-entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS prod
ARG INSTALL_HEAVY=true
RUN --mount=type=cache,target=/root/.cache/pip bash -c '\
    if [ "$INSTALL_HEAVY" = "true" ]; then \
        pip install --no-cache-dir -c /app/constraints.txt -r /app/requirements.txt; \
    else \
        pip install --no-cache-dir -c /app/constraints.txt -r /app/requirements-light.txt; \
    fi'
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
COPY src/ /app/src/
RUN mkdir -p /models /chroma_db /data
RUN chmod +x /app/docker-entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

ARG INSTALL_HEAVY=false
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    INSTALL_HEAVY=${INSTALL_HEAVY}

WORKDIR /app

# Core system dependencies required for optional OCR/pipeline features.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    poppler-utils \
    qpdf \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-bul \
    ghostscript \
    libjpeg62-turbo-dev \
    libtiff5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.heavy.txt requirements-dev.txt constraints.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip python - <<'PY'
import os
import subprocess


def should_install_heavy() -> bool:
    value = os.environ.get("INSTALL_HEAVY", "false").lower()
    return value in {"1", "true", "yes", "on"}


if should_install_heavy():
    subprocess.check_call([
        "python",
        "-m",
        "pip",
        "install",
        "-r",
        "requirements.heavy.txt",
    ])
PY

COPY docker-entrypoint.sh /app/docker-entrypoint.sh
COPY src/ /app/src/

RUN mkdir -p /models /chroma_db /data \
    && chmod +x /app/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

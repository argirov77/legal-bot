# syntax=docker/dockerfile:1.4
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

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

FROM base AS deps
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r requirements.txt -c constraints.txt

FROM deps AS runtime-base
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
COPY src/ /app/src/

RUN mkdir -p /models /chroma_db /data \
    && chmod +x /app/docker-entrypoint.sh

FROM runtime-base AS heavy-deps
COPY requirements.heavy.txt constraints.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r requirements.heavy.txt -c constraints.txt && \
    python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1+cu121

EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM heavy-deps AS dev
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r requirements-dev.txt -c constraints.txt

FROM heavy-deps AS app

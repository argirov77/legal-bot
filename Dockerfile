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

FROM runtime-base AS runtime
ARG INSTALL_HEAVY=false
ARG USE_CUDA=false
ENV INSTALL_HEAVY=${INSTALL_HEAVY}
ENV USE_CUDA=${USE_CUDA}

RUN --mount=type=cache,target=/root/.cache/pip python - <<'PY'
import os
import subprocess


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "false").strip().lower() in {"1", "true", "yes", "on"}


install_heavy = _env_flag("INSTALL_HEAVY")
use_cuda = install_heavy and _env_flag("USE_CUDA")

if install_heavy:
    subprocess.check_call([
        "python",
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "-r",
        "requirements.heavy.txt",
        "-c",
        "constraints.txt",
    ])

    if use_cuda:
        cuda_channel = os.environ.get("TORCH_CUDA_CHANNEL", "cu121")
        torch_version = os.environ.get("TORCH_VERSION", "2.3.1")
        spec = f"torch=={torch_version}+{cuda_channel}"
        subprocess.check_call([
            "python",
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--force-reinstall",
            spec,
            "--index-url",
            f"https://download.pytorch.org/whl/{cuda_channel}",
            "--extra-index-url",
            "https://pypi.org/simple",
        ])
PY

EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS heavy
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r requirements.heavy.txt -c constraints.txt
ENV INSTALL_HEAVY=true

FROM runtime AS dev
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r requirements-dev.txt -c constraints.txt

FROM runtime AS app

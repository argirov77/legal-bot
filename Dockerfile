FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

ARG INSTALL_HEAVY=false

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

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel

# Prepare a light requirements file so dev builds can skip heavy dependencies.
RUN python - <<'PY'
from pathlib import Path
import re
req_path = Path('/app/requirements.txt')
heavy = re.compile(r"\s*(torch|transformers|bitsandbytes|accelerate)(\b|==)")
light = [line for line in req_path.read_text().splitlines() if not heavy.match(line)]
Path('/app/requirements-light.txt').write_text('\n'.join(light) + ('\n' if light else ''))
PY

# Install dependencies from requirements.txt, optionally skipping heavy packages.
RUN if [ "$INSTALL_HEAVY" = "true" ]; then \
        pip install --no-cache-dir -r /app/requirements.txt; \
    else \
        pip install --no-cache-dir -r /app/requirements-light.txt; \
    fi

# copy sources
COPY src/. .

# create volumes dirs (optional)
RUN mkdir -p /models /chroma_db /data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

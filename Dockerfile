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

# Install light requirements by default, heavy deps are optional
RUN python - <<'PY'
from pathlib import Path
import re
req = Path('/app/requirements.txt').read_text().splitlines()
light = [line for line in req if not re.match(r'\s*(torch|transformers|bitsandbytes|accelerate)(\b|==)', line)]
Path('/app/requirements-light.txt').write_text('\n'.join(light) + ('\n' if light else ''))
PY
RUN pip install --no-cache-dir -r /app/requirements-light.txt
RUN if [ "$INSTALL_HEAVY" = "true" ]; then \
        pip install --no-cache-dir torch[cuda] transformers bitsandbytes accelerate; \
    else \
        echo "Skipping heavy dependencies"; \
    fi

# copy sources
COPY src/. .

# create volumes dirs (optional)
RUN mkdir -p /models /chroma_db /data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

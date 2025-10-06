FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy package sources so "import app" will work (app package at /app/app)
COPY src/. .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
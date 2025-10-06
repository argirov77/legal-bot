# Legal Bot Skeleton

Базовый каркас сервиса на FastAPI с инфраструктурой для дальнейшего расширения. Репозиторий содержит минимальный API, окружение для запуска через Docker Compose и заготовки для тестов и CI.

## Структура проекта

```
.
├── src/             # Исходный код приложения
│   └── app/
│       └── main.py  # Точка входа FastAPI
├── tests/           # Заготовки для юнит-тестов
├── data/            # Папка для служебных данных
├── models/          # Папка для ML-моделей
├── chroma_db/       # Хранилище ChromaDB
├── docker-compose.yml
├── Dockerfile
└── .env.example     # Переменные окружения по умолчанию
```

## Быстрый старт

1. Скопируйте файл переменных окружения:
   ```bash
   cp .env.example .env
   ```
2. Запустите сервис:
   ```bash
   docker compose up --build
   ```
3. Проверьте health-check:
   ```bash
   curl http://localhost:8000/
   # ok
   ```

4. Выполните тестовый запрос к векторному поиску (после наполнения БД чанками):
   ```bash
   curl -X POST "http://localhost:8000/query" \
        -H "Content-Type: application/json" \
        -d '{"text": "пример запроса", "k": 5}'
   ```

## Разработка без Docker

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

## Тесты и линтинг

```bash
ruff check src tests
pytest
```

## Переменные окружения

| Переменная            | Назначение                                |
|-----------------------|-------------------------------------------|
| `LLAMA_MODEL_PATH`    | Путь до модели LLaMA в контейнере         |
| `EMBEDDING_MODEL`     | Имя используемой embedding-модели          |
| `CHROMA_PERSIST_DIR`  | Директория для хранения ChromaDB          |
| `OCR_LANG`            | Язык для OCR-процессинга                  |

## Лицензия

MIT

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

## Пример использования адаптеров провайдеров

Псевдокод ниже демонстрирует планируемый способ обращения к унифицированным
адаптерам. Благодаря единым интерфейсам orchestration-слой сможет выбирать
конкретные реализации провайдеров без изменения бизнес-логики.

```python
from app.providers import EmbeddingProvider, LLMProvider, VectorStoreAdapter

embedding_provider: EmbeddingProvider = get_embedding_provider()
vector_store: VectorStoreAdapter = get_vector_store_adapter()
llm: LLMProvider = get_llm_provider()

question = "Какие условия расторжения в договоре?"
question_embedding = embedding_provider.encode([question])[0]

results = vector_store.query(
    collection="contracts",
    query_embedding=question_embedding,
    k=3,
)

context = "\n".join(hit.document for hit in results)
prompt = f"{context}\n\nВопрос: {question}\nОтвет:"
answer = llm.generate(prompt, max_tokens=200, temperature=0.1)
```

## Переменные окружения

| Переменная            | Назначение                                |
|-----------------------|-------------------------------------------|
| `LLM_BG1_PATH`         | Путь до основной болгарской LLM-модели    |
| `LLM_BG2_PATH`         | Альтернативная болгарская LLM-модель      |
| `EMBEDDING_MODEL_PATH` | Путь к используемой embedding-модели      |
| `CHROMA_PERSIST_DIR`   | Директория для хранения ChromaDB          |
| `OCR_LANG`             | Код языка для OCR-процессинга             |
| `INSTALL_HEAVY`        | Флаг установки тяжёлых зависимостей       |
| `LLM_PROVIDER`         | Провайдер LLM (`transformers` или `mock`) |

## Лицензия

MIT

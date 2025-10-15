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

1. Скопируйте файл переменных окружения и при необходимости отредактируйте его:
   ```bash
   cp .env.example .env
   ```
2. Установите модели (см. раздел «[Где хранить модели](#где-хранить-модели)»).
3. Запустите выбранный профиль Docker Compose (см. ниже).
4. Проверьте health-check:
   ```bash
   curl http://localhost:8000/
   # ok
   ```

После запуска сервиса можно воспользоваться готовыми HTTP-запросами:

* Загрузка документа в сессию:
  ```bash
  curl -X POST "http://localhost:8000/sessions/demo-session/ingest" \
       -F "files=@examples/sample_document.txt"
  ```
* Запрос по сессии:
  ```bash
  curl -X POST "http://localhost:8000/sessions/demo-session/query" \
       -H "Content-Type: application/json" \
       -d '{"question": "О чем документ?", "top_k": 3, "max_tokens": 128}'
  ```

## Профили Docker Compose

### Разработка без GPU

Команда запускает облегчённое окружение с автопересборкой исходников и CPU-only
зависимостями:

```bash
docker compose -f docker-compose.nogpu.yml up --build
```

Альтернативно можно воспользоваться «горячей» сборкой для разработчиков, которая
монтирует каталог `src/` и подготавливает среду для тестов:

```bash
docker compose -f docker-compose.dev.yml up --build
```

### Продакшн / GPU-профиль

Для полноценного окружения с поддержкой GPU используйте базовый compose-файл в
сочетании с GPU-оверлеем. Перед запуском убедитесь, что установлен
`nvidia-container-toolkit` и демон Docker настроен на использование GPU.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

Контейнер смонтирует локальные директории `./models` и `./chroma_db`, что позволяет
повторно использовать скачанные модели и сохранённые эмбеддинги между перезапусками.

### Готовые скрипты для smoke-тестов

После поднятия контейнеров можно использовать вспомогательные сценарии:

```bash
./examples/01_ingest.sh  # загрузить пример документа в выбранную сессию
./examples/02_query.sh   # выполнить запрос по ранее загруженным документам
```

Скрипты принимают переменные окружения `BASE_URL`, `SESSION_ID`, `QUESTION`, `TOP_K` и
`MAX_TOKENS`, что упрощает отладку API.

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

## Где хранить модели

Каталог `models/` примонтирован в контейнер по пути `/models`. Разместите в нём
скачанные веса LLM и эмбеддингов:

```
models/
├── all-MiniLM-L6-v2/           # эмбеддинги
├── bgpt-7b/                    # основная LLM
└── gemma-2-bg/                 # альтернативная LLM
```

Убедитесь, что имена директорий соответствуют путям в переменных окружения
`EMBEDDING_MODEL_PATH` и `LLM_MODEL_PATH`. Чтобы переключиться на альтернативную
болгарскую модель (например, Gemma 2 BG), измените значение `LLM_MODEL_PATH`
на `/models/gemma-2-bg` и перезапустите сервис.

## Наполнение (prefill) Chroma

1. Установите `VECTOR_STORE=chroma` и при необходимости задайте `CHROMA_PERSIST_DIR` в
   файле `.env`.
2. Подготовьте корпус документов в каталоге, доступном контейнеру (например,
   `./data/prefill`).
3. После запуска контейнеров выполните скрипт, который пройдётся по файлам,
   прогонит их через пайплайн и положит чанки в Chroma:

   ```bash
   docker compose exec legal-bot-app python - <<'PY'
   from pathlib import Path
   from app.ingest.pipeline import IngestPipeline
   from app.vectorstore import get_vector_store

   pipeline = IngestPipeline()
   store = get_vector_store()

   docs_dir = Path("/app/data/prefill")
   for path in docs_dir.glob("**/*"):
       if path.is_file():
           chunks = pipeline.ingest(path.read_bytes(), path.name)
           store.upsert_chunks(chunks)
           print(f"Loaded {path}")
   print("Prefill complete")
   PY
   ```

   После выполнения скрипта персистентный каталог `chroma_db/` будет содержать
   эмбеддинги, готовые к использованию API.

## Переменные окружения

| Переменная              | Назначение                                                                 |
|-------------------------|----------------------------------------------------------------------------|
| `CHROMA_PERSIST_DIR`    | Путь до директории с персистентной базой Chroma (по умолчанию `./chroma_db`)|
| `CHROMA_DB_IMPL`        | Реализация драйвера Chroma (`duckdb+parquet`, `duckdb` и т.д.)              |
| `CHROMA_DISTANCE_METRIC`| Метрика расстояния для коллекции (например, `cosine`)                      |
| `VECTOR_STORE`          | Используемый стор (`mock` или `chroma`)                                    |
| `EMBEDDING_MODEL_PATH`  | Путь к модели эмбеддингов в папке `models/`                                 |
| `EMBEDDING_DEVICE`      | Устройство для инференса эмбеддингов (`cpu`, `cuda`, `cuda:0` и т.п.)       |
| `LLM_PROVIDER`          | Провайдер LLM (`transformers`, `mock`)                                     |
| `LLM_MODEL_PATH`        | Путь к каталогу локальной LLM модели                                       |
| `LLM_DEVICE_MAP`        | Стратегия распределения слоёв модели (`single`, `auto`, `sequential`)       |
| `LLM_TORCH_DTYPE`       | Тип тензоров для загрузки модели (`float16`, `bfloat16`, `float32`)         |
| `LLM_MAX_TOKENS`        | Количество новых токенов, генерируемых по умолчанию                        |
| `LLM_TEMPERATURE`       | Базовая температура сэмплирования ответа                                  |
| `LLM_STUB`              | Принудительное использование заглушки вместо настоящей модели              |
| `OCR_LANG`              | Язык для OCR в пайплайне (например, `bul`)                                 |
| `INSTALL_HEAVY`         | Включение тяжёлых зависимостей (torch/transformers/ocr и т.д.) при сборке  |

Дополнительные переменные можно посмотреть в `.env.example`.

## Типичные команды для отладки

```bash
# Показать статус контейнеров
docker compose ps

# Смотреть логи приложения в реальном времени
docker compose logs -f app

# Выполнить команду внутри контейнера приложения
docker compose exec app bash

# Пересоздать окружение «с нуля»
docker compose down -v && docker compose up --build

# Проверить готовность API
curl http://localhost:8000/healthz
```

## Лицензия

MIT

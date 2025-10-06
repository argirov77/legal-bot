# Local RAG Legal Assistant — Архитектура и Спецификация

## 1. Цель и ограничения

### Цель
Построить серверный RAG-сервис, который в рамках `session_id` принимает документы заказчика, индексирует их и отвечает на вопросы от фронтенда, используя только локальные ресурсы. Ответы должны сопровождаться ссылками на исходные фрагменты.

### Ограничения и требования
- **Полностью офлайн**: языковая модель, эмбеддинги, хранение и векторный поиск выполняются локально (llama.cpp, TGI, Ollama и т. п.).
- **Поддержка юридических форматов**: PDF, DOCX, TXT; OCR для сканов.
- **API-эндпоинты**: `/sessions/{session_id}/ingest`, `/sessions/{session_id}/query`, админ-эндпоинты управления.
- **Обязательный аудит**: логирование использованных фрагментов и промптов.
- **Минимизация галлюцинаций**: низкая температура, обязательные цитаты на источники.
- **Быстрый прототип**: целевой стек — FastAPI, Chroma/FAISS, SentenceTransformers/BGE, локальный LLM.

## 2. Компоненты системы и поток данных

```
[Frontend] → [API Gateway / FastAPI] → [Session Manager]
                                  ↘
                                   [Ingest Pipeline] → [Embedding Service] → [Vector Store]
                                                             ↘                    ↙
                                                        [Retrieval / Reranker]
                                                                ↘
                                                             [LLM Adapter]
                                                                ↘
                                                           [Audit & Logging]
                                                                ↘
                                                              [Storage]
```

### API Gateway / FastAPI
- Принимает HTTP-запросы фронтенда.
- Авторизация (API key/JWT), маршрутизация по сессиям.
- Делегирует обработку пайплайнам инжеста и запросов.

### Ingest Pipeline
- Принимает загруженные документы, проводит OCR (если нужно), нормализует текст, чанкует и отправляет в сервис эмбеддингов.

### Embedding Service
- Локальная модель эмбеддингов (например, BGE, all-MiniLM).
- Обрабатывает чанки пакетно и возвращает векторы для записи в хранилище.

### Vector Store
- Локально развернутый Chroma (DuckDB + Parquet) или FAISS с персистентным слоем.
- Хранит текстовые фрагменты и метаданные, поддерживает поиск ближайших соседей.

### Retrieval / Reranking
- Multi-query генерация + ранжирование (Reciprocal Rank Fusion).
- Дополнительно — гибридный поиск (BM25 + векторный) и фильтрация по метаданным.

### Session Manager
- Хранит информацию о сессиях, историю запросов, политику ретенции.
- Контролирует, какие документы доступны конкретному session_id.

### LLM Adapter
- Унифицированный интерфейс генерации ответов.
- Реализации: llama-cpp-python, Text Generation Inference (TGI), Ollama.

### Audit & Logging
- Сохраняет вопросы, промпты, использованные фрагменты, ответы.
- Поддержка хеширования/шифрования чувствительных данных.

### Storage
- Файловое хранилище документов, бэкапы векторного хранилища и моделей.
- Поддержка резервирования и восстановления (snapshot'ы).

### Ops / Monitoring
- Контейнеризация (Docker Compose) или Kubernetes для продакшена.
- Мониторинг (Prometheus/Grafana), алертинг, логирование.

## 3. Ingest Pipeline (детали)

1. **Upload**: фронтенд отправляет файлы через `POST /sessions/{id}/ingest` (multipart/form-data).
2. **Format Detection**: определение типа файла по MIME/расширению.
3. **Text Extraction**:
   - PDF (текст): `pdfminer.six` или `PyPDF2`, сохраняем границы страниц.
   - PDF (скан): `ocrmypdf` (рекомендуется) или `tesseract`.
   - DOCX: `python-docx`.
   - TXT/RTF: прямое чтение.
4. **Normalization**: нормализация Unicode (NFC), удаление лишних пробелов, фиксация индексов символов.
5. **Chunking**:
   - Рекурсивный разделитель: приоритет абзацам → предложениям.
   - По умолчанию: `chunk_chars = 2500`, `overlap_chars = 400` (допустимо уменьшать до 1000/200).
6. **Metadata**: `file_id`, `file_name`, `page`, `chunk_index`, `char_start`, `char_end`, `language`, `hash`, `ingest_time`.
7. **Embeddings**: батчевое получение векторов у локальной модели.
8. **Persist**: запись в Vector DB (`id`, `embedding`, `document_text`, `metadata`).

Особенности для юридического домена:
- Сохраняем `page` и `char_offsets` для точного цитирования.
- OCR-документы должны содержать привязку к страницам и, по возможности, координаты.
- Фиксируем версию и дату инжеста.

## 4. Vector Store — структура и требования

- **Выбор**: Chroma (DuckDB + Parquet) для простоты; FAISS/Milvus/Weaviate — для масштабирования.
- **Схема**:
  - `id: string` — уникальный идентификатор чанка.
  - `embedding: float[]` — эмбеддинг чанка.
  - `document_text: string` — текст фрагмента.
  - `metadata: json` — метаданные (`file_id`, `file_name`, `page`, `chunk_index`, `char_start`, `char_end`, `language`, `ingest_time`, `source_hash`).
- **API**:
  - `query(embedding, top_k)` → `[ {id, distance, document_text, metadata} ]`.
  - `query_by_text(text, top_k)` → вычисляет эмбеддинг и вызывает `query`.
- **Персистентность**: снапшоты, резервные копии, контроль версий.

## 5. Retrieval & Reranking

- **Multi-query**: генерируем 3–5 перефразированных запросов для повышения полноты (recall) на юридическом языке.
- **Поиск**: для каждого запроса получаем top-M (например, 10) кандидатов из векторного хранилища.
- **Агрегация**: используем Reciprocal Rank Fusion — `score(doc) = Σ 1/(k + rank(doc))`, где `k ≈ 60`.
- **Финальный отбор**: top-K фрагментов для LLM.
- **Дополнительно**:
  - Гибридный поиск (BM25 + векторный) при необходимости.
  - Фильтрация по метаданным (дата, юрисдикция и т. д.).

## 6. Prompt Engineering и шаблоны

### System Prompt (английский)
```
You are a legal assistant. You must answer based ONLY on the provided CONTEXT fragments. If the answer cannot be found in the context, say "I don't have the information in the provided documents." Do not invent laws, dates, or clauses. For each factual statement, include citations in the format [file_name:page:chunk_index]. First provide a short direct answer (1-3 sentences), then an optional explanation with citations. Always include list of sources used.
```

### User Prompt (пример)
```
QUESTION: {user_question}


CONTEXT:
{top_k_fragments}


Answer concisely. Cite fragments inline using [file:page:chunk].
```

### Сбор контекста
- Включать только `file_name`, `page`, текст фрагмента.
- Ограничивать суммарный размер промпта (отбор релевантных и разнообразных фрагментов).
- Температура LLM: 0.0–0.2.

### Русскоязычный System Prompt
```
Вы — юридический ассистент. Отвечайте только на основании предоставленного КОНТЕКСТА. Если ответ отсутствует, честно признайте это. Для каждой фактической позиции давайте ссылку в формате [file_name:page:chunk_index]. Первой строкой давайте краткий ответ (1-2 предложения), затем — развернутый аргумент с цитатами. Всегда перечисляйте источники.
```

## 7. LLM Adapter — интерфейс и реализации

```python
from typing import Protocol, Dict, Any

class LLMClient(Protocol):
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        """Возвращает dict с полями: text (str), raw (dict)."""
```

### Реализация A: llama-cpp-python
- Используем `llama_cpp.Llama` с локальным GGUF.
- Пример:
```python
from llama_cpp import Llama

llm = Llama(model_path="/models/legal-llama.gguf", n_ctx=4096)
res = llm.create(prompt=prompt, max_tokens=512, temperature=0.0)
text = res["choices"][0]["text"]
```

### Реализация B: Text Generation Inference (TGI)
- HTTP POST `/v1/generate` → `{ "model": "legal-llm", "input": prompt, "parameters": {"max_new_tokens": 512, "temperature": 0.0} }`.
- Ответ JSON парсится в `text`, `raw`.

### Реализация C: Ollama
- HTTP API или CLI (`ollama run`).
- Аналогичный вызов с контролем таймаутов и ретраями.

### Общие рекомендации
- Таймауты, ретраи, лимиты параллелизма.
- Пул воркеров/очередь (например, Redis + RQ) при высоких нагрузках.

## 8. API (HTTP контракты)

Все ответы — JSON. Эндпоинты защищены (API key / JWT).

### `POST /sessions/{session_id}/ingest`
- **Описание**: загрузка одного или нескольких файлов.
- **Запрос**: `multipart/form-data` — поля `files[]`, опционально `source_name`, `metadata` (JSON).
- **Ответ**:
```json
{
  "status": "ok",
  "added_chunks": 128,
  "file_ids": ["f123", "f124"]
}
```

### `POST /sessions/{session_id}/query`
- **Описание**: задаёт вопрос, получает ответ от LLM.
- **Запрос**:
```json
{
  "question": "Каков срок расторжения?",
  "k": 5,
  "rerank": true,
  "session_history": false
}
```
- **Ответ**:
```json
{
  "answer": "Срок расторжения — 30 календарных дней с момента уведомления.",
  "sources": [
    {
      "file": "contract.pdf",
      "page": 5,
      "chunk_index": 3,
      "text": "Пункт 5. Срок расторжения — 30 дней...",
      "distance": 0.12
    }
  ],
  "raw": {
    "llm": { "tokens": 234, "time_ms": 1530 }
  }
}
```

### `GET /sessions/{session_id}/metadata`
- Возвращает список загруженных файлов, количество чанков, дату инжеста.

### `POST /admin/reindex`
- Переиндексация сессии/всего хранилища (требует админ-прав).

## 9. Формат хранения чанков и метаданные

```json
{
  "id": "c1a2b3",
  "file_id": "f123",
  "file_name": "contract.pdf",
  "page": 12,
  "chunk_index": 3,
  "char_start": 3450,
  "char_end": 4989,
  "language": "ru",
  "ingest_time": "2025-10-06T08:00:00Z",
  "source_hash": "sha256:..."
}
```

- **Версионирование**: при обновлении документа генерировать новый `file_id` и `source_hash`.
- **Сжатие**: по возможности хранить уплотнённый текст (сжатие на уровне Vector DB).

## 10. Prompt Logging & Audit

- Логировать: время запроса, session_id, вопрос, промпт, ответ LLM, список использованных чанков.
- Чувствительные данные — хэшировать или шифровать (например, с помощью KMS).
- Хранение логов в неизменяемом журнале (append-only, WORM).

## 11. Ops & Deployment

### Docker Compose (PoC)
- Сервисы: `app` (FastAPI), `embedding` (опционально отдельный сервис), `vector_db` (Chroma), `llm` (llama.cpp или TGI), `monitoring` (Prometheus/Grafana).
- Volume'ы: `/models`, `/data/vector_db`, `/data/documents`.

### Kubernetes (Production)
- `Deployment` для FastAPI, `StatefulSet` для Vector Store.
- `HPA` для TGI, `PersistentVolumeClaim` для данных.
- Секреты (API key/JWT) в Kubernetes Secrets.

### Мониторинг и алертинг
- Метрики FastAPI (latency, throughput), LLM (latency, очередь), Vector DB (время поиска).
- Prometheus + Grafana dashboards.
- Alertmanager для SLA (latency > threshold, ошибки > X%).

### Backup & Disaster Recovery
- Регулярные снапшоты `/data/vector_db` и исходных документов.
- Резервное копирование моделей (если не хранятся централизованно).
- Тестирование восстановления.

### Масштабирование
- Горизонтальное масштабирование LLM (TGI cluster, несколько GPU).
- Шардирование Vector DB по `session_id` или `customer_id`.
- Кэширование популярных ответов (Redis) и очереди задач.

## 12. Security & Privacy

- **Аутентификация**: API key / JWT, привязка к session_id и ACL.
- **Сеть**: LLM и Vector DB доступны только внутри приватного сегмента.
- **Шифрование**: TLS для внешних соединений, шифрование данных на диске.
- **Контроль доступа**: политики удаления, ручка удаления сессии/документов.
- **Аудит**: неизменяемые логи, контроль доступа к логам.

## 13. Testing & Evaluation

### Юнит-тесты
- Извлечение и чанкинг (валидность границ, метаданных).
- Корректность эмбеддингов и запись в Vector DB.

### Интеграционные тесты
- Полный цикл: ingest → query → проверка, что ответ ссылается на правильный фрагмент.
- Тестирование OCR-потока (сканы → текст → поиск).

### Метрики качества
- **Retrieval**: Precision@K, Recall@K, nDCG на размеченном корпусе.
- **Ответы**: точность, полнота, уровень галлюцинаций (ручная оценка).
- **Производительность**: средняя латентность (холодный/тёплый старт), throughput.

### Бенчмарки
- Корпус: 50–500 юридических документов.
- Измеряем: время инжеста (на файл/100 страниц), среднюю латентность запроса, время генерации LLM, использование ресурсов (CPU/GPU/RAM).

## 14. Дорожная карта внедрения

| Неделя | Этап | Результат |
|--------|------|-----------|
| 1 | Каркас репозитория, FastAPI, простой инжест (PDF/DOCX), Chroma, базовые эмбеддинги, llama-cpp | PoC, минимальные ответы |
| 2 | Multi-query RRF, Session Manager, промпты с цитированием | Улучшенная релевантность, снижение галлюцинаций |
| 3 | OCR-пайплайн, аудит логов, тесты, Docker Compose | Полноценный локальный прототип |
| 4 | Адаптер TGI, бенчмарки, нагрузочное тестирование, деплой на GPU | Подготовка к продакшену |
| 5+ | Безопасность, K8s, мониторинг, SLA, расширение моделей | Production-ready система |

## 15. Production Checklist

- [ ] Настроены API key/JWT и ACL.
- [ ] Шифрование данных и транспортного уровня.
- [ ] Регулярные бэкапы Vector DB и документов.
- [ ] Мониторинг и алерты активны.
- [ ] Логи аудита доступны и защищены.
- [ ] Проведено тестирование восстановления после сбоя.
- [ ] Нагрузка проверена (пиковые запросы, стресс-тесты).
- [ ] Проведён юридический и security review.

## 16. Примеры промптов и ответов

### Контекст
```
Source 1 — contract.pdf (page 5, chunk 1):
"Пункт 5. Срок расторжения — 30 дней со дня получения уведомления..."
```

### Вопрос
```
QUESTION: Каков срок расторжения по контракту?
```

### Ожидаемый ответ
```
Кратко: Срок расторжения — 30 календарных дней с момента получения письменного уведомления. [contract.pdf:5:1]

Развернуто: Согласно пункту 5 (стр.5) договора, стороны могут расторгнуть договор при уведомлении за 30 дней... [contract.pdf:5:1]

Источники: contract.pdf:5:1
```

## Заключение

Данный документ описывает архитектуру локального RAG-ассистента для юридических документов, охватывая цели, компоненты, API, хранение, промпт-инжиниринг, адаптеры LLM, операционные и тестовые аспекты, а также дорожную карту внедрения. Следующий шаг — выбрать конкретную реализацию LLM (например, llama-cpp для прототипа или TGI для продакшена) и подготовить PoC, например Docker Compose с FastAPI + Chroma + локальной LLM.

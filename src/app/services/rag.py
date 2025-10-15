from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from fastapi import UploadFile

from app.ingest.models import DocumentChunk
from app.ingest.pipeline import IngestPipeline, IngestPipelineConfig
from app.llm_provider import DEFAULT_STUB_RESPONSE, LLMGenerationError, get_llm
from app.storage import save_upload
from app.vectorstore import (
    ChunkSearchResult,
    ChunkVectorStore,
    VectorStoreUnavailableError,
    get_vector_store,
)

LOGGER = logging.getLogger(__name__)
AUDIT_LOGGER = logging.getLogger("app.ingest.audit")

BG_NO_DATA_MESSAGE = "Нямам достатъчно информация в заредените документи."
BG_MODEL_UNAVAILABLE_MESSAGE = DEFAULT_STUB_RESPONSE


def _heavy_dependencies_enabled() -> bool:
    flag = os.getenv("INSTALL_HEAVY", "false").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        LOGGER.warning("Invalid integer for %s: %s; using default %s", name, value, default)
        return default


def _float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        LOGGER.warning("Invalid float for %s: %s; using default %s", name, value, default)
        return default


@dataclass(slots=True)
class StoredChunk:
    """Container describing a retrieved chunk along with its score."""

    id: str
    content: str
    metadata: dict[str, object]
    score: float


@dataclass(slots=True)
class IngestResult:
    """Structured result returned from :meth:`RAGService.ingest`."""

    session_id: str
    saved_files: List[str]
    chunk_count: int
    duration_seconds: float


@dataclass(slots=True)
class AnswerResult:
    """Structured result returned from :meth:`RAGService.answer`."""

    session_id: str
    question: str
    top_k: int
    max_tokens: int | None
    answer: str
    sources: List[StoredChunk]


class RAGService:
    """High level orchestration for the legal assistant RAG workflow."""

    def __init__(
        self,
        *,
        pipeline: IngestPipeline | None = None,
        vector_store: ChunkVectorStore | None = None,
        heavy_enabled: bool | None = None,
    ) -> None:
        self.pipeline = pipeline or IngestPipeline(IngestPipelineConfig())
        self._vector_store = vector_store or get_vector_store()
        self._heavy_enabled = heavy_enabled if heavy_enabled is not None else _heavy_dependencies_enabled()
        self._default_max_tokens = _int_from_env("LLM_MAX_TOKENS", 256)
        self._default_temperature = _float_from_env("LLM_TEMPERATURE", 0.0)
        self._recent_chunks: dict[str, List[DocumentChunk]] = {}

    async def ingest(self, session_id: str, files: Iterable[UploadFile]) -> IngestResult:
        start_time = time.perf_counter()
        saved_files: List[str] = []
        chunk_count = 0

        for upload in files:
            if upload is None:
                continue

            save_started = time.perf_counter()
            destination = await save_upload(session_id, upload)
            display_name = Path(upload.filename or destination.name).name
            saved_files.append(f"{session_id}/{display_name}")
            save_duration = time.perf_counter() - save_started
            LOGGER.info(
                "Saved upload %s for session %s in %.3fs", upload.filename, session_id, save_duration
            )

            file_bytes = destination.read_bytes()
            chunks = self._run_pipeline(file_bytes, upload.filename or destination.name)
            chunk_count += len(chunks)
            LOGGER.info(
                "Generated %s chunks for %s in session %s", len(chunks), upload.filename, session_id
            )

            if not chunks:
                continue

            self._persist_chunks(session_id, chunks)
            AUDIT_LOGGER.info(
                {
                    "event": "ingest",
                    "session_id": session_id,
                    "file_name": upload.filename,
                    "chunk_count": len(chunks),
                }
            )

        duration = time.perf_counter() - start_time
        return IngestResult(
            session_id=session_id,
            saved_files=saved_files,
            chunk_count=chunk_count,
            duration_seconds=duration,
        )

    def answer(
        self,
        session_id: str,
        question: str,
        *,
        top_k: int = 3,
        max_tokens: int | None = None,
    ) -> AnswerResult:
        results = self._query_chunks(session_id, question, top_k)
        stored_chunks = [self._convert_result(result) for result in results]
        LOGGER.debug(
            "Query for session %s returned %d results (stored=%d)",
            session_id,
            len(results),
            len(stored_chunks),
        )

        effective_max_tokens = self._resolve_max_tokens(max_tokens)

        if not stored_chunks:
            stored_chunks = self._fallback_chunks(session_id, top_k)

        if not stored_chunks:
            answer_text = BG_NO_DATA_MESSAGE
        elif not self._heavy_enabled:
            answer_text = BG_MODEL_UNAVAILABLE_MESSAGE
        else:
            prompt = self._build_generation_prompt(question, stored_chunks)
            llm = get_llm()
            temperature = self._default_temperature
            try:
                answer_text = llm.generate(
                    prompt,
                    max_tokens=effective_max_tokens,
                    temperature=temperature,
                )
            except LLMGenerationError as error:
                LOGGER.exception("LLM generation failed for session %s", session_id)
                raise RuntimeError("Неуспешно генериране на отговор от LLM") from error

        AUDIT_LOGGER.info(
            {
                "event": "query",
                "session_id": session_id,
                "question": question,
                "sources": [chunk.id for chunk in stored_chunks],
            }
        )

        return AnswerResult(
            session_id=session_id,
            question=question,
            top_k=top_k,
            max_tokens=effective_max_tokens,
            answer=answer_text,
            sources=stored_chunks,
        )

    def _run_pipeline(self, file_bytes: bytes, file_name: str) -> List[DocumentChunk]:
        try:
            return self.pipeline.ingest(file_bytes, file_name)
        except Exception as error:  # pragma: no cover - depends on optional deps
            LOGGER.exception("Ingest pipeline failed for %s", file_name)
            raise RuntimeError(f"Failed to process {file_name}") from error

    def _persist_chunks(self, session_id: str, chunks: List[DocumentChunk]) -> None:
        for chunk in chunks:
            chunk.metadata.session_id = session_id
        try:
            self._vector_store.upsert_chunks(chunks)
        except VectorStoreUnavailableError:
            raise
        except Exception as error:  # pragma: no cover - defensive guard
            LOGGER.exception("Unexpected error while persisting chunks")
            raise VectorStoreUnavailableError("Failed to persist chunks", cause=error) from error
        self._recent_chunks[session_id] = list(chunks)

    def _query_chunks(self, session_id: str, question: str, top_k: int) -> List[ChunkSearchResult]:
        search_k = max(top_k * 5, top_k + 5)
        try:
            results = self._vector_store.query_by_text(question, k=search_k)
        except VectorStoreUnavailableError:
            raise
        except Exception as error:  # pragma: no cover - defensive guard
            LOGGER.exception("Unexpected error while querying the vector store")
            raise VectorStoreUnavailableError("Vector store query failed", cause=error) from error

        filtered = [
            result
            for result in results
            if str(result.metadata.get("session_id")) == session_id
        ]
        return filtered[:top_k]

    def _convert_result(self, result: ChunkSearchResult) -> StoredChunk:
        metadata = dict(result.metadata)
        distance = float(result.distance)
        score = max(0.0, 1.0 - distance)
        metadata.setdefault("distance", distance)
        metadata["score"] = score
        return StoredChunk(id=result.id, content=result.content, metadata=metadata, score=score)

    def _build_generation_prompt(self, question: str, chunks: List[StoredChunk]) -> str:
        cleaned_question = question.strip() or "(няма въпрос)"
        bullet_lines: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            snippet = " ".join(chunk.content.split())
            if snippet:
                bullet_lines.append(f"{index}. {snippet}")
        if not bullet_lines:
            bullet_lines.append("Няма релевантен контекст.")
        context_block = "\n".join(bullet_lines)
        instructions = (
            "Използвай предоставения контекст, за да отговориш на въпроса. "
            "Ако контекстът не съдържа достатъчно информация, отговори точно "
            "\"Нямам достатъчно информация в заредените документи.\""
        )
        return (
            f"{instructions}\n\n"
            f"Въпрос: {cleaned_question}\n\n"
            f"Контекст:\n{context_block}\n\n"
            "Отговор:"
        )

    def _resolve_max_tokens(self, max_tokens: int | None) -> int:
        if max_tokens is None or max_tokens <= 0:
            return self._default_max_tokens
        return max_tokens

    def _fallback_chunks(self, session_id: str, top_k: int) -> List[StoredChunk]:
        recent = self._recent_chunks.get(session_id, [])
        if not recent:
            return []

        fallback_chunks: List[StoredChunk] = []
        for chunk in recent[: max(top_k, 1)]:
            metadata = {
                "session_id": session_id,
                "file_name": chunk.metadata.file_name,
                "page": chunk.metadata.page,
                "chunk_index": chunk.metadata.chunk_index,
                "char_start": chunk.metadata.char_start,
                "char_end": chunk.metadata.char_end,
                "language": chunk.metadata.language,
                "distance": 1.0,
                "score": 0.0,
            }
            chunk_id = self._make_chunk_id(chunk)
            fallback_chunks.append(
                StoredChunk(id=chunk_id, content=chunk.content, metadata=metadata, score=0.0)
            )
        return fallback_chunks

    @staticmethod
    def _make_chunk_id(chunk: DocumentChunk) -> str:
        metadata = chunk.metadata
        seed = (
            f"{metadata.file_id}:{metadata.chunk_index}:{metadata.char_start}:{metadata.char_end}:{metadata.file_name}"
        )
        return uuid.uuid5(uuid.NAMESPACE_URL, seed).hex


_rag_service = RAGService()


def get_rag_service() -> RAGService:
    """FastAPI dependency returning the shared :class:`RAGService` instance."""

    return _rag_service

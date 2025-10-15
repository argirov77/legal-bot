from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from fastapi import UploadFile

from app.ingest.models import DocumentChunk
from app.ingest.pipeline import IngestPipeline, IngestPipelineConfig
from app.storage import save_upload
from app.vectorstore import (
    ChunkSearchResult,
    ChunkVectorStore,
    VectorStoreUnavailableError,
    get_vector_store,
)

LOGGER = logging.getLogger(__name__)
AUDIT_LOGGER = logging.getLogger("app.ingest.audit")

BG_NO_DATA_MESSAGE = "Съжалявам — няма достатъчно информация в заредените документи."
BG_MODEL_UNAVAILABLE_MESSAGE = "Моделата локално не е налична."


def _heavy_dependencies_enabled() -> bool:
    flag = os.getenv("INSTALL_HEAVY", "false").strip().lower()
    return flag not in {"0", "false", "no", "off"}


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

        if not stored_chunks:
            answer_text = BG_NO_DATA_MESSAGE
        elif not self._heavy_enabled:
            answer_text = BG_MODEL_UNAVAILABLE_MESSAGE
        else:
            answer_text = self._compose_answer(question, stored_chunks, max_tokens)

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
            max_tokens=max_tokens,
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

    def _compose_answer(
        self,
        question: str,
        chunks: List[StoredChunk],
        max_tokens: int | None,
    ) -> str:
        cleaned_question = question.strip() or "(няма въпрос)"
        lines = [
            f"Въз основа на заредените документи за въпроса „{cleaned_question}“ намерих:"  # noqa: B950
        ]
        for chunk in chunks:
            snippet = " ".join(chunk.content.split())
            if snippet:
                lines.append(f"- {snippet}")
        if len(lines) == 1:
            lines.append("Контекстът не съдържа достатъчно информация.")
        answer = "\n".join(lines)
        return self._truncate_to_tokens(answer, max_tokens)

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int | None) -> str:
        if not max_tokens or max_tokens <= 0:
            return text
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text
        return " ".join(tokens[:max_tokens])


_rag_service = RAGService()


def get_rag_service() -> RAGService:
    """FastAPI dependency returning the shared :class:`RAGService` instance."""

    return _rag_service

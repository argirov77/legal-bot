"""Simple in-memory RAG service used by the API layer."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, MutableMapping
from uuid import uuid4

from fastapi import UploadFile

from app.chunker import chunk_text


@dataclass(slots=True)
class StoredChunk:
    """Container for a chunk stored in the in-memory index."""

    id: str
    content: str
    metadata: dict[str, object]


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
    """A lightweight Retrieval-Augmented Generation service.

    The implementation keeps all data in memory which makes it predictable and
    fast for unit tests.  It is intentionally simple – chunking relies on the
    existing :func:`app.chunker.chunk_text` helper and similarity search is
    approximated by returning the most recently ingested chunks.
    """

    def __init__(self) -> None:
        self._sessions: MutableMapping[str, List[StoredChunk]] = {}

    async def ingest(self, session_id: str, files: Iterable[UploadFile]) -> IngestResult:
        """Persist uploaded files in memory and create chunks."""

        start_time = time.perf_counter()
        saved_files: List[str] = []
        chunk_count = 0
        session_store = self._sessions.setdefault(session_id, [])

        for upload in files:
            file_name = upload.filename or "upload"
            virtual_path = f"{session_id}/{file_name}"
            saved_files.append(virtual_path)

            contents = await upload.read()
            upload.file.seek(0)
            text = contents.decode("utf-8", errors="ignore")

            chunks = chunk_text(text)
            for index, chunk in enumerate(chunks):
                chunk_id = uuid4().hex
                metadata = {
                    "filename": file_name,
                    "session_id": session_id,
                    "chunk_index": index,
                    "char_start": chunk.get("meta", {}).get("start", 0),
                    "char_end": chunk.get("meta", {}).get("end", len(chunk["text"])),
                }
                session_store.append(
                    StoredChunk(id=chunk_id, content=chunk["text"], metadata=metadata)
                )
            chunk_count += len(chunks)

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
        """Return a deterministic answer composed from stored chunks."""

        session_chunks = self._sessions.get(session_id, [])
        selected = session_chunks[:top_k] if top_k > 0 else []

        combined = " ".join(chunk.content for chunk in selected).strip()
        if not combined:
            combined = "I do not have enough information to answer yet."

        answer = self._truncate_to_tokens(combined, max_tokens)
        return AnswerResult(
            session_id=session_id,
            question=question,
            top_k=top_k,
            max_tokens=max_tokens,
            answer=answer,
            sources=list(selected),
        )

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

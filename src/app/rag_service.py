"""RAG service orchestration using lightweight mock components."""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from starlette.datastructures import UploadFile


async def _maybe_await(result: Any) -> Any:
    """Await result if it is awaitable."""

    if inspect.isawaitable(result):
        return await result
    return result


class InMemoryStorage:
    """Persist uploaded files in memory for the duration of the process."""

    def __init__(self) -> None:
        self._files: Dict[str, List[Dict[str, Any]]] = {}

    async def save(self, session_id: str, upload_file: UploadFile) -> bytes:
        content = await upload_file.read()
        entry = {"filename": upload_file.filename, "content": content}
        self._files.setdefault(session_id, []).append(entry)
        return content


class PlainTextExtractor:
    """Extract plain text from uploaded content."""

    def extract(self, content: bytes, filename: str | None = None) -> str:
        return content.decode("utf-8", errors="ignore")


class SimpleTextChunker:
    """Break long text into simple newline-delimited chunks."""

    def __init__(self, max_chars: int = 400, overlap: int = 0) -> None:
        self.max_chars = max_chars
        self.overlap = overlap

    def _chunk_lines(self, text: str) -> List[str]:
        current: List[str] = []
        current_len = 0
        chunks: List[str] = []
        for line in text.splitlines():
            if current_len + len(line) + (1 if current else 0) > self.max_chars and current:
                chunks.append("\n".join(current).strip())
                if self.overlap and chunks[-1]:
                    overlap_text = chunks[-1][-self.overlap :]
                    current = [overlap_text]
                    current_len = len(overlap_text)
                else:
                    current = []
                    current_len = 0
            if line.strip():
                current.append(line)
                current_len += len(line) + (1 if current_len else 0)
        if current:
            chunks.append("\n".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def split(self, text: str) -> List[str]:
        if not text.strip():
            return []
        return self._chunk_lines(text)


class MockEmbeddingProvider:
    """Return deterministic numeric embeddings for provided texts."""

    async def encode(self, texts: Sequence[str]) -> List[List[float]]:
        return [[float(len(text))] for text in texts]


@dataclass
class StoredChunk:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]


class InMemoryVectorStore:
    """Store document chunks in memory grouped by session."""

    def __init__(self) -> None:
        self._store: Dict[str, List[StoredChunk]] = {}

    async def add(
        self,
        session_id: str,
        filename: str,
        chunks: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> List[str]:
        session_chunks = self._store.setdefault(session_id, [])
        chunk_ids: List[str] = []
        for index, chunk in enumerate(chunks):
            chunk_id = f"{filename or 'file'}-{len(session_chunks) + index}"
            metadata = {
                "session_id": session_id,
                "filename": filename,
                "chunk_index": len(session_chunks) + index,
            }
            session_chunks.append(StoredChunk(chunk_id=chunk_id, content=chunk, metadata=metadata))
            chunk_ids.append(chunk_id)
        return chunk_ids

    async def retrieve(self, session_id: str, query: str, top_k: int) -> List[StoredChunk]:
        chunks = list(self._store.get(session_id, []))
        if not chunks or top_k <= 0:
            return []
        query_terms = {token.lower() for token in query.split() if token}
        scored: List[tuple[int, StoredChunk]] = []
        for chunk in chunks:
            chunk_terms = {token.lower() for token in chunk.content.split() if token}
            score = len(query_terms & chunk_terms)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]


class MockRetriever:
    """Return top chunks from the vector store for a given session."""

    def __init__(self, vector_store: InMemoryVectorStore) -> None:
        self.vector_store = vector_store

    async def retrieve(self, session_id: str, question: str, top_k: int) -> List[StoredChunk]:
        return await self.vector_store.retrieve(session_id, question, top_k)


class PromptBuilder:
    """Build a simple prompt using retrieved chunks."""

    def build(self, question: str, chunks: Sequence[StoredChunk]) -> str:
        context = "\n\n".join(chunk.content for chunk in chunks)
        return f"Context:\n{context}\n\nQuestion: {question}"


class MockLLM:
    """Return a deterministic answer prefix for testing."""

    async def generate(self, prompt: str, max_tokens: int) -> str:
        preview = prompt[:max_tokens]
        return f"MOCK_ANSWER: {preview}"


class RAGService:
    """Coordinate ingest and answer flows for RAG workflows using mock components."""

    _AUDIT_LOGGER = logging.getLogger("app.ingest.audit")

    def __init__(
        self,
        *,
        storage: Optional[InMemoryStorage] = None,
        text_extractor: Optional[PlainTextExtractor] = None,
        chunker: Optional[SimpleTextChunker] = None,
        embedding_provider: Optional[MockEmbeddingProvider] = None,
        vector_store: Optional[InMemoryVectorStore] = None,
        retriever: Optional[MockRetriever] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        llm: Optional[MockLLM] = None,
    ) -> None:
        self.storage = storage or InMemoryStorage()
        self.text_extractor = text_extractor or PlainTextExtractor()
        self.chunker = chunker or SimpleTextChunker()
        self.embedding_provider = embedding_provider or MockEmbeddingProvider()
        self.vector_store = vector_store or InMemoryVectorStore()
        self.retriever = retriever or MockRetriever(self.vector_store)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm = llm or MockLLM()

    async def ingest(self, session_id: str, files: Sequence[UploadFile]) -> Dict[str, Any]:
        durations: Dict[str, float] = {
            "save": 0.0,
            "extract": 0.0,
            "chunk": 0.0,
            "embed": 0.0,
            "store": 0.0,
        }
        added_chunks = 0

        for upload_file in files:
            if upload_file is None:
                continue

            start = time.perf_counter()
            content = await self.storage.save(session_id, upload_file)
            durations["save"] += time.perf_counter() - start

            start = time.perf_counter()
            text = self.text_extractor.extract(content, upload_file.filename)
            durations["extract"] += time.perf_counter() - start

            start = time.perf_counter()
            chunks = self.chunker.split(text)
            durations["chunk"] += time.perf_counter() - start
            chunk_count = len(chunks)

            if not chunks:
                self._log_ingest_audit(session_id, upload_file.filename, chunk_count)
                continue

            start = time.perf_counter()
            embeddings = await _maybe_await(self.embedding_provider.encode(chunks))
            durations["embed"] += time.perf_counter() - start

            start = time.perf_counter()
            chunk_ids = await _maybe_await(
                self.vector_store.add(session_id, upload_file.filename, chunks, embeddings)
            )
            durations["store"] += time.perf_counter() - start
            added_chunks += len(chunk_ids)

            self._log_ingest_audit(session_id, upload_file.filename, chunk_count)

        return {
            "added_chunks": added_chunks,
            "durations": durations,
            "files_processed": sum(1 for f in files if f is not None),
        }

    def _log_ingest_audit(self, session_id: str, filename: str | None, chunks: int) -> None:
        self._AUDIT_LOGGER.info(
            {
                "event": "ingest_file",
                "session_id": session_id,
                "filename": filename,
                "chunks": chunks,
            }
        )

    async def answer(
        self,
        session_id: str,
        question: str,
        top_k: int = 3,
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        retrieve_start = time.perf_counter()
        retrieved_chunks = await self.retriever.retrieve(session_id, question, top_k)
        retrieve_duration = time.perf_counter() - retrieve_start

        prompt = self.prompt_builder.build(question, retrieved_chunks)

        answer_start = time.perf_counter()
        answer_text = await self.llm.generate(prompt, max_tokens)
        answer_duration = time.perf_counter() - answer_start

        sources: List[Dict[str, Any]] = [
            {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
            for chunk in retrieved_chunks
        ]

        return {
            "answer": answer_text,
            "sources": sources,
            "meta": {
                "prompt": prompt,
                "retrieved_chunks": len(retrieved_chunks),
                "durations": {
                    "retrieve": retrieve_duration,
                    "generate": answer_duration,
                },
            },
        }


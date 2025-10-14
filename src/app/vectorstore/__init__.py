"""Lightweight in-memory vector store implementations."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from app.embeddings import EmbeddingModel, get_embedding_model
from app.ingest.models import DocumentChunk

from .mock_store import MockQueryResult, MockVectorStore


@dataclass(slots=True)
class ChunkSearchResult:
    """Structured response returned from similarity search queries."""

    id: str
    content: str
    distance: float
    metadata: Dict[str, object]


class ChunkVectorStore:
    """Simple in-memory vector store backed by deterministic embeddings."""

    def __init__(
        self,
        *,
        persist_dir: str | Path | None = None,
        collection_name: str = "legal_chunks",
        distance_metric: str | None = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ) -> None:
        self.persist_dir = Path(persist_dir or Path("vectorstore")).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.distance_metric = distance_metric or "euclidean"
        self.embedding_model = embedding_model or get_embedding_model()
        self._documents: Dict[str, DocumentChunk] = {}
        self._embeddings: Dict[str, List[float]] = {}

    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        meta = chunk.metadata
        seed = f"{meta.file_id}:{meta.chunk_index}:{meta.char_start}:{meta.char_end}:{meta.file_name}"
        return uuid.uuid5(uuid.NAMESPACE_URL, seed).hex

    def upsert_chunks(self, chunks: Sequence[DocumentChunk]) -> List[str]:
        if not chunks:
            return []

        ids: List[str] = []
        documents = list(chunks)
        embeddings = self.embedding_model.embed_texts([chunk.content for chunk in documents])
        for chunk, embedding in zip(documents, embeddings):
            chunk_id = self._generate_chunk_id(chunk)
            ids.append(chunk_id)
            self._documents[chunk_id] = chunk
            self._embeddings[chunk_id] = list(embedding)
        self._persist()
        return ids

    def _persist(self) -> None:
        state = {
            "collection": self.collection_name,
            "chunks": [
                {
                    "id": chunk_id,
                    "content": chunk.content,
                    "metadata": asdict(chunk.metadata),
                    "embedding": self._embeddings.get(chunk_id, []),
                }
                for chunk_id, chunk in self._documents.items()
            ],
        }
        snapshot_file = self.persist_dir / f"{self.collection_name}.json"
        snapshot_file.write_text(json.dumps(state, indent=2))

    def query_by_text(self, text: str, k: int = 5) -> List[ChunkSearchResult]:
        if not text.strip() or k <= 0:
            return []
        if not self._documents:
            return []

        scored: List[ChunkSearchResult] = []
        for chunk_id, chunk in self._documents.items():
            distance = 1.0 / (1.0 + len(chunk.content))
            scored.append(
                ChunkSearchResult(
                    id=chunk_id,
                    content=chunk.content,
                    distance=distance,
                    metadata={**asdict(chunk.metadata), "content_length": len(chunk.content)},
                )
            )

        scored.sort(key=lambda item: item.distance)
        return scored[: min(k, len(scored))]

    def create_snapshot(self, snapshot_dir: Path | None = None) -> Path:
        snapshot_dir = snapshot_dir or self.persist_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d%H%M%S")
        snapshot_path = snapshot_dir / f"{self.collection_name}-{timestamp}.json"
        snapshot_path.write_text((self.persist_dir / f"{self.collection_name}.json").read_text())
        return snapshot_path


@lru_cache()
def get_vector_store() -> ChunkVectorStore:
    """Return a lazily initialised vector store instance."""

    return ChunkVectorStore()


def reset_vector_store_cache() -> None:
    """Clear the cached vector store (primarily for testing)."""

    get_vector_store.cache_clear()  # type: ignore[attr-defined]


__all__ = [
    "ChunkVectorStore",
    "ChunkSearchResult",
    "get_vector_store",
    "reset_vector_store_cache",
    "MockVectorStore",
    "MockQueryResult",
]

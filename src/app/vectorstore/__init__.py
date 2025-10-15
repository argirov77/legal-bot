"""Vector store helpers backed by a persistent Chroma database."""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from app.embeddings import EmbeddingModel, get_embedding_model
from app.ingest.models import DocumentChunk

from .chroma_store import ChromaStore
from .mock_store import MockQueryResult, MockVectorStore


DEFAULT_COLLECTION_NAME = "legal_chunks"
DEFAULT_DISTANCE_METRIC = "cosine"


@dataclass(slots=True)
class ChunkSearchResult:
    """Structured response returned from similarity search queries."""

    id: str
    content: str
    distance: float
    metadata: Dict[str, object]


class ChunkVectorStore:
    """Persist document chunks in a Chroma vector store."""

    def __init__(
        self,
        *,
        persist_dir: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        distance_metric: str | None = None,
        embedding_model: Optional[EmbeddingModel] = None,
        chroma_store: Optional[ChromaStore] = None,
    ) -> None:
        base_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
        self.persist_dir = Path(base_dir).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.distance_metric = distance_metric or os.getenv(
            "CHROMA_DISTANCE_METRIC", DEFAULT_DISTANCE_METRIC
        )
        self.embedding_model = embedding_model or get_embedding_model()
        self._store = chroma_store or ChromaStore(self.persist_dir)
        self._store.create_collection(
            self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )

    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        meta = chunk.metadata
        seed = f"{meta.file_id}:{meta.chunk_index}:{meta.char_start}:{meta.char_end}:{meta.file_name}"
        return uuid.uuid5(uuid.NAMESPACE_URL, seed).hex

    def upsert_chunks(self, chunks: Sequence[DocumentChunk]) -> List[str]:
        if not chunks:
            return []

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, object]] = []

        for chunk in chunks:
            chunk_id = self._generate_chunk_id(chunk)
            ids.append(chunk_id)
            documents.append(chunk.content)
            metadata = asdict(chunk.metadata)
            metadata["content_length"] = len(chunk.content)
            metadatas.append(metadata)

        embeddings = self.embedding_model.embed_texts(documents)
        self._store.add(
            self.collection_name,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return ids

    def query_by_text(self, text: str, k: int = 5) -> List[ChunkSearchResult]:
        if not text.strip() or k <= 0:
            return []

        query_embeddings = self.embedding_model.embed_texts([text])
        if not query_embeddings:
            return []

        neighbours = self._store.query(
            self.collection_name, query_embedding=query_embeddings[0], k=k
        )

        results: List[ChunkSearchResult] = []
        for neighbour in neighbours:
            metadata = dict(neighbour.get("metadata", {}))
            results.append(
                ChunkSearchResult(
                    id=str(neighbour.get("id", "")),
                    content=str(neighbour.get("document", "")),
                    distance=float(neighbour.get("distance", 0.0)),
                    metadata=metadata,
                )
            )
        return results

    def create_snapshot(self, snapshot_dir: Path | None = None) -> Path:
        snapshot_dir = snapshot_dir or self.persist_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d%H%M%S")
        snapshot_path = snapshot_dir / f"{self.collection_name}-{timestamp}.json"

        collection = self._store.get_collection(self.collection_name)
        records = collection.get(include=["ids", "documents", "metadatas", "embeddings"])

        ids = records.get("ids", []) or []
        documents = records.get("documents", []) or []
        metadatas = records.get("metadatas", []) or []
        embeddings = records.get("embeddings", []) or []

        snapshot_payload = [
            {
                "id": item_id,
                "content": document,
                "metadata": metadata or {},
                "embedding": embedding or [],
            }
            for item_id, document, metadata, embedding in zip(
                ids, documents, metadatas, embeddings
            )
        ]

        snapshot_path.write_text(
            json.dumps(snapshot_payload, indent=2, ensure_ascii=False)
        )
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
    "ChromaStore",
    "MockVectorStore",
    "MockQueryResult",
]

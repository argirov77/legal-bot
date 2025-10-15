"""Vector store helpers backed by pluggable backends."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

from app.ingest.models import DocumentChunk

from .errors import VectorStoreUnavailableError
from .mock_store import InMemoryChunkVectorStore, MockQueryResult, MockVectorStore

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from app.embeddings import EmbeddingModel
    from .chroma_store import ChromaStore


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
        embedding_model: Optional["EmbeddingModel"] = None,
        chroma_store: Optional["ChromaStore"] = None,
        vector_store: Optional[MockVectorStore] = None,
    ) -> None:
        base_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
        self.persist_dir = Path(base_dir).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.distance_metric = distance_metric or os.getenv(
            "CHROMA_DISTANCE_METRIC", DEFAULT_DISTANCE_METRIC
        )

        if embedding_model is None:
            from app.embeddings import get_embedding_model

            embedding_model = get_embedding_model()
        self.embedding_model = embedding_model

        if chroma_store is None and vector_store is None:
            from .chroma_store import ChromaStore

            chroma_store = ChromaStore(self.persist_dir)

        self._store = chroma_store or vector_store
        if self._store is None:  # pragma: no cover - defensive
            raise RuntimeError("A vector store backend must be provided")

        try:
            self._store.create_collection(  # type: ignore[call-arg]
                self.collection_name,
                metadata={"hnsw:space": self.distance_metric},
            )
        except VectorStoreUnavailableError:
            raise
        except Exception as exc:  # pragma: no cover - unexpected backend failure
            raise VectorStoreUnavailableError(
                "Failed to initialise vector store collection",
                cause=exc,
            ) from exc

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
        try:
            self._store.add(  # type: ignore[attr-defined]
                self.collection_name,
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except VectorStoreUnavailableError:
            raise
        except Exception as exc:  # pragma: no cover - unexpected backend failure
            raise VectorStoreUnavailableError("Failed to upsert chunks into vector store", cause=exc) from exc
        return ids

    def query_by_text(self, text: str, k: int = 5) -> List[ChunkSearchResult]:
        if not text.strip() or k <= 0:
            return []

        query_embeddings = self.embedding_model.embed_texts([text])
        if not query_embeddings:
            return []

        try:
            neighbours = self._store.query(  # type: ignore[attr-defined]
                self.collection_name, query_embedding=query_embeddings[0], k=k
            )
        except VectorStoreUnavailableError:
            raise
        except Exception as exc:  # pragma: no cover - unexpected backend failure
            raise VectorStoreUnavailableError("Vector store query failed", cause=exc) from exc

        results: List[ChunkSearchResult] = []
        for neighbour in neighbours:
            if isinstance(neighbour, MockQueryResult):
                metadata = dict(neighbour.metadata)
                results.append(
                    ChunkSearchResult(
                        id=neighbour.id,
                        content=neighbour.document,
                        distance=float(neighbour.distance),
                        metadata=metadata,
                    )
                )
                continue

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

        collection = self._store.get_collection(self.collection_name)  # type: ignore[attr-defined]
        records = collection.get(include=["ids", "documents", "metadatas", "embeddings"])

        ids = records.get("ids", []) or []
        documents = records.get("documents", []) or []
        metadatas = records.get("metadatas", []) or []
        embeddings = records.get("embeddings", []) or []

        if ids and isinstance(ids[0], list):
            iterable = zip(ids[0], documents[0], metadatas[0], embeddings[0])
        else:
            iterable = zip(ids, documents, metadatas, embeddings)

        snapshot_payload = [
            {
                "id": item_id,
                "content": document,
                "metadata": metadata or {},
                "embedding": embedding or [],
            }
            for item_id, document, metadata, embedding in iterable
        ]

        snapshot_path.write_text(
            json.dumps(snapshot_payload, indent=2, ensure_ascii=False)
        )
        return snapshot_path


class _MockVectorStoreAdapter:
    """Adapter that wraps the in-memory implementation to match the public API."""

    def __init__(self, delegate: InMemoryChunkVectorStore) -> None:
        self._delegate = delegate

    def upsert_chunks(self, chunks: Sequence[DocumentChunk]) -> List[str]:
        return self._delegate.upsert_chunks(chunks)

    def query_by_text(self, text: str, k: int = 5) -> List[ChunkSearchResult]:
        results = self._delegate.query_by_text(text, k)
        converted: List[ChunkSearchResult] = []
        for item in results:
            converted.append(
                ChunkSearchResult(
                    id=item.id,
                    content=item.document,
                    distance=float(item.distance),
                    metadata=dict(item.metadata),
                )
            )
        return converted

    def create_snapshot(self, snapshot_dir: Path | None = None) -> Path:
        return self._delegate.create_snapshot(snapshot_dir)


@lru_cache()
def get_vector_store() -> ChunkVectorStore | _MockVectorStoreAdapter:
    """Return a lazily initialised vector store instance based on configuration."""

    backend = os.getenv("VECTOR_STORE", "mock").strip().lower()

    if backend == "mock":
        return _MockVectorStoreAdapter(InMemoryChunkVectorStore())

    if backend == "chroma":
        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "chroma_db")

        try:
            from chromadb.config import Settings
            import chromadb
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise VectorStoreUnavailableError(
                "VECTOR_STORE=chroma requires the 'chromadb' package to be installed",
                cause=exc,
            ) from exc

        settings = Settings(
            chroma_db_impl=os.getenv("CHROMA_DB_IMPL", "duckdb+parquet"),
            persist_directory=persist_dir,
        )
        try:
            client = chromadb.Client(settings)
        except Exception as exc:  # pragma: no cover - depends on chromadb runtime
            raise VectorStoreUnavailableError("Failed to initialise Chroma client", cause=exc) from exc

        from .chroma_store import ChromaStore

        try:
            store = ChromaStore(persist_dir, client=client)
        except VectorStoreUnavailableError:
            raise
        except Exception as exc:  # pragma: no cover - unexpected backend failure
            raise VectorStoreUnavailableError("Failed to initialise Chroma store", cause=exc) from exc
        return ChunkVectorStore(persist_dir=persist_dir, chroma_store=store)

    raise ValueError(f"Unsupported VECTOR_STORE backend: {backend!r}")


def reset_vector_store_cache() -> None:
    """Clear the cached vector store (primarily for testing)."""

    get_vector_store.cache_clear()  # type: ignore[attr-defined]


__all__ = [
    "ChunkVectorStore",
    "ChunkSearchResult",
    "get_vector_store",
    "reset_vector_store_cache",
    "InMemoryChunkVectorStore",
    "MockVectorStore",
    "MockQueryResult",
    "VectorStoreUnavailableError",
]

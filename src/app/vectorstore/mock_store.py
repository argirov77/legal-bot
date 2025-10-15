"""Simple in-memory vector store for testing purposes."""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence

from app.ingest.models import DocumentChunk

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MockQueryResult:
    """Container for similarity search results."""

    id: str
    document: str
    metadata: dict
    distance: float


@dataclass(slots=True)
class _MockStoredItem:
    """Internal representation of a stored vector."""

    id: str
    embedding: List[float]
    document: str
    metadata: dict


class _MockCollection:
    """Lightweight collection wrapper mimicking the Chroma API."""

    def __init__(self, items: List[_MockStoredItem]) -> None:
        self._items = items

    def get(self, include: Sequence[str] | None = None) -> Dict[str, List[object]]:
        include = list(include or [])
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[dict] = []
        embeddings: List[List[float]] = []

        for item in self._items:
            ids.append(item.id)
            documents.append(item.document)
            metadatas.append(item.metadata)
            embeddings.append(list(item.embedding))

        payload: Dict[str, List[object]] = {
            "ids": ids if "ids" in include else [],
            "documents": documents if "documents" in include else [],
            "metadatas": metadatas if "metadatas" in include else [],
            "embeddings": embeddings if "embeddings" in include else [],
        }
        return payload


class MockVectorStore:
    """A minimal in-memory vector store implementation."""

    def __init__(self) -> None:
        self._collections: Dict[str, List[_MockStoredItem]] = {}

    def create_collection(self, name: str, *, metadata: Optional[Dict[str, object]] = None) -> _MockCollection:
        """Create a new collection if it does not exist yet."""

        if name not in self._collections:
            self._collections[name] = []
        return _MockCollection(self._collections[name])

    def add(
        self,
        name: str,
        *,
        ids: Iterable[str],
        embeddings: Iterable[Sequence[float]],
        documents: Iterable[str],
        metadatas: Iterable[dict | None] | None = None,
    ) -> None:
        """Add documents to a collection."""

        if name not in self._collections:
            raise KeyError(f"Collection '{name}' does not exist")

        metadatas = metadatas or []

        id_list = list(ids)
        embedding_list = [list(map(float, embedding)) for embedding in embeddings]
        document_list = list(documents)
        metadata_list = list(metadatas) if metadatas else [None] * len(id_list)

        if not (
            len(id_list)
            == len(embedding_list)
            == len(document_list)
            == len(metadata_list)
        ):
            raise ValueError("All inputs must be of the same length")

        collection = self._collections[name]
        for idx, embedding, document, metadata in zip(
            id_list, embedding_list, document_list, metadata_list
        ):
            collection.append(
                _MockStoredItem(
                    id=idx,
                    embedding=list(embedding),
                    document=document,
                    metadata=dict(metadata or {}),
                )
            )

    def query(
        self,
        name: str,
        query_embedding: Sequence[float],
        k: int = 5,
    ) -> List[MockQueryResult]:
        """Return the *k* closest results to the provided embedding."""

        if k <= 0:
            return []
        if name not in self._collections:
            raise KeyError(f"Collection '{name}' does not exist")

        results = self._collections[name]
        if not results:
            return []

        scored_results: List[tuple[float, _MockStoredItem]] = []
        for item in results:
            distance = _euclidean_distance(query_embedding, item.embedding)
            scored_results.append((distance, item))

        top_results = sorted(scored_results, key=lambda item: item[0])[:k]
        return [
            MockQueryResult(
                id=result.id,
                document=result.document,
                metadata=result.metadata,
                distance=distance,
            )
            for distance, result in top_results
        ]

    def get_collection(self, name: str) -> _MockCollection:
        """Expose the collection wrapper for snapshotting."""

        if name not in self._collections:
            raise KeyError(f"Collection '{name}' does not exist")
        return _MockCollection(self._collections[name])


class PersistentMockVectorStore(MockVectorStore):
    """Mock vector store that persists its state on disk for reuse."""

    def __init__(self, persist_dir: Path) -> None:
        super().__init__()
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._data_path = self._persist_dir / "mock_store.json"
        self._load()

    def create_collection(self, name: str, *, metadata: Optional[Dict[str, object]] = None) -> _MockCollection:
        collection = super().create_collection(name, metadata=metadata)
        self._save()
        return collection

    def add(
        self,
        name: str,
        *,
        ids: Iterable[str],
        embeddings: Iterable[Sequence[float]],
        documents: Iterable[str],
        metadatas: Iterable[dict | None] | None = None,
    ) -> None:
        super().add(
            name,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        self._save()

    def _load(self) -> None:
        if not self._data_path.exists():
            return
        try:
            payload = json.loads(self._data_path.read_text(encoding="utf-8"))
        except Exception:
            LOGGER.warning("Failed to load mock vector store from %s", self._data_path)
            return

        for name, records in payload.items():
            collection = self._collections.setdefault(name, [])
            for record in records:
                try:
                    collection.append(
                        _MockStoredItem(
                            id=str(record.get("id", "")),
                            embedding=[float(value) for value in record.get("embedding", [])],
                            document=str(record.get("document", "")),
                            metadata=dict(record.get("metadata", {})),
                        )
                    )
                except Exception:
                    continue

    def _save(self) -> None:
        payload: Dict[str, List[dict]] = {}
        for name, items in self._collections.items():
            payload[name] = [
                {
                    "id": item.id,
                    "embedding": list(map(float, item.embedding)),
                    "document": item.document,
                    "metadata": dict(item.metadata),
                }
                for item in items
            ]
        tmp_path = self._data_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(self._data_path)


class _DeterministicEmbeddingModel:
    """Fallback embedding model that operates without external dependencies."""

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            lowered = text.lower()
            tokens = lowered.split()
            length = len(lowered)
            avg_word_len = sum(len(token) for token in tokens) / len(tokens) if tokens else 0.0
            vowel_ratio = (
                sum(1 for char in lowered if char in "aeiou") / length if length else 0.0
            )
            chunk_count = lowered.count("chunk")
            long_count = lowered.count("long")
            embeddings.append(
                [
                    length / 50.0,
                    avg_word_len / 10.0,
                    vowel_ratio,
                    float(chunk_count),
                    float(long_count),
                ]
            )
        return embeddings


class EmbeddingModelLike(Protocol):
    """Protocol describing the embedding interface used by the stores."""

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        ...


class InMemoryChunkVectorStore:
    """Store document chunks in memory using deterministic embeddings."""

    DEFAULT_COLLECTION_NAME = "legal_chunks"

    def __init__(
        self,
        *,
        collection_name: str | None = None,
        embedding_model: Optional[EmbeddingModelLike] = None,
        vector_store: Optional[MockVectorStore] = None,
    ) -> None:
        self.collection_name = collection_name or self.DEFAULT_COLLECTION_NAME
        self.embedding_model = embedding_model or _DeterministicEmbeddingModel()
        self._store = vector_store or MockVectorStore()
        self._store.create_collection(self.collection_name)

    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        metadata = chunk.metadata
        seed = (
            f"{metadata.file_id}:{metadata.chunk_index}:{metadata.char_start}:{metadata.char_end}:{metadata.file_name}"
        )
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
            metadata = chunk.metadata
            metadata_dict = {
                "file_id": metadata.file_id,
                "file_name": metadata.file_name,
                "page": metadata.page,
                "chunk_index": metadata.chunk_index,
                "char_start": metadata.char_start,
                "char_end": metadata.char_end,
                "language": metadata.language,
                "session_id": getattr(metadata, "session_id", None),
                "content_length": len(chunk.content),
            }
            metadatas.append(metadata_dict)

        embeddings = self.embedding_model.embed_texts(documents)
        self._store.add(
            self.collection_name,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return ids

    def query_by_text(self, text: str, k: int = 5) -> List[MockQueryResult]:
        if not text.strip() or k <= 0:
            return []

        query_embeddings = self.embedding_model.embed_texts([text])
        if not query_embeddings:
            return []

        return self._store.query(
            self.collection_name,
            query_embedding=query_embeddings[0],
            k=k,
        )

    def create_snapshot(self, snapshot_dir: Path | None = None) -> Path:
        snapshot_dir = snapshot_dir or Path.cwd() / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d%H%M%S")
        snapshot_path = snapshot_dir / f"{self.collection_name}-{timestamp}.json"

        collection = self._store.get_collection(self.collection_name)
        records = collection.get(include=["ids", "documents", "metadatas", "embeddings"])

        ids = records.get("ids", [])
        documents = records.get("documents", [])
        metadatas = records.get("metadatas", [])
        embeddings = records.get("embeddings", [])

        if ids and isinstance(ids[0], list):
            iterable = zip(ids[0], documents[0], metadatas[0], embeddings[0])
        else:
            iterable = zip(ids, documents, metadatas, embeddings)

        payload = [
            {
                "id": item_id,
                "content": document,
                "metadata": metadata or {},
                "embedding": embedding or [],
            }
            for item_id, document, metadata, embedding in iterable
        ]

        snapshot_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return snapshot_path


def _euclidean_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of the same dimension")
    return sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)) ** 0.5


__all__ = [
    "InMemoryChunkVectorStore",
    "MockQueryResult",
    "MockVectorStore",
    "PersistentMockVectorStore",
]

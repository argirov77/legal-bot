from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pytest

from app.ingest.embedding_pipeline import ChunkEmbeddingPipeline
from app.ingest.models import ChunkMetadata, DocumentChunk
from app.vectorstore import ChunkSearchResult, ChunkVectorStore


@dataclass
class _DummyEmbeddingModel:
    """Simple deterministic embedding model used for tests."""

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - trivial
        return [[float(len(text))] for text in texts]


class _InMemoryCollection:
    def __init__(self) -> None:
        self._records: List[Dict[str, object]] = []

    def upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, object]],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        self._records = [
            {
                "id": chunk_id,
                "document": document,
                "metadata": metadata,
                "embedding": list(embedding),
            }
            for chunk_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings)
        ]

    def count(self) -> int:
        return len(self._records)

    def query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int,
        include: Sequence[str],
    ) -> Dict[str, List[List[object]]]:
        if not self._records:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        def _score(record: Dict[str, object]) -> float:
            metadata = record.get("metadata") or {}
            length = metadata.get("content_length")
            if not isinstance(length, (int, float)):
                length = len(record.get("document", ""))
            return 1.0 / (float(length) + 1.0)

        ranked = sorted(self._records, key=_score)[:n_results]

        return {
            "ids": [[record["id"] for record in ranked]],
            "documents": [[record["document"] for record in ranked]],
            "metadatas": [[record["metadata"] for record in ranked]],
            "distances": [[_score(record) for record in ranked]],
        }


class _InMemoryClient:
    def __init__(self, persist_dir: Path, collection: _InMemoryCollection) -> None:
        self.persist_dir = persist_dir
        self._collection = collection
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    def get_or_create_collection(self, *, name: str, metadata: Dict[str, object]):  # pragma: no cover - simple pass-through
        return self._collection

    def persist(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        marker = self.persist_dir / ".persisted"
        marker.write_text("ok", encoding="utf-8")
@pytest.fixture()
def vector_store(tmp_path: Path) -> ChunkVectorStore:
    collection = _InMemoryCollection()
    client = _InMemoryClient(tmp_path / "chroma", collection)
    return ChunkVectorStore(
        persist_dir=tmp_path / "chroma",
        distance_metric="l2",
        embedding_model=_DummyEmbeddingModel(),
        client=client,
        collection=collection,
    )


def _make_chunk(index: int, content: str = "Example text", file_id: str = "file-1") -> DocumentChunk:
    metadata = ChunkMetadata(
        file_id=file_id,
        file_name="document.txt",
        page=1,
        chunk_index=index,
        char_start=index * 10,
        char_end=index * 10 + len(content),
        language="ru",
    )
    return DocumentChunk(content=content, metadata=metadata)


def test_upsert_and_query_returns_results(vector_store: ChunkVectorStore) -> None:
    chunks = [
        _make_chunk(0, content="short"),
        _make_chunk(1, content="a little bit longer"),
        _make_chunk(2, content="the longest chunk text so far"),
    ]

    pipeline = ChunkEmbeddingPipeline(vector_store=vector_store)
    pipeline.run(chunks)

    results = vector_store.query_by_text("long chunk", k=2)
    assert len(results) == 2
    # Shortest distance should belong to the longest chunk because of our dummy embeddings
    assert results[0].metadata["chunk_index"] == 2
    assert isinstance(results[0], ChunkSearchResult)


def test_create_snapshot_creates_directory(vector_store: ChunkVectorStore, tmp_path: Path) -> None:
    vector_store.upsert_chunks([_make_chunk(0)])

    snapshot_root = tmp_path / "snapshots"
    snapshot_path = vector_store.create_snapshot(snapshot_root)

    assert snapshot_path.exists()
    assert snapshot_path.parent == snapshot_root

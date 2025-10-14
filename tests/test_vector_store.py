from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pytest

from app.ingest.embedding_pipeline import ChunkEmbeddingPipeline
from app.ingest.models import ChunkMetadata, DocumentChunk
from app.vectorstore import ChunkSearchResult, ChunkVectorStore


@dataclass
class _DummyEmbeddingModel:
    """Simple deterministic embedding model used for tests."""

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - trivial
        return [[float(len(text))] for text in texts]


@pytest.fixture()
def vector_store(tmp_path: Path) -> ChunkVectorStore:
    return ChunkVectorStore(persist_dir=tmp_path / "chroma", distance_metric="l2", embedding_model=_DummyEmbeddingModel())


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

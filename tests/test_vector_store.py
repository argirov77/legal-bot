from __future__ import annotations

from pathlib import Path

import pytest

from app.ingest.embedding_pipeline import ChunkEmbeddingPipeline
from app.ingest.models import ChunkMetadata, DocumentChunk
from app.vectorstore import ChunkSearchResult, get_vector_store, reset_vector_store_cache


@pytest.fixture()
def vector_store(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setenv("VECTOR_STORE", "mock")
    reset_vector_store_cache()
    store = get_vector_store()
    yield store
    reset_vector_store_cache()


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


def test_upsert_and_query_returns_results(vector_store: object) -> None:
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


def test_create_snapshot_creates_directory(vector_store: object, tmp_path: Path) -> None:
    vector_store.upsert_chunks([_make_chunk(0)])

    snapshot_root = tmp_path / "snapshots"
    snapshot_path = vector_store.create_snapshot(snapshot_root)

    assert snapshot_path.exists()
    assert snapshot_path.parent == snapshot_root

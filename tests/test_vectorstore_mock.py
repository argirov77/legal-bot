"""Tests for the in-memory mock vector store."""
from app.vectorstore import MockVectorStore


def test_query_returns_results_ordered_by_distance() -> None:
    store = MockVectorStore()
    store.create_collection("test")

    store.add(
        "test",
        ids=["doc-1", "doc-2", "doc-3"],
        embeddings=[
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ],
        documents=["Document 1", "Document 2", "Document 3"],
        metadatas=[{"page": 1}, {"page": 2}, {"page": 3}],
    )

    results = store.query("test", query_embedding=[1.0, 0.0], k=3)
    assert [result.id for result in results] == ["doc-1", "doc-3", "doc-2"]
    assert results[0].metadata == {"page": 1}


def test_query_limits_number_of_results() -> None:
    store = MockVectorStore()
    store.create_collection("test")
    store.add(
        "test",
        ids=["doc-1", "doc-2"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["Document 1", "Document 2"],
        metadatas=[{"page": 1}, {"page": 2}],
    )

    results = store.query("test", query_embedding=[1.0, 0.0], k=1)
    assert len(results) == 1
    assert results[0].id == "doc-1"

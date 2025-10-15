"""End-to-end tests for ingesting and querying via the public API."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.rag import RAGService, get_rag_service


@pytest.mark.heavy
def test_ingest_and_query_round_trip() -> None:
    """Uploading a document and querying it returns deterministic sources."""

    service = RAGService()
    app.dependency_overrides[get_rag_service] = lambda: service

    client = TestClient(app)
    try:
        ingest_files = {
            "files": (
                "policy.txt",
                b"The service agreement allows cancellation with two weeks notice.",
                "text/plain",
            )
        }
        ingest_response = client.post("/sessions/e2e/ingest", files=ingest_files)
        assert ingest_response.status_code == 200

        ingest_payload = ingest_response.json()
        assert ingest_payload["status"] == "ok"
        assert ingest_payload["session_id"] == "e2e"
        assert ingest_payload["added_chunks"] >= 1
        assert ingest_payload["saved_files"] == ["e2e/policy.txt"]

        query_request = {
            "question": "What are the cancellation terms?",
            "top_k": 1,
            "max_tokens": 20,
        }
        query_response = client.post("/sessions/e2e/query", json=query_request)
        assert query_response.status_code == 200

        query_payload = query_response.json()
        assert query_payload["session_id"] == "e2e"
        assert query_payload["question"] == query_request["question"]
        assert query_payload["top_k"] == query_request["top_k"]
        assert query_payload["max_tokens"] == query_request["max_tokens"]
        assert isinstance(query_payload["answer"], str)
        assert "cancellation" in query_payload["answer"].lower()

        sources = query_payload["sources"]
        assert isinstance(sources, list)
        assert len(sources) == 1

        first_source = sources[0]
        assert set(first_source.keys()) == {"id", "content", "metadata"}
        assert first_source["content"]

        metadata = first_source["metadata"]
        assert metadata["filename"] == "policy.txt"
        assert metadata["session_id"] == "e2e"
        assert isinstance(metadata["chunk_index"], int)
    finally:
        app.dependency_overrides.clear()

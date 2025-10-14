from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.services.rag import RAGService, get_rag_service


def test_ingest_then_query_returns_sources() -> None:
    service = RAGService()
    app.dependency_overrides[get_rag_service] = lambda: service
    client = TestClient(app)

    files = {"files": ("contract.txt", b"The agreement lasts for two years.", "text/plain")}
    ingest_response = client.post("/sessions/demo/ingest", files=files)
    assert ingest_response.status_code == 200

    ingest_payload = ingest_response.json()
    assert ingest_payload["status"] == "ok"
    assert ingest_payload["session_id"] == "demo"
    assert ingest_payload["added_chunks"] >= 1
    assert ingest_payload["saved_files"] == ["demo/contract.txt"]

    query_response = client.post(
        "/sessions/demo/query",
        json={"question": "How long is the agreement?", "top_k": 1, "max_tokens": 10},
    )
    app.dependency_overrides.clear()

    assert query_response.status_code == 200
    query_payload = query_response.json()
    assert query_payload["session_id"] == "demo"
    assert query_payload["question"] == "How long is the agreement?"
    assert isinstance(query_payload["answer"], str)
    assert query_payload["answer"]
    assert query_payload["top_k"] == 1
    assert query_payload["max_tokens"] == 10
    assert isinstance(query_payload["sources"], list)
    assert len(query_payload["sources"]) == 1
    first_source = query_payload["sources"][0]
    assert first_source["metadata"]["filename"] == "contract.txt"
    assert "agreement" in first_source["content"].lower()


def test_query_prefers_most_recent_chunks() -> None:
    service = RAGService()
    app.dependency_overrides[get_rag_service] = lambda: service
    client = TestClient(app)

    try:
        first_files = {
            "files": ("initial.txt", b"Historical terms should be superseded.", "text/plain")
        }
        second_files = {
            "files": ("update.txt", b"The latest addendum changes the outcome.", "text/plain")
        }

        first_ingest = client.post("/sessions/demo/ingest", files=first_files)
        assert first_ingest.status_code == 200

        second_ingest = client.post("/sessions/demo/ingest", files=second_files)
        assert second_ingest.status_code == 200

        query_response = client.post(
            "/sessions/demo/query",
            json={"question": "What is current?", "top_k": 1},
        )
        assert query_response.status_code == 200

        payload = query_response.json()
        assert payload["sources"], "Expected at least one retrieved chunk"
        latest_source = payload["sources"][0]
        assert latest_source["metadata"]["filename"] == "update.txt"
        assert "latest addendum" in latest_source["content"].lower()
    finally:
        app.dependency_overrides.clear()

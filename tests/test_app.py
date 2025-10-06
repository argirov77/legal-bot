from fastapi.testclient import TestClient

from app.main import app
from app.vectorstore import ChunkSearchResult, get_vector_store


def test_read_root_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == "ok"


def test_query_endpoint_returns_results() -> None:
    class DummyStore:
        def query_by_text(self, text: str, k: int = 5):
            return [ChunkSearchResult(id="chunk-1", content="answer", distance=0.05, metadata={"chunk_index": 1})]

    app.dependency_overrides[get_vector_store] = lambda: DummyStore()
    client = TestClient(app)
    response = client.post("/query", json={"text": "question", "k": 1})
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["id"] == "chunk-1"

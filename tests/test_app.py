from fastapi.testclient import TestClient

from app.embeddings import get_embedding_model
from app.main import app, _get_vector_store_or_503
from app.vectorstore import (
    ChunkSearchResult,
    VectorStoreUnavailableError,
    get_vector_store,
)
from app.llm_provider import LLM, LLMStatus


def test_read_root_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == "ok"


def test_healthz_returns_ok(monkeypatch) -> None:
    status = LLMStatus(model_loaded=True, model_name="dummy-model", device="cpu")
    monkeypatch.setattr("app.main.get_llm_status", lambda: status)

    client = TestClient(app)
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.text == "ok"


def test_healthz_returns_503_when_model_unavailable(monkeypatch) -> None:
    status = LLMStatus(
        model_loaded=False,
        model_name="stub",
        device="cpu",
        error="Model not available",
    )
    monkeypatch.setattr("app.main.get_llm_status", lambda: status)

    client = TestClient(app)
    response = client.get("/healthz")

    assert response.status_code == 503
    assert response.json()["detail"] == "Model not available"


def test_readyz_returns_ok_with_mock_backend(monkeypatch) -> None:
    monkeypatch.delenv("VECTOR_STORE", raising=False)
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.text == "ok"


def test_readyz_checks_dependencies(monkeypatch) -> None:
    class DummyEmbeddingModel:
        def embed_texts(self, texts: list[str]):
            return [[0.0 for _ in texts]]

    class DummyVectorStore:
        def query_by_text(self, text: str, k: int = 1):
            return []

    monkeypatch.setenv("VECTOR_STORE", "chroma")
    app.dependency_overrides[get_embedding_model] = lambda: DummyEmbeddingModel()
    app.dependency_overrides[get_vector_store] = lambda: DummyVectorStore()

    client = TestClient(app)
    try:
        response = client.get("/readyz")
    finally:
        app.dependency_overrides.clear()
        monkeypatch.delenv("VECTOR_STORE", raising=False)

    assert response.status_code == 200
    assert response.text == "ok"


def test_readyz_returns_error_when_dependency_unavailable(monkeypatch) -> None:
    class FailingEmbeddingModel:
        def embed_texts(self, texts: list[str]):
            raise RuntimeError("embedding boom")

    class DummyVectorStore:
        def query_by_text(self, text: str, k: int = 1):
            return []

    monkeypatch.setenv("VECTOR_STORE", "chroma")
    app.dependency_overrides[get_embedding_model] = lambda: FailingEmbeddingModel()
    app.dependency_overrides[get_vector_store] = lambda: DummyVectorStore()

    client = TestClient(app)
    try:
        response = client.get("/readyz")
    finally:
        app.dependency_overrides.clear()
        monkeypatch.delenv("VECTOR_STORE", raising=False)

    assert response.status_code == 503
    assert "embedding_model_unavailable" in response.json()["detail"]


def test_readyz_returns_error_when_vector_store_unavailable(monkeypatch) -> None:
    class DummyEmbeddingModel:
        def embed_texts(self, texts: list[str]):
            return [[0.0 for _ in texts]]

    class BrokenVectorStore:
        def query_by_text(self, text: str, k: int = 1):
            raise VectorStoreUnavailableError("chroma down")

    monkeypatch.setenv("VECTOR_STORE", "chroma")
    app.dependency_overrides[get_embedding_model] = lambda: DummyEmbeddingModel()
    app.dependency_overrides[get_vector_store] = lambda: BrokenVectorStore()

    client = TestClient(app)
    try:
        response = client.get("/readyz")
    finally:
        app.dependency_overrides.clear()
        monkeypatch.delenv("VECTOR_STORE", raising=False)

    assert response.status_code == 503
    assert "vector_store_unavailable" in response.json()["detail"]


def test_model_status_endpoint_reports_model_state(monkeypatch) -> None:
    status = LLMStatus(model_loaded=True, model_name="dummy-model", device="cpu")
    monkeypatch.setattr("app.main.get_llm_status", lambda: status)

    client = TestClient(app)
    response = client.get("/model_status")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "model_loaded": True,
        "model_name": "dummy-model",
        "device": "cpu",
    }


def test_model_health_endpoint_includes_reason(monkeypatch) -> None:
    status = LLMStatus(
        model_loaded=False,
        model_name="dummy-model",
        device="cpu",
        error="failed to import torch",
    )
    monkeypatch.setattr("app.main.get_llm_status", lambda: status)

    client = TestClient(app)
    response = client.get("/healthz/model")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "model_loaded": False,
        "device": "cpu",
        "name": "dummy-model",
        "reason": "failed to import torch",
    }


def test_query_endpoint_returns_results() -> None:
    class DummyStore:
        def query_by_text(self, text: str, k: int = 5):
            return [ChunkSearchResult(id="chunk-1", content="answer", distance=0.05, metadata={"chunk_index": 1})]

    app.dependency_overrides[_get_vector_store_or_503] = lambda: DummyStore()
    client = TestClient(app)
    response = client.post("/query", json={"text": "question", "k": 1})
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["id"] == "chunk-1"


def test_query_endpoint_returns_503_on_vector_store_error() -> None:
    class BrokenStore:
        def query_by_text(self, text: str, k: int = 5):
            raise VectorStoreUnavailableError("vector store offline")

    app.dependency_overrides[_get_vector_store_or_503] = lambda: BrokenStore()
    client = TestClient(app)
    response = client.post("/query", json={"text": "question", "k": 1})
    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["detail"] == "vector store offline"

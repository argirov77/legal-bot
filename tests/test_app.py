from app.main import QueryRequest, QueryResponse, app, query_vector_store, read_root
from app.vectorstore import ChunkSearchResult, get_vector_store


def test_read_root_returns_ok() -> None:
    assert read_root() == "ok"


def test_query_endpoint_returns_results() -> None:
    class DummyStore:
        def query_by_text(self, text: str, k: int = 5):
            return [ChunkSearchResult(id="chunk-1", content="answer", distance=0.05, metadata={"chunk_index": 1})]

    app.dependency_overrides[get_vector_store] = lambda: DummyStore()
    request = QueryRequest(text="question", k=1)
    response: QueryResponse = query_vector_store(request, store=DummyStore())
    app.dependency_overrides.clear()

    assert response.results[0].id == "chunk-1"

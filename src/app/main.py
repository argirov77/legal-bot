import os
from typing import Any, Callable, TypeVar

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.api.rag import router as rag_router
from app.logging_config import configure_logging
from app.embeddings import get_embedding_model
from app.vectorstore import ChunkSearchResult, ChunkVectorStore, get_vector_store

# TODO: Wire LLMProvider dependency to expose BgGPT/Gemma completion endpoints.

configure_logging()

app = FastAPI(title="Legal Bot API")
app.include_router(rag_router)


T = TypeVar("T")


def _resolve_dependency(factory: Callable[[], T]) -> T:
    """Resolve a dependency while respecting FastAPI overrides."""

    override: Any | None = app.dependency_overrides.get(factory)
    resolved: Any = override if override is not None else factory
    return resolved() if callable(resolved) else resolved


def _is_mock_vector_store() -> bool:
    return os.getenv("VECTOR_STORE", "mock").strip().lower() == "mock"


@app.get("/", response_class=PlainTextResponse)
def read_root() -> str:
    """Healthcheck endpoint for the service."""
    return "ok"


@app.get("/healthz", response_class=PlainTextResponse)
def healthcheck() -> str:
    """Liveness probe used by container orchestrators."""

    return "ok"


@app.get("/readyz", response_class=PlainTextResponse)
def readiness_probe() -> str:
    """Readiness probe that ensures external dependencies are available."""

    if _is_mock_vector_store():
        return "ok"

    errors: list[str] = []

    try:
        embedding_model = _resolve_dependency(get_embedding_model)
        embedding_model.embed_texts(["__readyz__"])
    except Exception as exc:  # pragma: no cover - defensive guard
        errors.append(f"embedding_model_unavailable: {exc}")

    try:
        store = _resolve_dependency(get_vector_store)
        if hasattr(store, "query_by_text"):
            store.query_by_text("__readyz__", k=1)
        else:
            raise AttributeError("vector store is missing 'query_by_text'")
    except Exception as exc:  # pragma: no cover - defensive guard
        errors.append(f"vector_store_unavailable: {exc}")

    if errors:
        raise HTTPException(status_code=503, detail="; ".join(errors))

    return "ok"


class QueryRequest(BaseModel):
    text: str = Field(..., description="Query text to search for similar chunks.")
    k: int = Field(5, ge=1, le=50, description="Number of results to return.")


class QueryResponseItem(BaseModel):
    id: str
    distance: float
    content: str
    metadata: dict


class QueryResponse(BaseModel):
    results: list[QueryResponseItem]


def _serialize_result(result: ChunkSearchResult) -> QueryResponseItem:
    return QueryResponseItem(
        id=result.id,
        distance=result.distance,
        content=result.content,
        metadata=result.metadata,
    )


@app.post("/query", response_model=QueryResponse)
def query_vector_store(
    request: QueryRequest,
    store: ChunkVectorStore = Depends(get_vector_store),
) -> QueryResponse:
    """Query the vector store by plain text and return similar chunks."""

    results = store.query_by_text(request.text, k=request.k)
    return QueryResponse(results=[_serialize_result(item) for item in results])

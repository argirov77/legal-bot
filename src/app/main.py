import logging
import os
from typing import Any, Callable, TypeVar

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.api.rag import router as rag_router
from app.logging_config import configure_logging
from app.embeddings import get_embedding_model
from app.vectorstore import (
    ChunkSearchResult,
    ChunkVectorStore,
    VectorStoreUnavailableError,
    get_vector_store,
)
from app.llm_provider import get_llm, get_llm_status, load_llm_on_startup

# TODO: Wire LLMProvider dependency to expose BgGPT/Gemma completion endpoints.

configure_logging()

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Legal Bot API")
app.include_router(rag_router)


@app.on_event("startup")
async def _startup_model_loader() -> None:
    """Optionally preload the local LLM when requested via environment flags."""

    load_llm_on_startup()


T = TypeVar("T")


def _resolve_dependency(factory: Callable[[], T]) -> T:
    """Resolve a dependency while respecting FastAPI overrides."""

    override: Any | None = app.dependency_overrides.get(factory)
    resolved: Any = override if override is not None else factory
    return resolved() if callable(resolved) else resolved


def _is_mock_vector_store() -> bool:
    return os.getenv("VECTOR_STORE", "mock").strip().lower() == "mock"


def _get_vector_store_or_503() -> ChunkVectorStore:
    try:
        return get_vector_store()
    except VectorStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/", response_class=PlainTextResponse)
def read_root() -> str:
    """Healthcheck endpoint for the service."""
    return "ok"


@app.get("/healthz", response_class=PlainTextResponse)
def healthcheck() -> str:
    """Liveness probe used by container orchestrators."""
    status = get_llm_status()
    if not status.model_loaded:
        detail = status.error or "LLM model is not loaded"
        raise HTTPException(status_code=503, detail=detail)

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
        try:
            store = _resolve_dependency(get_vector_store)
        except VectorStoreUnavailableError as exc:
            errors.append(f"vector_store_unavailable: {exc}")
            store = None

        if store is None:
            raise HTTPException(status_code=503, detail="; ".join(errors))
        if hasattr(store, "query_by_text"):
            try:
                store.query_by_text("__readyz__", k=1)
            except VectorStoreUnavailableError as exc:
                errors.append(f"vector_store_unavailable: {exc}")
        else:
            raise AttributeError("vector store is missing 'query_by_text'")
    except Exception as exc:  # pragma: no cover - defensive guard
        errors.append(f"vector_store_unavailable: {exc}")

    if errors:
        raise HTTPException(status_code=503, detail="; ".join(errors))

    return "ok"


@app.get("/healthz/model")
def model_healthcheck() -> dict[str, object]:
    """Expose eager/lazy model loading status and device placement."""

    status = get_llm_status()
    payload: dict[str, object] = {
        "model_loaded": status.model_loaded,
        "device": status.device,
        "name": status.model_name,
    }
    if status.error:
        payload["reason"] = status.error
    return payload


@app.get("/model_status")
def model_status() -> dict[str, object]:
    """Expose lazy model loading status for backwards compatibility."""

    status = get_llm_status()
    payload: dict[str, object] = {
        "model_loaded": status.model_loaded,
        "model_name": status.model_name,
        "device": status.device,
    }
    if status.error:
        payload["reason"] = status.error
    return payload


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
    store: ChunkVectorStore = Depends(_get_vector_store_or_503),
) -> QueryResponse:
    """Query the vector store by plain text and return similar chunks."""

    try:
        results = store.query_by_text(request.text, k=request.k)
    except VectorStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return QueryResponse(results=[_serialize_result(item) for item in results])

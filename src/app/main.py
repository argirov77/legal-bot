from fastapi import Depends, FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.vectorstore import ChunkSearchResult, ChunkVectorStore, get_vector_store

app = FastAPI(title="Legal Bot API")

# include ingest router
from app.ingest import router as ingest_router

app.include_router(ingest_router)


@app.get("/", response_class=PlainTextResponse)
def read_root() -> str:
    """Healthcheck endpoint for the service."""
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

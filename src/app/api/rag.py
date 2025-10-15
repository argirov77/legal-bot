"""API router exposing ingest and query endpoints for the RAG service."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.rag import AnswerResult, IngestResult, RAGService, StoredChunk, get_rag_service
from app.vectorstore import VectorStoreUnavailableError

router = APIRouter(prefix="/sessions", tags=["rag"])


class IngestResponse(BaseModel):
    """Response body returned from the ingest endpoint."""

    status: str
    session_id: str
    added_chunks: int
    saved_files: list[str]
    duration_seconds: float


class QueryRequest(BaseModel):
    """Request body accepted by the query endpoint."""

    question: str = Field(..., min_length=1, description="User question to ask against the knowledge base.")
    top_k: int = Field(3, ge=1, le=20, description="How many chunks should be considered.")
    max_tokens: int | None = Field(
        None,
        ge=1,
        le=2048,
        description="Maximum number of whitespace-delimited tokens returned in the answer.",
    )


class AnswerSource(BaseModel):
    """Individual source chunk returned in an answer."""

    id: str
    content: str
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    """Response payload for the query endpoint."""

    session_id: str
    question: str
    answer: str
    top_k: int
    max_tokens: int | None
    sources: list[AnswerSource]


async def _run_ingest(
    session_id: str,
    files: list[UploadFile],
    rag_service: RAGService,
) -> IngestResponse:
    result: IngestResult = await rag_service.ingest(session_id, files)
    return IngestResponse(
        status="ok",
        session_id=result.session_id,
        added_chunks=result.chunk_count,
        saved_files=result.saved_files,
        duration_seconds=result.duration_seconds,
    )


def _serialise_sources(chunks: list[StoredChunk]) -> list[AnswerSource]:
    serialised: list[AnswerSource] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        metadata.setdefault("score", chunk.score)
        if "filename" not in metadata and "file_name" in metadata:
            metadata["filename"] = metadata["file_name"]
        serialised.append(AnswerSource(id=chunk.id, content=chunk.content, metadata=metadata))
    return serialised


@router.post("/{session_id}/ingest", response_model=IngestResponse)
async def ingest_documents(
    session_id: str,
    files: list[UploadFile] = File(...),
    rag_service: RAGService = Depends(get_rag_service),
) -> IngestResponse:
    """Ingest one or more documents for the provided session."""

    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be provided")

    try:
        return await _run_ingest(session_id, files, rag_service)
    except VectorStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{session_id}/query", response_model=QueryResponse)
async def query_documents(
    session_id: str,
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """Query the ingested documents for a given session."""

    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty")

    try:
        result: AnswerResult = rag_service.answer(
            session_id,
            request.question,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
        )
    except VectorStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return QueryResponse(
        session_id=result.session_id,
        question=result.question,
        answer=result.answer,
        top_k=result.top_k,
        max_tokens=result.max_tokens,
        sources=_serialise_sources(result.sources),
    )

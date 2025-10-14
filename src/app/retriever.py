"""Utilities for retrieving relevant context from the vector store."""
from __future__ import annotations

from typing import Dict, List, Protocol, Sequence


class EmbeddingProvider(Protocol):
    """Protocol describing the embedding provider contract."""

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        """Return embeddings for the provided texts."""


class VectorStore(Protocol):
    """Protocol describing the vector store contract used by the retriever."""

    def query(self, *, session_id: str, embedding: List[float], top_k: int) -> List[Dict]:
        """Execute a similarity search against the vector store."""


class Retriever:
    """High level orchestrator for retrieving context snippets."""

    def __init__(self, vectorstore: VectorStore, embedding_provider: EmbeddingProvider) -> None:
        self._vectorstore = vectorstore
        self._embedding_provider = embedding_provider

    async def retrieve(self, session_id: str, question: str, top_k: int = 5) -> List[Dict]:
        """Return the top matching chunks for the supplied question."""

        if top_k <= 0:
            return []

        embeddings = self._embedding_provider.encode([question])
        if not embeddings:
            return []

        query_embedding = embeddings[0]
        results = self._vectorstore.query(session_id=session_id, embedding=query_embedding, top_k=top_k)
        return list(results)

"""Chroma vector store adapter."""
from __future__ import annotations

from typing import Iterable, Sequence


class ChromaStore:
    """Adapter around a Chroma vector database."""

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError

    def create_collection(self, name: str) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError

    def add(
        self,
        name: str,
        *,
        ids: Iterable[str],
        embeddings: Iterable[Sequence[float]],
        documents: Iterable[str],
        metadatas: Iterable[dict | None] | None = None,
    ) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError

    def query(
        self,
        name: str,
        query_embedding: Sequence[float],
        k: int = 5,
    ):  # pragma: no cover - placeholder
        raise NotImplementedError

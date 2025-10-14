"""Simple in-memory vector store for testing purposes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class MockQueryResult:
    """Container for similarity search results."""

    id: str
    document: str
    metadata: dict
    distance: float


class MockVectorStore:
    """A minimal in-memory vector store implementation."""

    def __init__(self) -> None:
        self._collections: Dict[str, List[Tuple[str, Sequence[float], str, dict]]] = {}

    def create_collection(self, name: str) -> None:
        """Create a new collection if it does not exist yet."""

        if name not in self._collections:
            self._collections[name] = []

    def add(
        self,
        name: str,
        *,
        ids: Iterable[str],
        embeddings: Iterable[Sequence[float]],
        documents: Iterable[str],
        metadatas: Iterable[dict | None] | None = None,
    ) -> None:
        """Add documents to a collection."""

        if name not in self._collections:
            raise KeyError(f"Collection '{name}' does not exist")

        metadatas = metadatas or []

        id_list = list(ids)
        embedding_list = list(embeddings)
        document_list = list(documents)
        metadata_list = list(metadatas) if metadatas else [None] * len(id_list)

        if not (
            len(id_list)
            == len(embedding_list)
            == len(document_list)
            == len(metadata_list)
        ):
            raise ValueError("All inputs must be of the same length")

        collection = self._collections[name]
        for idx, embedding, document, metadata in zip(
            id_list, embedding_list, document_list, metadata_list
        ):
            collection.append((idx, embedding, document, metadata or {}))

    def query(
        self,
        name: str,
        query_embedding: Sequence[float],
        k: int = 5,
    ) -> List[MockQueryResult]:
        """Return the *k* closest results to the provided embedding."""

        if k <= 0:
            return []
        if name not in self._collections:
            raise KeyError(f"Collection '{name}' does not exist")

        results = self._collections[name]
        if not results:
            return []

        scored_results: List[Tuple[float, Tuple[str, Sequence[float], str, dict]]] = []
        for item in results:
            distance = _euclidean_distance(query_embedding, item[1])
            scored_results.append((distance, item))

        top_results = sorted(scored_results, key=lambda item: item[0])[:k]
        return [
            MockQueryResult(
                id=result[0],
                document=result[2],
                metadata=result[3],
                distance=distance,
            )
            for distance, result in top_results
        ]


def _euclidean_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of the same dimension")
    return sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)) ** 0.5

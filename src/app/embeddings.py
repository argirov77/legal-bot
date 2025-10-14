"""Lightweight embedding helpers used by the in-memory test setup."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import List, Sequence

LOGGER_NAME = "app.embeddings"


class EmbeddingModel:
    """Deterministic embedding model that does not require external downloads."""

    def __init__(self, dimension: int = 8, *_: object, **__: object) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self._dimension = dimension

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector: List[float] = []
            for index in range(self._dimension):
                start = (index * 4) % len(digest)
                chunk = digest[start : start + 4]
                value = int.from_bytes(chunk, "big") / 0xFFFFFFFF
                vector.append(value * 2.0 - 1.0)
            embeddings.append(vector)
        return embeddings

    @property
    def dimension(self) -> int:
        return self._dimension


@lru_cache()
def get_embedding_model() -> EmbeddingModel:
    """Return a cached embedding model instance."""

    return EmbeddingModel()

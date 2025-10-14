"""Mock embedding provider for tests and offline development."""
from __future__ import annotations

import hashlib
import random
from typing import Iterable, List


class MockEmbeddingProvider:
    """Return deterministic embedding vectors for provided texts."""

    def __init__(self, dimension: int = 8) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self.dimension = dimension

    def encode(self, texts: Iterable[str]) -> List[List[float]]:
        """Generate deterministic embeddings derived from each text."""

        vectors: List[List[float]] = []
        for text in texts:
            seed = hashlib.sha256(text.encode("utf-8")).hexdigest()
            rng = random.Random(seed)
            vector = [(rng.random() * 2.0) - 1.0 for _ in range(self.dimension)]
            vectors.append(vector)
        return vectors

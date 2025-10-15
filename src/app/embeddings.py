"""Embedding helpers backed by Sentence Transformers."""
from __future__ import annotations

import hashlib
import logging
import os
import random
from functools import lru_cache
from typing import List, Sequence

LOGGER_NAME = "app.embeddings"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_DIMENSION = 64

LOGGER = logging.getLogger(LOGGER_NAME)


class EmbeddingModel:
    """Wrapper around a SentenceTransformer embedding model with fallback support."""

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        device: str | None = None,
    ) -> None:
        model_path = model_name_or_path or os.getenv("EMBEDDING_MODEL_PATH", DEFAULT_MODEL_NAME)
        embedding_device = device or os.getenv("EMBEDDING_DEVICE")

        self._model = None
        self._dimension = FALLBACK_DIMENSION
        self._embedder = self._fallback_embed_texts

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore import-not-found
        except Exception as error:  # pragma: no cover - depends on optional deps
            LOGGER.warning(
                "sentence-transformers is unavailable; using deterministic fallback embeddings (%s).",
                error,
            )
            return

        try:
            self._model = SentenceTransformer(model_path, device=embedding_device)
        except Exception as error:  # pragma: no cover - unexpected backend errors
            LOGGER.warning(
                "Failed to initialize sentence-transformers model '%s': %s. "
                "Using deterministic fallback embeddings instead.",
                model_path,
                error,
            )
            self._model = None
            return

        self._dimension = int(self._model.get_sentence_embedding_dimension())
        self._embedder = self._embed_texts_with_model

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._embedder(texts)

    def _embed_texts_with_model(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings = self._model.encode(  # type: ignore[union-attr]
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return embeddings.tolist()

    def _fallback_embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._deterministic_embedding(str(text)) for text in texts]

    def _deterministic_embedding(self, text: str) -> List[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big")
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(self._dimension)]

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        """Compatibility method mirroring the sentence-transformers API."""

        return self.embed_texts(texts)

    @property
    def dimension(self) -> int:
        return int(self._dimension)


@lru_cache()
def get_embedding_model() -> EmbeddingModel:
    """Return a cached embedding model instance."""

    return EmbeddingModel()

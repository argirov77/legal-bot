"""Embedding helpers backed by Sentence Transformers."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Sequence

from sentence_transformers import SentenceTransformer

LOGGER_NAME = "app.embeddings"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Wrapper around a SentenceTransformer embedding model."""

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        device: str | None = None,
    ) -> None:
        model_path = model_name_or_path or os.getenv("EMBEDDING_MODEL_PATH", DEFAULT_MODEL_NAME)
        embedding_device = device or os.getenv("EMBEDDING_DEVICE")
        self._model = SentenceTransformer(model_path, device=embedding_device)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return embeddings.tolist()

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        """Compatibility method mirroring the sentence-transformers API."""

        return self.embed_texts(texts)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())


@lru_cache()
def get_embedding_model() -> EmbeddingModel:
    """Return a cached embedding model instance."""

    return EmbeddingModel()

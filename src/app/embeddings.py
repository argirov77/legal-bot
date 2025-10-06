"""Utility helpers for working with embedding models."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
FALLBACK_MODEL_NAME = "sentence-transformers/stsb-xlm-r-multilingual"


def _models_root() -> Path:
    return Path(os.getenv("MODELS_DIR", "models"))


def _find_local_model_path(model_name: str) -> Path | None:
    """Search for a locally downloaded embedding model."""

    candidates: Iterable[Path] = (
        Path(model_name),
        _models_root() / model_name,
        _models_root() / model_name.split("/")[-1],
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_model_name() -> str:
    """Choose the embedding model with fallback logic."""

    env_model = os.getenv("EMBEDDING_MODEL")
    if env_model:
        LOGGER.info("Using embedding model from environment: %s", env_model)
        return env_model

    local_bge = _find_local_model_path(DEFAULT_MODEL_NAME)
    if local_bge:
        LOGGER.info("Found local BGE-M3 model at %s", local_bge)
        return str(local_bge)

    LOGGER.info("Falling back to default multilingual STS-B model")
    return FALLBACK_MODEL_NAME


class EmbeddingModel:
    """Wrapper around a sentence-transformer embedding model."""

    def __init__(self, model_name: str | None = None, batch_size: int = 32, normalize: bool = True) -> None:
        self.model_name = model_name or _resolve_model_name()
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = self._load_model(self.model_name)

    @staticmethod
    def _load_model(model_name: str) -> SentenceTransformer:
        LOGGER.info("Loading embedding model: %s", model_name)
        return SentenceTransformer(model_name)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())


@lru_cache()
def get_embedding_model() -> EmbeddingModel:
    """Return a cached embedding model instance."""

    return EmbeddingModel()


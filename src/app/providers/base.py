"""Base provider interfaces for embeddings and language models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

__all__ = ["EmbeddingProvider", "LLMProvider"]


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode the provided texts into embeddings."""


class LLMProvider(ABC):
    """Abstract interface for large language model providers."""

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text from the given prompt."""

"""Language model adapter interfaces and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.providers.mock import MockLLMProvider


class LLMAdapter(ABC):
    """Abstract base class for all language model adapters."""

    @abstractmethod
    def load(self, model_path: Optional[str] = None) -> None:
        """Load model weights or perform provider specific initialisation."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int) -> str:
        """Generate a response for the supplied prompt."""


class MockLLMAdapter(LLMAdapter):
    """Adapter that proxies calls to :class:`MockLLMProvider`."""

    def __init__(self, provider: Optional[MockLLMProvider] = None) -> None:
        self._provider = provider or MockLLMProvider()
        self._is_loaded = False

    def load(self, model_path: Optional[str] = None) -> None:  # pragma: no cover - trivial
        self._provider.load(model_path)
        self._is_loaded = True

    def generate(self, prompt: str, max_tokens: int) -> str:
        if not self._is_loaded:
            self.load()
        return self._provider.generate(prompt, max_tokens)


class TransformersLLMAdapter(LLMAdapter):
    """Adapter backed by Hugging Face transformers models."""

    def load(self, model_path: Optional[str] = None) -> None:  # pragma: no cover - not implemented
        raise NotImplementedError("TransformersLLMAdapter.load is not implemented yet")

    def generate(self, prompt: str, max_tokens: int) -> str:  # pragma: no cover - not implemented
        raise NotImplementedError("TransformersLLMAdapter.generate is not implemented yet")


__all__ = [
    "LLMAdapter",
    "MockLLMAdapter",
    "TransformersLLMAdapter",
]

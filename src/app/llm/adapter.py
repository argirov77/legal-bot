"""Adapter abstractions for LLM providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.providers import MockLLMProvider


class LLMAdapter(ABC):
    """Common contract for interacting with different LLM providers."""

    @abstractmethod
    def load(self, model_path: str | None = None) -> None:
        """Load the underlying model or initialise the backend."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """Generate a completion for the provided prompt."""


class MockLLMAdapter(LLMAdapter):
    """LLM adapter that wraps the :class:`MockLLMProvider`."""

    def __init__(self, provider: Optional[MockLLMProvider] = None) -> None:
        self._provider: Optional[MockLLMProvider] = provider
        self._is_loaded = provider is not None

    def load(self, model_path: str | None = None) -> None:
        """Initialise the mock provider (model path is ignored)."""

        del model_path  # The mock backend does not rely on a model path.
        if self._provider is None:
            self._provider = MockLLMProvider()
        self._is_loaded = True

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """Delegate generation to the underlying mock provider."""

        if not self._is_loaded or self._provider is None:
            raise RuntimeError("MockLLMAdapter must be loaded before calling generate().")
        return self._provider.generate(prompt, max_tokens=max_tokens)


class TransformersLLMAdapter(LLMAdapter):
    """Placeholder adapter for future HuggingFace Transformers integration."""

    def load(self, model_path: str | None = None) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError("TransformersLLMAdapter.load is not implemented yet.")

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:  # pragma: no cover - placeholder
        raise NotImplementedError("TransformersLLMAdapter.generate is not implemented yet.")

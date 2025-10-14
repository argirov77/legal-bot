"""Mock implementations of provider interfaces for testing."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MockLLMProvider:
    """Deterministic language model provider used for tests."""

    model_path: Optional[str] = None

    def load(self, model_path: Optional[str] = None) -> None:
        """Simulate loading a model from disk."""

        self.model_path = model_path

    def generate(self, prompt: str, max_tokens: int) -> str:
        """Return a predictable completion for the supplied prompt."""

        return f"Mock response to '{prompt}' with max_tokens={max_tokens}."

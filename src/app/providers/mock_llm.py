"""Mock LLM provider that echoes prompts for deterministic testing."""
from __future__ import annotations


class MockLLMProvider:
    """Return a deterministic response for any prompt."""

    def generate(self, prompt: str, max_tokens: int | None = None, **_: object) -> str:
        """Generate a canned response with a predictable prefix."""

        del max_tokens  # Unused in the mock implementation.
        return f"MOCK_ANSWER: {prompt[:100]}"

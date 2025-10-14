"""Unit tests for LLM adapter implementations."""
from __future__ import annotations

from app.llm import MockLLMAdapter


def test_mock_llm_adapter_generate_returns_expected_string() -> None:
    adapter = MockLLMAdapter()
    adapter.load(model_path=None)

    prompt = "Explain the difference between contracts and torts."
    expected = "MOCK_ANSWER: Explain the difference between contracts and torts."

    assert adapter.generate(prompt) == expected

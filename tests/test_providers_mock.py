"""Tests for mock provider implementations."""
from __future__ import annotations

from app.providers import MockEmbeddingProvider, MockLLMProvider


def test_mock_embedding_provider_returns_deterministic_vectors() -> None:
    provider = MockEmbeddingProvider(dimension=5)
    texts = ["hello", "world"]

    vectors = provider.encode(texts)

    assert len(vectors) == len(texts)
    for vector in vectors:
        assert len(vector) == 5
        assert all(isinstance(value, float) for value in vector)

    # Deterministic output for the same inputs.
    assert vectors == provider.encode(texts)


def test_mock_llm_provider_generates_prefixed_response() -> None:
    provider = MockLLMProvider()
    prompt = "x" * 150

    response = provider.generate(prompt, max_tokens=10)

    assert response.startswith("MOCK_ANSWER: ")
    assert response == "MOCK_ANSWER: " + prompt[:100]

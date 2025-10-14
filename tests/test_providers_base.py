from abc import ABC

from app.providers.base import EmbeddingProvider, LLMProvider


def test_embedding_provider_is_abstract() -> None:
    assert issubclass(EmbeddingProvider, ABC)


def test_llm_provider_is_abstract() -> None:
    assert issubclass(LLMProvider, ABC)

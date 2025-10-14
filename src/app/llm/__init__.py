"""LLM module scaffolding for upcoming provider integration."""

from .adapter import LLMAdapter, MockLLMAdapter, TransformersLLMAdapter

__all__ = ["LLMAdapter", "MockLLMAdapter", "TransformersLLMAdapter"]

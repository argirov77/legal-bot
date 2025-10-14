"""Abstractions for interacting with local LLM backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Common contract for BgGPT/Gemma adapters."""

    # TODO: implement loading logic supporting transformers/local_server/mock modes.

    def is_loaded(self) -> bool:
        """Return True when the underlying backend is ready for inference."""
        ...

    def mem_usage(self) -> dict[str, float | int]:
        """Expose memory utilisation details for observability."""
        ...

    def warmup(self) -> None:
        """Run a lightweight prompt to prime caches and check connectivity."""
        ...


# TODO: add factory helpers (e.g., `create_provider(config: dict[str, Any])`).

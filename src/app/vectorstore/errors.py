"""Common exceptions for vector store integrations."""
from __future__ import annotations


class VectorStoreUnavailableError(RuntimeError):
    """Raised when the vector store backend cannot be initialised or queried."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause

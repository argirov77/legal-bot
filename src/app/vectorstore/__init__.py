"""Vector store interfaces and implementations."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .chroma_store import ChromaStore
from .mock_store import MockQueryResult, MockVectorStore

_VECTORSTORE_PATH = Path(__file__).resolve().parent.parent / "vectorstore.py"
_SPEC = importlib.util.spec_from_file_location("app._vectorstore_module", _VECTORSTORE_PATH)
if _SPEC and _SPEC.loader:  # pragma: no branch - defensive
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules.setdefault("app._vectorstore_module", _MODULE)
    _SPEC.loader.exec_module(_MODULE)
else:  # pragma: no cover - loader resolution failure should not happen
    raise ImportError(f"Unable to load vectorstore module from {_VECTORSTORE_PATH}")

ChunkVectorStore = _MODULE.ChunkVectorStore
ChunkSearchResult = _MODULE.ChunkSearchResult
get_vector_store = _MODULE.get_vector_store

__all__ = [
    "ChromaStore",
    "MockVectorStore",
    "MockQueryResult",
    "ChunkVectorStore",
    "ChunkSearchResult",
    "get_vector_store",
]

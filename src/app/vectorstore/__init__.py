"""Vector store interfaces and implementations."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .chroma_store import ChromaStore
from .mock_store import MockQueryResult, MockVectorStore

_MODULE_NAME = "app._vectorstore_impl"
_MODULE_PATH = Path(__file__).resolve().parent.parent / "vectorstore.py"

if _MODULE_NAME in sys.modules:
    _module = sys.modules[_MODULE_NAME]
else:
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load vectorstore module from {_MODULE_PATH}")
    _module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = _module
    spec.loader.exec_module(_module)

ChunkSearchResult = _module.ChunkSearchResult  # type: ignore[attr-defined]
ChunkVectorStore = _module.ChunkVectorStore  # type: ignore[attr-defined]
get_vector_store = _module.get_vector_store  # type: ignore[attr-defined]
reset_vector_store_cache = _module.reset_vector_store_cache  # type: ignore[attr-defined]

__all__ = [
    "ChromaStore",
    "ChunkSearchResult",
    "ChunkVectorStore",
    "MockQueryResult",
    "MockVectorStore",
    "get_vector_store",
    "reset_vector_store_cache",
]

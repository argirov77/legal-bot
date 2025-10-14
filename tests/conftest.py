"""Test configuration providing lightweight stubs for heavy dependencies."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Sequence


def _patch_pydantic_typing() -> None:
    try:
        import pydantic.typing as typing_mod  # type: ignore
    except Exception:  # pragma: no cover - defensive
        return

    evaluate_forwardref = getattr(typing_mod, "evaluate_forwardref", None)
    if not callable(evaluate_forwardref):  # pragma: no cover - defensive
        return

    signature = inspect.signature(evaluate_forwardref)
    if "recursive_guard" not in signature.parameters:
        original = evaluate_forwardref

        def _patched_evaluate_forwardref(type_, globalns, localns=None, recursive_guard=None):
            try:
                return original(type_, globalns, localns)
            except TypeError:
                target = getattr(type_, "_evaluate", None)
                if callable(target):
                    guard = recursive_guard if recursive_guard is not None else set()
                    return target(globalns, localns, recursive_guard=guard)
                raise

        typing_mod.evaluate_forwardref = _patched_evaluate_forwardref  # type: ignore[attr-defined]

    def _patch_forward_ref(cls: object) -> None:
        method = getattr(cls, "_evaluate", None)
        if callable(method):
            method_signature = inspect.signature(method)
            if "recursive_guard" not in method_signature.parameters:
                def _patched_forward_ref_evaluate(self, globalns, localns, recursive_guard=None, **kwargs):
                    return method(self, globalns, localns)

                setattr(cls, "_evaluate", _patched_forward_ref_evaluate)

    forward_ref = getattr(typing_mod, "ForwardRef", None)
    _patch_forward_ref(forward_ref)

    try:
        import typing as typing_stdlib
    except Exception:  # pragma: no cover - defensive
        typing_stdlib = None
    if typing_stdlib is not None:
        _patch_forward_ref(getattr(typing_stdlib, "ForwardRef", None))


_patch_pydantic_typing()


class _EmbeddingResult(list):
    """Simple container that mimics numpy arrays returned by real models."""

    def tolist(self) -> List[List[float]]:  # pragma: no cover - simple passthrough
        return list(self)


class _SentenceTransformerStub:
    """Deterministic and dependency-free sentence transformer stub."""

    def __init__(self, model_name: str, *args, **kwargs) -> None:  # pragma: no cover - trivial
        self.model_name = model_name

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **_: object,
    ) -> _EmbeddingResult:
        del batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings
        embeddings = _EmbeddingResult([[float(len(text))] for text in texts])
        return embeddings

    def get_sentence_embedding_dimension(self) -> int:  # pragma: no cover - trivial
        return 1


class _InMemoryCollection:
    """Minimal collection implementation for Chroma-like interactions."""

    def __init__(self, name: str, metadata: Dict[str, object] | None = None) -> None:
        self.name = name
        self.metadata = metadata or {}
        self._records: List[Dict[str, object]] = []

    # API used by the application -------------------------------------------------
    def add(
        self,
        *,
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]] | Sequence[List[float]],
        metadatas: Sequence[Dict[str, object]],
        ids: Sequence[str],
    ) -> None:
        self._records.extend(
            {
                "id": chunk_id,
                "document": document,
                "metadata": metadata,
                "embedding": list(embedding),
            }
            for chunk_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings)
        )

    def upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Dict[str, object]],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        existing = {record["id"]: record for record in self._records}
        for chunk_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            existing[chunk_id] = {
                "id": chunk_id,
                "document": document,
                "metadata": metadata,
                "embedding": list(embedding),
            }
        self._records = list(existing.values())

    # Helper methods --------------------------------------------------------------
    def count(self) -> int:  # pragma: no cover - trivial
        return len(self._records)

    def query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]] | None = None,
        query_texts: Sequence[str] | None = None,
        n_results: int = 5,
        include: Sequence[str] | None = None,
        **_: object,
    ) -> Dict[str, List[List[object]]]:
        include = list(include or [])
        ranked = list(self._records)

        def _score(record: Dict[str, object]) -> float:
            metadata = record.get("metadata") or {}
            length = metadata.get("content_length")
            if not isinstance(length, (int, float)):
                length = len(record.get("document", ""))
            return -float(length)

        ranked.sort(key=_score)
        ranked = ranked[: max(0, int(n_results))]

        result: Dict[str, List[List[object]]] = {}
        if "ids" in include:
            result["ids"] = [[record["id"] for record in ranked]]
        if "documents" in include:
            result["documents"] = [[record.get("document", "") for record in ranked]]
        if "metadatas" in include:
            result["metadatas"] = [[record.get("metadata", {}) for record in ranked]]
        if "distances" in include:
            result["distances"] = [[float(index) for index, _ in enumerate(ranked)]]
        return result


class _ChromaBackend:
    """Backend shared between multiple client instances via persist directory."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.collections: Dict[str, _InMemoryCollection] = {}

    def ensure_persisted(self) -> None:
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        marker = self.path / ".persisted"
        marker.write_text("ok", encoding="utf-8")


_BACKENDS: Dict[str, _ChromaBackend] = {}


def _get_backend(identifier: str) -> _ChromaBackend:
    backend = _BACKENDS.get(identifier)
    if backend is None:
        path = None if identifier == ":memory:" else Path(identifier)
        backend = _BACKENDS[identifier] = _ChromaBackend(path)
    return backend


class _ClientBase:
    def __init__(self, identifier: str) -> None:
        self._backend = _get_backend(identifier)

    def get_or_create_collection(self, *, name: str, metadata: Dict[str, object] | None = None):
        collection = self._backend.collections.get(name)
        if collection is None:
            collection = _InMemoryCollection(name, metadata=metadata)
            self._backend.collections[name] = collection
        return collection

    def get_collection(self, name: str):
        return self._backend.collections[name]

    def create_collection(self, name: str, metadata: Dict[str, object] | None = None):
        collection = _InMemoryCollection(name, metadata=metadata)
        self._backend.collections[name] = collection
        return collection

    def list_collections(self) -> Iterable[_InMemoryCollection]:
        return list(self._backend.collections.values())

    def persist(self) -> None:
        self._backend.ensure_persisted()


class _Settings:
    def __init__(self, *args: object, persist_directory: str | None = None, **kwargs: object) -> None:  # pragma: no cover - trivial
        del args, kwargs
        self.persist_directory = persist_directory


def _install_sentence_transformers_stub() -> None:
    if "sentence_transformers" in sys.modules:
        return
    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = _SentenceTransformerStub
    sys.modules["sentence_transformers"] = module


def _install_chromadb_stub() -> None:
    if "chromadb" in sys.modules:
        return

    chromadb_module = ModuleType("chromadb")
    api_module = ModuleType("chromadb.api")
    models_module = ModuleType("chromadb.api.models")
    collection_module = ModuleType("chromadb.api.models.Collection")
    config_module = ModuleType("chromadb.config")

    class _PersistentClient(_ClientBase):
        def __init__(self, path: str | Path | None = None, settings: _Settings | None = None, **_: object) -> None:
            identifier = ":memory:"
            if path:
                identifier = str(Path(path))
            elif settings and settings.persist_directory:
                identifier = str(Path(settings.persist_directory))
            super().__init__(identifier)
            self.path = identifier

    def _client_factory(settings: _Settings | None = None, **_: object) -> _PersistentClient:
        identifier = ":memory:"
        if settings and settings.persist_directory:
            identifier = str(Path(settings.persist_directory))
        return _PersistentClient(path=identifier)

    chromadb_module.PersistentClient = _PersistentClient
    chromadb_module.Client = _client_factory
    chromadb_module.__all__ = ["PersistentClient", "Client"]

    api_module.ClientAPI = _PersistentClient
    models_module.Collection = _InMemoryCollection
    collection_module.Collection = _InMemoryCollection
    config_module.Settings = _Settings

    sys.modules.update(
        {
            "chromadb": chromadb_module,
            "chromadb.api": api_module,
            "chromadb.api.models": models_module,
            "chromadb.api.models.Collection": collection_module,
            "chromadb.config": config_module,
        }
    )


_install_sentence_transformers_stub()
_install_chromadb_stub()

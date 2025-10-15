"""Chroma vector store adapter."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import chromadb  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade when unavailable
    chromadb = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection
else:  # pragma: no cover - runtime fallback types
    ClientAPI = Any  # type: ignore
    Collection = Any  # type: ignore

try:  # pragma: no cover - depends on chromadb internals
    from chromadb.errors import NotEnoughElementsException
except ImportError:  # pragma: no cover - fallback for older versions
    try:
        from chromadb.api.types import NotEnoughElementsException  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover - ultimate fallback
        class NotEnoughElementsException(Exception):
            """Fallback exception when Chroma does not expose the expected error."""

from .errors import VectorStoreUnavailableError


class ChromaStore:
    """Adapter around a Chroma vector database."""

    def __init__(
        self,
        persist_dir: str | Path,
        *,
        client: Optional[ClientAPI] = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        if chromadb is None and client is None:
            raise VectorStoreUnavailableError(
                "chromadb is not installed; cannot initialise persistent vector store"
            )

        try:
            if client is not None:
                self._client = client
            else:
                self._client = chromadb.PersistentClient(path=str(self.persist_dir))  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - depends on chromadb runtime
            raise VectorStoreUnavailableError(
                "Failed to initialise Chroma persistent client",
                cause=exc,
            ) from exc
        self._collections: Dict[str, Collection] = {}

    def create_collection(self, name: str, *, metadata: Optional[Dict[str, Any]] = None) -> Collection:
        """Return an existing collection or create a new one."""

        collection = self._collections.get(name)
        if collection is None:
            collection = self._client.get_or_create_collection(name=name, metadata=metadata)
            self._collections[name] = collection
        elif metadata and hasattr(collection, "modify"):
            # Update metadata if new metadata is provided after creation.
            collection.modify(metadata=metadata)
        return collection

    def add(
        self,
        name: str,
        *,
        ids: Iterable[str],
        embeddings: Iterable[Sequence[float]],
        documents: Iterable[str],
        metadatas: Iterable[Dict[str, Any] | None] | None = None,
    ) -> None:
        """Add embeddings and corresponding documents to a collection."""

        collection = self.create_collection(name)

        id_list = list(ids)
        embedding_list = [list(map(float, embedding)) for embedding in embeddings]
        document_list = list(documents)
        metadata_source = list(metadatas) if metadatas is not None else [None] * len(id_list)

        if not (
            len(id_list)
            == len(embedding_list)
            == len(document_list)
            == len(metadata_source)
        ):
            raise ValueError("All inputs must be of the same length")

        metadata_list: List[Dict[str, Any]] = [metadata or {} for metadata in metadata_source]
        collection.upsert(
            ids=id_list,
            embeddings=embedding_list,
            documents=document_list,
            metadatas=metadata_list,
        )

    def query(
        self,
        name: str,
        query_embedding: Sequence[float],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Query the underlying Chroma collection for the nearest neighbours."""

        if k <= 0:
            return []

        collection = self.create_collection(name)
        try:
            result = collection.query(
                query_embeddings=[list(map(float, query_embedding))],
                n_results=k,
            )
        except NotEnoughElementsException:
            return []

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        neighbours: List[Dict[str, Any]] = []
        for idx, doc, metadata, distance in zip(ids, documents, metadatas, distances):
            neighbours.append(
                {
                    "id": idx,
                    "document": doc,
                    "metadata": metadata or {},
                    "distance": float(distance) if distance is not None else 0.0,
                }
            )
        return neighbours

    def get_collection(self, name: str) -> Collection:
        """Expose the underlying Chroma collection for advanced operations."""

        return self.create_collection(name)


__all__ = ["ChromaStore"]

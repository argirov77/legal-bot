"""Chroma vector store adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection


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
        self._client: ClientAPI = client or chromadb.PersistentClient(path=str(self.persist_dir))
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
        result = collection.query(query_embeddings=[list(map(float, query_embedding))], n_results=k)

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

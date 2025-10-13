"""Vector store implementation backed by ChromaDB."""
from __future__ import annotations

import datetime as dt
import logging
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from app.embeddings import EmbeddingModel, get_embedding_model
from app.ingest.models import DocumentChunk

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkSearchResult:
    """Structured response returned from similarity search queries."""

    id: str
    content: str
    distance: float
    metadata: Dict[str, Any]


class ChunkVectorStore:
    """Wrapper around Chroma persistent collections for chunk storage."""

    def __init__(
        self,
        *,
        persist_dir: str | Path | None = None,
        collection_name: str = "legal_chunks",
        distance_metric: str = "cosine",
        embedding_model: Optional[EmbeddingModel] = None,
    ) -> None:
        self.persist_dir = Path(
            persist_dir or os.getenv("CHROMA_PERSIST_DIR", Path("chroma_db"))
        ).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self.embedding_model = embedding_model or get_embedding_model()
        self._client = self._create_client(self.persist_dir)
        self._collection = self._get_or_create_collection()

    @staticmethod
    def _create_client(persist_dir: Path) -> ClientAPI:
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            anonymized_telemetry=False,
        )
        return chromadb.PersistentClient(path=str(persist_dir), settings=settings)

    def _get_or_create_collection(self) -> Collection:
        LOGGER.info("Opening Chroma collection '%s' in %s", self.collection_name, self.persist_dir)
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )

    @property
    def collection(self) -> Collection:
        return self._collection

    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        meta = chunk.metadata
        hash_source = f"{meta.file_id}:{meta.chunk_index}:{meta.char_start}:{meta.char_end}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, hash_source))

    def upsert_chunks(self, chunks: Sequence[DocumentChunk]) -> List[str]:
        if not chunks:
            return []

        documents = [chunk.content for chunk in chunks]
        metadatas = [self._prepare_metadata(chunk) for chunk in chunks]
        ids = [self._generate_chunk_id(chunk) for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(documents)

        LOGGER.info("Upserting %s chunks into vector store", len(chunks))
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        self._client.persist()
        return ids

    def _prepare_metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
        metadata = asdict(chunk.metadata)
        metadata["content_length"] = len(chunk.content)
        return metadata

    def query_by_text(self, text: str, k: int = 5) -> List[ChunkSearchResult]:
        if not text.strip():
            return []
        if k <= 0:
            return []

        available_results = self.collection.count()
        if available_results == 0:
            return []

        query_embeddings = self.embedding_model.embed_texts([text])
        if not query_embeddings:
            return []
        query_embedding = query_embeddings[0]
        n_results = min(k, available_results)
        if n_results <= 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "documents", "distances", "ids"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        search_results: List[ChunkSearchResult] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            search_results.append(
                ChunkSearchResult(id=chunk_id, content=document, distance=float(distance), metadata=metadata or {})
            )
        return search_results

    def create_snapshot(self, snapshot_dir: Path | None = None) -> Path:
        if snapshot_dir is None:
            snapshot_dir = self.persist_dir.parent / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        timestamp = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        snapshot_path = snapshot_dir / f"{self.collection_name}-{timestamp}"
        LOGGER.info("Creating vector store snapshot at %s", snapshot_path)
        if not self.persist_dir.exists():
            raise FileNotFoundError(f"Persist directory {self.persist_dir} does not exist")
        self._client.persist()
        shutil.copytree(self.persist_dir, snapshot_path)
        return snapshot_path
@lru_cache()
def get_vector_store() -> ChunkVectorStore:
    """Return a lazily initialised vector store instance."""

    return ChunkVectorStore()


def reset_vector_store_cache() -> None:
    """Clear the cached vector store (primarily for testing)."""

    get_vector_store.cache_clear()  # type: ignore[attr-defined]


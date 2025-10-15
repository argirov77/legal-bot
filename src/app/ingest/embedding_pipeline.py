"""Pipeline for computing embeddings and storing them in the vector DB."""
from __future__ import annotations

from typing import List, Optional, Sequence

from app.ingest.models import DocumentChunk
from app.vectorstore import ChunkVectorStore, get_vector_store


class ChunkEmbeddingPipeline:
    """Takes prepared document chunks and persists them into Chroma."""

    def __init__(self, vector_store: Optional[ChunkVectorStore] = None) -> None:
        self.vector_store = vector_store or get_vector_store()

    def run(self, chunks: Sequence[DocumentChunk]) -> List[str]:
        """Compute embeddings for chunks and upsert them into the vector store."""

        return self.vector_store.upsert_chunks(list(chunks))


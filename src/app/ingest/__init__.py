"""Document ingestion pipeline for Legal Bot."""
from .embedding_pipeline import ChunkEmbeddingPipeline
from .pipeline import IngestPipeline
from .models import DocumentChunk, ChunkMetadata, PageContent
from .format_detection import DocumentFormat, DocumentFormatDetector

__all__ = [
    "IngestPipeline",
    "ChunkEmbeddingPipeline",
    "DocumentChunk",
    "ChunkMetadata",
    "PageContent",
    "DocumentFormat",
    "DocumentFormatDetector",
]

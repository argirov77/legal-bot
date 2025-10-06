"""Document ingestion pipeline for Legal Bot."""
from .pipeline import IngestPipeline
from .models import DocumentChunk, ChunkMetadata, PageContent
from .format_detection import DocumentFormat, DocumentFormatDetector

__all__ = [
    "IngestPipeline",
    "DocumentChunk",
    "ChunkMetadata",
    "PageContent",
    "DocumentFormat",
    "DocumentFormatDetector",
]

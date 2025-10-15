"""Data models used by the ingestion pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class PageContent:
    """Represents text extracted from a page in the source document."""

    page_number: int
    text: str
    char_offset: int


@dataclass(slots=True)
class ChunkMetadata:
    """Metadata attached to an individual chunk."""

    file_id: str
    file_name: str
    page: int
    chunk_index: int
    char_start: int
    char_end: int
    language: Optional[str]
    session_id: Optional[str] = None


@dataclass(slots=True)
class DocumentChunk:
    """Container that pairs chunk text with associated metadata."""

    content: str
    metadata: ChunkMetadata

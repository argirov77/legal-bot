"""Chunking utilities for breaking text into embedding-friendly units."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

from .models import ChunkMetadata, DocumentChunk, PageContent

_FALLBACK_SENTENCE_RE = re.compile(r"(.+?(?:[.!?](?=\s)|$))", re.DOTALL)
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkingConfig:
    chunk_chars: int
    overlap_chars: int


class SemanticTextChunker:
    """Split document text into chunks respecting semantic boundaries."""

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def chunk_pages(
        self,
        pages: Iterable[PageContent],
        file_id: str,
        file_name: str,
        language: str | None,
    ) -> Iterator[DocumentChunk]:
        chunk_index = 0
        document_offset = 0
        for page in pages:
            for chunk_text, start_offset, end_offset in self._chunk_text(page.text):
                metadata = ChunkMetadata(
                    file_id=file_id,
                    file_name=file_name,
                    page=page.page_number,
                    chunk_index=chunk_index,
                    char_start=document_offset + start_offset,
                    char_end=document_offset + end_offset,
                    language=language,
                )
                LOGGER.debug(
                    "Chunk %s page %s offsets %s-%s",  # noqa: G004 - f-string not required
                    chunk_index,
                    page.page_number,
                    metadata.char_start,
                    metadata.char_end,
                )
                yield DocumentChunk(content=chunk_text, metadata=metadata)
                chunk_index += 1
            document_offset += len(page.text)

    def _chunk_text(self, text: str) -> Iterator[Tuple[str, int, int]]:
        if not text:
            return
        chunk_chars = max(self.config.chunk_chars, 1)
        overlap_chars = max(self.config.overlap_chars, 0)
        text_length = len(text)
        start = 0
        while start < text_length:
            tentative_end = min(start + chunk_chars, text_length)
            chunk_end = self._find_semantic_break(text, start, tentative_end)
            if chunk_end <= start:
                chunk_end = min(start + chunk_chars, text_length)
            if chunk_end <= start:
                chunk_end = text_length
            raw_chunk = text[start:chunk_end]
            stripped_chunk = raw_chunk.strip()
            if not stripped_chunk:
                start = chunk_end if chunk_end > start else start + 1
                continue
            leading_ws = len(raw_chunk) - len(raw_chunk.lstrip())
            trailing_ws = len(raw_chunk) - len(raw_chunk.rstrip())
            final_start = start + leading_ws
            final_end = chunk_end - trailing_ws
            yield text[final_start:final_end], final_start, final_end
            if final_end >= text_length:
                break
            next_start = final_end - overlap_chars
            if next_start <= final_start and overlap_chars > 0:
                next_start = final_end
            start = max(0, next_start)

    def _find_semantic_break(self, text: str, start: int, tentative_end: int) -> int:
        if tentative_end >= len(text):
            return len(text)
        segment = text[start:tentative_end]
        paragraph_break = segment.rfind("\n\n")
        if paragraph_break != -1 and paragraph_break >= self.config.chunk_chars // 3:
            return start + paragraph_break + 2
        sentence_break = self._find_sentence_break(segment)
        if sentence_break is not None and sentence_break >= self.config.chunk_chars // 4:
            return start + sentence_break
        word_break = segment.rfind(" ")
        if word_break != -1 and word_break >= self.config.chunk_chars // 4:
            return start + word_break
        return tentative_end

    @staticmethod
    def _find_sentence_break(segment: str) -> int | None:
        matches = list(_FALLBACK_SENTENCE_RE.finditer(segment))
        if not matches:
            return None
        return matches[-1].end()

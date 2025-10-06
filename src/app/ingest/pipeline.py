"""High level ingestion pipeline entry point."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .chunking import ChunkingConfig, SemanticTextChunker
from .extractors import DocxExtractor, PDFExtractor, PDFExtractionResult, TextExtractor
from .format_detection import DocumentFormat, DocumentFormatDetector
from .language import LanguageDetector
from .models import DocumentChunk, PageContent
from .normalization import normalize_text

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestPipelineConfig:
    chunk_chars: int = 1000
    overlap_chars: int = 100
    ocr_language: str = "bul"
    pdf_min_text_ratio: float = 0.05


class IngestPipeline:
    """Pipeline orchestrating document extraction, normalisation and chunking."""

    def __init__(self, config: Optional[IngestPipelineConfig] = None) -> None:
        self.config = config or IngestPipelineConfig()
        self.pdf_extractor = PDFExtractor(
            ocr_language=self.config.ocr_language,
            min_text_ratio=self.config.pdf_min_text_ratio,
        )
        self.docx_extractor = DocxExtractor()
        self.text_extractor = TextExtractor()
        self.chunker = SemanticTextChunker(
            ChunkingConfig(chunk_chars=self.config.chunk_chars, overlap_chars=self.config.overlap_chars)
        )
        self.language_detector = LanguageDetector()

    def ingest(
        self,
        file_bytes: bytes,
        file_name: str,
        file_id: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Process an uploaded file and return embedding-ready chunks."""

        document_format = DocumentFormatDetector.detect(file_name, mime_type)
        file_id = file_id or str(uuid.uuid4())
        LOGGER.info("Processing file %s (%s) with id %s", file_name, document_format, file_id)

        pages = self._extract_pages(file_bytes, document_format)
        normalized_pages = [
            PageContent(page_number=page.page_number, text=normalize_text(page.text), char_offset=page.char_offset)
            for page in pages
        ]
        language = self.language_detector.detect("\n".join(page.text for page in normalized_pages))
        LOGGER.debug("Language detected for %s: %s", file_name, language)

        chunks = list(self.chunker.chunk_pages(normalized_pages, file_id, file_name, language))
        LOGGER.info("Generated %s chunks for file %s", len(chunks), file_name)
        return chunks

    def _extract_pages(self, file_bytes: bytes, document_format: DocumentFormat) -> Iterable[PageContent]:
        if document_format is DocumentFormat.PDF:
            result: PDFExtractionResult = self.pdf_extractor.extract(file_bytes)
            if result.ocr_performed:
                LOGGER.info("OCR performed on PDF document")
            return result.pages
        if document_format is DocumentFormat.DOCX:
            return list(self.docx_extractor.extract(file_bytes))
        if document_format is DocumentFormat.TXT:
            return list(self.text_extractor.extract(file_bytes))
        raise ValueError(f"Unsupported document format: {document_format}")

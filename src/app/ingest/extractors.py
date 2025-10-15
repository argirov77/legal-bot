"""Extractors for supported document types."""
from __future__ import annotations

import io
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterable, List

from typing import TYPE_CHECKING

try:  # pragma: no cover - optional heavy dependency
    from PyPDF2 import PdfReader  # type: ignore
except Exception:  # pragma: no cover - gracefully handle missing dependency
    PdfReader = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from docx import Document as DocxDocument

from .models import PageContent

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PDFExtractionResult:
    pages: List[PageContent]
    ocr_performed: bool


class PDFExtractor:
    """Extract text from PDF documents with optional OCR fallback."""

    def __init__(self, ocr_language: str = "bul", min_text_ratio: float = 0.05) -> None:
        self.ocr_language = ocr_language
        self.min_text_ratio = min_text_ratio

    def extract(self, data: bytes) -> PDFExtractionResult:
        """Extract text, running OCR when native text is insufficient."""

        pages = self._extract_text(data)
        total_chars = sum(len(page.text.strip()) for page in pages)
        if pages and total_chars / max(len(pages), 1) >= self.min_text_ratio * 1000:
            return PDFExtractionResult(pages=pages, ocr_performed=False)

        LOGGER.info("PDF text content too small, attempting OCR fallback")
        try:
            ocr_data = self._perform_ocr(data)
        except RuntimeError as error:
            LOGGER.warning("OCR failed (%s); falling back to original extraction", error)
            return PDFExtractionResult(pages=pages, ocr_performed=False)

        return PDFExtractionResult(pages=self._extract_text(ocr_data), ocr_performed=True)

    def _extract_text(self, data: bytes) -> List[PageContent]:
        if PdfReader is None:
            LOGGER.warning(
                "PyPDF2 is not installed; falling back to naive UTF-8 decoding for PDF extraction"
            )
            fallback_text = data.decode("utf-8", errors="ignore")
            return [PageContent(page_number=1, text=fallback_text, char_offset=0)]

        reader = PdfReader(io.BytesIO(data))
        pages: List[PageContent] = []
        char_offset = 0
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as error:  # pragma: no cover - depends on pdfminer backend
                LOGGER.warning("Failed to extract text from PDF page %s: %s", index, error)
                text = ""
            pages.append(PageContent(page_number=index, text=text, char_offset=char_offset))
            char_offset += len(text)
        return pages

    def _perform_ocr(self, data: bytes) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as src, tempfile.NamedTemporaryFile(
            suffix=".pdf"
        ) as dst:
            src.write(data)
            src.flush()
            cmd = [
                "ocrmypdf",
                "--force-ocr",
                "--output-type",
                "pdf",
                "-l",
                self.ocr_language,
                src.name,
                dst.name,
            ]
            LOGGER.debug("Running OCR command: %s", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except FileNotFoundError as exc:  # pragma: no cover - depends on environment
                raise RuntimeError("ocrmypdf is not installed") from exc
            except subprocess.CalledProcessError as exc:  # pragma: no cover - depends on data
                raise RuntimeError(f"ocrmypdf failed: {exc.stderr.decode(errors='ignore')}") from exc

            dst.seek(0)
            return dst.read()


class DocxExtractor:
    """Extract text from Microsoft Word documents."""

    def extract(self, data: bytes) -> Iterable[PageContent]:
        document = self._load_document(data)
        if document is not None:
            text_parts = []
            for paragraph in document.paragraphs:
                if paragraph.text:
                    text_parts.append(paragraph.text)
            text = "\n\n".join(text_parts)
            return [PageContent(page_number=1, text=text, char_offset=0)]

        return self._fallback_extract(data)

    def _load_document(self, data: bytes):  # pragma: no cover - thin wrapper
        try:
            from docx import Document as DocxDocument  # type: ignore
        except Exception as error:
            LOGGER.warning(
                "python-docx is unavailable; falling back to lightweight DOCX extraction (%s)",
                error,
            )
            return None

        try:
            return DocxDocument(io.BytesIO(data))
        except Exception as error:
            LOGGER.warning("python-docx failed to parse DOCX content: %s", error)
            return None

    def _fallback_extract(self, data: bytes) -> Iterable[PageContent]:
        try:
            import xml.etree.ElementTree as ET
            import zipfile

            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                xml_bytes = archive.read("word/document.xml")
        except Exception as error:
            LOGGER.warning(
                "DOCX fallback extraction failed to read XML payload: %s", error
            )
            text = data.decode("utf-8", errors="ignore")
            return [PageContent(page_number=1, text=text, char_offset=0)]

        try:
            root = ET.fromstring(xml_bytes)
            namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            text_parts = [node.text for node in root.iter(f"{namespace}t") if node.text]
            text = "\n\n".join(text_parts)
        except Exception as error:
            LOGGER.warning("DOCX fallback XML parsing failed: %s", error)
            text = xml_bytes.decode("utf-8", errors="ignore")

        return [PageContent(page_number=1, text=text, char_offset=0)]


class TextExtractor:
    """Extract text from plaintext documents."""

    def extract(self, data: bytes, encoding: str = "utf-8") -> Iterable[PageContent]:
        text = data.decode(encoding)
        return [PageContent(page_number=1, text=text, char_offset=0)]

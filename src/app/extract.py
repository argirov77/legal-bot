"""Utilities for extracting text from supported document types."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency during tests
    from pdfminer.high_level import extract_text as pdf_extract_text  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade when unavailable
    pdf_extract_text = None  # type: ignore

try:  # pragma: no cover - optional dependency during tests
    from docx import Document  # type: ignore
except Exception:  # pragma: no cover - we provide a fallback below
    Document = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def extract_text(path: Path, use_ocr: bool = False) -> str:
    """Extract textual content from the provided document.

    The function supports plain text, Microsoft Word (.docx), and PDF documents.
    It will attempt to gracefully handle extraction failures by logging the error
    and returning an empty string instead of propagating exceptions.
    """

    try:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return _read_text_file(path)
        if suffix == ".docx":
            return _extract_docx(path)
        if suffix == ".pdf":
            return _extract_pdf(path, use_ocr=use_ocr)
        LOGGER.warning("Unsupported file type for extraction: %s", path)
        return ""
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.exception("Unexpected error while extracting text from %s", path)
        return ""


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-16")
        except UnicodeDecodeError:
            try:
                return path.read_text(encoding="latin-1")
            except Exception:
                LOGGER.exception("Failed to read text file %s", path)
                return ""
    except Exception:
        LOGGER.exception("Failed to read text file %s", path)
        return ""


def _extract_docx(path: Path) -> str:
    if Document is not None:
        try:
            document = Document(path)
            paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
            return "\n\n".join(paragraphs)
        except Exception:
            LOGGER.warning("python-docx failed to process %s; attempting fallback", path)

    return _fallback_docx(path)


def _fallback_docx(path: Path) -> str:
    try:
        import xml.etree.ElementTree as ET
        import zipfile

        with zipfile.ZipFile(path) as archive:
            xml_bytes = archive.read("word/document.xml")
        root = ET.fromstring(xml_bytes)
        namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        texts = []
        for node in root.iter(f"{namespace}t"):
            if node.text:
                texts.append(node.text)
        return "".join(texts)
    except Exception:
        LOGGER.exception("Fallback DOCX extraction failed for %s", path)
        return ""


def _extract_pdf(path: Path, use_ocr: bool = False) -> str:
    text = _pdfminer_extract(path)
    if text.strip():
        return text

    if not use_ocr:
        LOGGER.info("PDF %s appears to require OCR, but OCR is disabled", path)
        return text

    ocr_text = _perform_pdf_ocr(path)
    if ocr_text is None:
        LOGGER.warning("OCR could not be performed on %s", path)
        return ""

    return ocr_text


def _pdfminer_extract(path: Path) -> str:
    if pdf_extract_text is None:
        LOGGER.warning("pdfminer.six is not installed; skipping PDF extraction for %s", path)
        return ""
    try:
        return pdf_extract_text(str(path)) or ""
    except Exception:
        LOGGER.exception("pdfminer failed to extract text from %s", path)
        return ""


def _perform_pdf_ocr(path: Path) -> Optional[str]:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_pdf = Path(tmpdir) / "ocr-output.pdf"
            sidecar = Path(tmpdir) / "ocr-output.txt"
            cmd = [
                "ocrmypdf",
                "--force-ocr",
                "--sidecar",
                str(sidecar),
                str(path),
                str(output_pdf),
            ]
            LOGGER.debug("Running OCR command: %s", " ".join(cmd))
            result = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                LOGGER.warning(
                    "ocrmypdf exited with code %s for %s: %s",
                    result.returncode,
                    path,
                    result.stderr.decode(errors="ignore"),
                )
                return None
            if sidecar.exists():
                try:
                    return sidecar.read_text(encoding="utf-8")
                except Exception:
                    LOGGER.exception("Failed to read OCR sidecar text for %s", path)
                    return None
            return _pdfminer_extract(output_pdf)
    except FileNotFoundError:
        LOGGER.warning("ocrmypdf is not installed; OCR cannot be performed for %s", path)
        return None
    except Exception:
        LOGGER.exception("Unexpected error while running ocrmypdf for %s", path)
        return None

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from docx import Document
from pdfminer.high_level import extract_text as pdf_extract_text

logger = logging.getLogger(__name__)


def _read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            logger.exception("Failed to read text file %s", path)
    except Exception:
        logger.exception("Unexpected error reading text file %s", path)
    return ""


def _read_docx(path: Path) -> str:
    try:
        document = Document(path)
        parts = []
        for paragraph in document.paragraphs:
            parts.append(paragraph.text)
        for table in document.tables:
            for row in table.rows:
                parts.extend(cell.text for cell in row.cells)
        return "\n".join(part for part in parts if part)
    except Exception:
        logger.exception("Failed to read docx file %s", path)
        try:
            import zipfile
            from xml.etree import ElementTree

            with zipfile.ZipFile(path) as docx_zip:
                xml_content = docx_zip.read("word/document.xml")
            root = ElementTree.fromstring(xml_content)
            texts = [
                node.text
                for node in root.iter()
                if node.text and node.tag.endswith("}t")
            ]
            return "\n".join(texts)
        except Exception:
            logger.exception("Fallback docx parsing failed for %s", path)
            return ""


def _read_pdf(path: Path, use_ocr: bool) -> str:
    try:
        text = pdf_extract_text(str(path)) or ""
    except Exception:
        logger.exception("Failed to extract text from pdf %s", path)
        text = ""

    if text.strip() or not use_ocr:
        return text

    return _run_ocrmypdf(path)


def _run_ocrmypdf(path: Path) -> str:
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            sidecar_path = tmpdir_path / "ocr.txt"
            output_pdf_path = tmpdir_path / "ocr.pdf"
            cmd = [
                "ocrmypdf",
                "--sidecar",
                str(sidecar_path),
                "--force-ocr",
                str(path),
                str(output_pdf_path),
            ]
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(
                    "ocrmypdf failed for %s with code %s: %s",
                    path,
                    result.returncode,
                    result.stderr,
                )
                return ""
            if sidecar_path.exists():
                try:
                    return sidecar_path.read_text(encoding="utf-8")
                except Exception:
                    logger.exception("Failed to read OCR sidecar %s", sidecar_path)
            return ""
    except FileNotFoundError:
        logger.warning("ocrmypdf not installed; OCR required for %s", path)
        return ""
    except Exception:
        logger.exception("Unexpected error running ocrmypdf for %s", path)
        return ""


def extract_text(path: Path, use_ocr: bool = False) -> str:
    """Extract text content from supported document types.

    Args:
        path: Path to the input document.
        use_ocr: Whether to attempt OCR for PDFs when text extraction fails.

    Returns:
        Extracted text content. Returns an empty string when extraction fails.
    """

    if not isinstance(path, Path):
        path = Path(path)

    if not path.exists():
        logger.warning("File not found: %s", path)
        return ""

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _read_txt(path)
    if suffix == ".docx":
        return _read_docx(path)
    if suffix == ".pdf":
        return _read_pdf(path, use_ocr)

    logger.warning("Unsupported file extension '%s' for %s", suffix, path)
    return ""

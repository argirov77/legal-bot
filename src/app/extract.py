from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def extract_text(path: Path, use_ocr: bool = False) -> str:
    """Extract text content from the supported document types.

    Parameters
    ----------
    path: Path
        The path to the document to extract text from.
    use_ocr: bool
        When ``True`` the function attempts to run ``ocrmypdf`` on PDF files
        that do not contain extractable text.
    """

    try:
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            return ""

        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return _extract_txt(file_path)
        if suffix == ".docx":
            return _extract_docx(file_path)
        if suffix == ".pdf":
            text = _extract_pdf(file_path)
            if text.strip():
                return text
            if use_ocr:
                return _extract_pdf_via_ocr(file_path)
            return text
        return ""
    except Exception:
        return ""


def _extract_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    except Exception:
        return ""


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document
    except Exception:
        return ""

    try:
        document = Document(path)
    except Exception:
        return ""

    try:
        paragraphs = [p.text for p in getattr(document, "paragraphs", []) if p.text]
        return "\n".join(paragraphs)
    except Exception:
        return ""


def _extract_pdf(path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text as pdf_extract_text
    except Exception:
        return ""

    try:
        return pdf_extract_text(str(path)) or ""
    except Exception:
        return ""


def _extract_pdf_via_ocr(path: Path) -> str:
    output_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            output_path = Path(tmp.name)

        try:
            subprocess.run(
                ["ocrmypdf", "--skip-text", str(path), str(output_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

        return _extract_pdf(output_path)
    except Exception:
        return ""
    finally:
        if output_path and output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass

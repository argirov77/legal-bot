"""Utilities for detecting the format of uploaded documents."""
from __future__ import annotations

import mimetypes
from enum import Enum
from pathlib import Path
from typing import Optional


class DocumentFormat(str, Enum):
    """Supported document formats."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class DocumentFormatDetector:
    """Detects the document format based on file name and optional MIME type."""

    _MIME_MAP = {
        "application/pdf": DocumentFormat.PDF,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentFormat.DOCX,
        "text/plain": DocumentFormat.TXT,
    }

    @classmethod
    def detect(cls, file_name: str, mime_type: Optional[str] = None) -> DocumentFormat:
        """Return the detected document format.

        The detector first considers an explicit MIME type value, falling back to
        `mimetypes.guess_type` and finally checking the file suffix.
        """

        if mime_type and mime_type in cls._MIME_MAP:
            return cls._MIME_MAP[mime_type]

        guessed_type, _ = mimetypes.guess_type(file_name)
        if guessed_type and guessed_type in cls._MIME_MAP:
            return cls._MIME_MAP[guessed_type]

        suffix = Path(file_name).suffix.lower().lstrip(".")
        try:
            return DocumentFormat(suffix)
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Unsupported file format: {file_name}") from exc

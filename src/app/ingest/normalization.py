"""Text normalisation utilities."""
from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")


def normalize_text(text: str) -> str:
    """Normalise whitespace and Unicode representation."""

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    normalized = _TRAILING_SPACE_RE.sub("\n", normalized)
    normalized = _MULTIPLE_NEWLINES_RE.sub("\n\n", normalized)
    return normalized.strip()

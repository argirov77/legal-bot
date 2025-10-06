"""Language detection helpers."""
from __future__ import annotations

import logging
from typing import Optional

from langdetect import DetectorFactory, LangDetectException, detect

LOGGER = logging.getLogger(__name__)
DetectorFactory.seed = 0


class LanguageDetector:
    """Wraps langdetect providing a robust API."""

    def detect(self, text: str) -> Optional[str]:
        cleaned = text.strip()
        if not cleaned:
            return None
        try:
            language = detect(cleaned)
            LOGGER.debug("Detected language: %s", language)
            return language
        except LangDetectException:
            LOGGER.info("Unable to determine language for text of length %s", len(text))
            return None

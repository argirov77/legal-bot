from __future__ import annotations

import re
from typing import Dict, List


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Split *text* into overlapping chunks.

    Each chunk is represented as a dictionary with the chunked ``text`` and
    metadata describing the ``start`` and ``end`` offsets in the original text.
    The function attempts to split on whitespace to avoid breaking words while
    still respecting the desired ``chunk_size`` and ``overlap``.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if not text:
        return []

    text_length = len(text)
    chunks: List[Dict] = []
    start = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            window = text[start:end]
            split_at = None
            for match in re.finditer(r"\s", window):
                position = match.start()
                if position == 0:
                    continue
                split_at = position
            if split_at is not None:
                candidate_end = start + split_at + 1
                if start < candidate_end < end:
                    end = candidate_end

        chunk_text_value = text[start:end]
        chunk_meta = {"start": start, "end": end}
        chunks.append({"text": chunk_text_value, "meta": chunk_meta})

        if end == text_length:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks

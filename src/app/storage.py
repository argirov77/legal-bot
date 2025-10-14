"""Utilities for persisting user uploads on disk."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Final
from uuid import uuid4

from fastapi import UploadFile

_DATA_DIR: Final[Path] = Path("data")
_FILENAME_SAFE_CHARS_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe filename preserving the extension when possible."""
    if not filename:
        filename = "upload"
    # Remove any path components and replace disallowed characters.
    sanitized = Path(filename).name
    sanitized = _FILENAME_SAFE_CHARS_RE.sub("_", sanitized)
    sanitized = sanitized.strip("._") or "upload"
    return sanitized


async def save_upload(session_id: str, upload: UploadFile) -> Path:
    """Persist an uploaded file inside the data directory for the given session."""
    session_dir = _DATA_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    sanitized_name = _sanitize_filename(upload.filename or "")
    base = Path(sanitized_name).stem or "upload"
    suffix = Path(sanitized_name).suffix
    unique_name = f"{base}-{uuid4().hex}{suffix}" if suffix else f"{base}-{uuid4().hex}"
    destination = session_dir / unique_name

    contents = await upload.read()
    destination.write_bytes(contents)
    upload.file.seek(0)

    return destination.resolve()

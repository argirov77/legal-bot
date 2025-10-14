from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import SpooledTemporaryFile

from starlette.datastructures import Headers, UploadFile

from app.logging_config import configure_logging
from app.rag_service import RAGService


def _create_upload_file(content: str, filename: str = "audit.txt") -> UploadFile:
    temp_file = SpooledTemporaryFile()
    temp_file.write(content.encode("utf-8"))
    temp_file.seek(0)
    headers = Headers({"content-type": "text/plain"})
    return UploadFile(file=temp_file, filename=filename, headers=headers)


def test_ingest_writes_audit_log(tmp_path: Path) -> None:
    log_path = Path("logs/ingest_audit.log")
    if log_path.exists():
        log_path.unlink()

    configure_logging()

    async def runner() -> None:
        service = RAGService()
        upload_file = _create_upload_file("Example content for audit logging.")

        await service.ingest("audit-session", [upload_file])

    asyncio.run(runner())

    assert log_path.exists(), "Audit log file was not created"

    with log_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    assert lines, "Audit log file is empty"
    last_entry = json.loads(lines[-1])

    assert last_entry["session_id"] == "audit-session"
    assert last_entry["filename"] == "audit.txt"
    assert "timestamp" in last_entry
    assert last_entry["chunks"] >= 0

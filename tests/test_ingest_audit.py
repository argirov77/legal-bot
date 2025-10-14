import asyncio
import importlib
import io
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_ingest_module(monkeypatch):
    try:
        return importlib.import_module("app.ingest"), importlib.import_module("app.logging_config")
    except Exception:
        sys.modules.pop("app.ingest", None)

        fastapi_stub = ModuleType("fastapi")

        class DummyAPIRouter:
            def __init__(self, *args, **kwargs):
                self.routes = []

            def add_api_route(self, *args, **kwargs):
                self.routes.append((args, kwargs))

            def post(self, *args, **kwargs):
                def decorator(func):
                    self.add_api_route(*args, endpoint=func, **kwargs)
                    return func

                return decorator

        fastapi_stub.APIRouter = DummyAPIRouter
        fastapi_stub.UploadFile = object

        def file_dependency(*_args, **_kwargs):
            return None

        fastapi_stub.File = file_dependency
        monkeypatch.setitem(sys.modules, "fastapi", fastapi_stub)

        st_stub = ModuleType("sentence_transformers")

        class DummySentenceTransformer:
            def __init__(self, *args, **kwargs):
                pass

            def encode(self, chunks, show_progress_bar=False):
                return [[0.0] * 3 for _ in chunks]

        st_stub.SentenceTransformer = DummySentenceTransformer
        monkeypatch.setitem(sys.modules, "sentence_transformers", st_stub)

        chromadb_stub = ModuleType("chromadb")

        class DummyClient:
            def __init__(self, *args, **kwargs):
                pass

            def get_collection(self, name):
                raise RuntimeError("not found")

            def create_collection(self, name):
                return SimpleNamespace(add=lambda **kwargs: None)

            def persist(self):
                pass

        chromadb_stub.Client = DummyClient
        monkeypatch.setitem(sys.modules, "chromadb", chromadb_stub)

        chromadb_config_stub = ModuleType("chromadb.config")
        chromadb_config_stub.Settings = SimpleNamespace
        monkeypatch.setitem(sys.modules, "chromadb.config", chromadb_config_stub)

        pdfminer_stub = ModuleType("pdfminer")
        pdfminer_high_level_stub = ModuleType("pdfminer.high_level")

        def extract_text_to_fp(_file, out):
            out.write("")

        pdfminer_high_level_stub.extract_text_to_fp = extract_text_to_fp
        monkeypatch.setitem(sys.modules, "pdfminer", pdfminer_stub)
        monkeypatch.setitem(sys.modules, "pdfminer.high_level", pdfminer_high_level_stub)

        docx_stub = ModuleType("docx")

        class DummyDocument:
            def __init__(self, *args, **kwargs):
                self.paragraphs = []

        docx_stub.Document = DummyDocument
        monkeypatch.setitem(sys.modules, "docx", docx_stub)

        ingest_module = importlib.import_module("app.ingest")
        logging_config_module = importlib.import_module("app.logging_config")
        return ingest_module, logging_config_module


def test_ingest_writes_audit_log(tmp_path, monkeypatch):
    logs_dir = Path("logs")
    existed_before = logs_dir.exists()

    ingest_module, logging_config_module = _load_ingest_module(monkeypatch)

    audit_path = tmp_path / "logs" / "ingest_audit.log"
    monkeypatch.setattr(logging_config_module, "_AUDIT_LOG_PATH", audit_path)

    audit_logger = ingest_module.audit_logger
    for handler in list(audit_logger.handlers):
        audit_logger.removeHandler(handler)
        handler.close()
    setattr(audit_logger, "_audit_configured", False)
    new_logger = logging_config_module.get_ingest_audit_logger()
    monkeypatch.setattr(ingest_module, "audit_logger", new_logger)

    dummy_collection = SimpleNamespace(add=lambda **_: None)

    monkeypatch.setattr(ingest_module, "get_or_create_collection", lambda session_id: dummy_collection)
    monkeypatch.setattr(ingest_module.chroma_client, "persist", lambda: None)

    def fake_encode(chunks, show_progress_bar=False):
        return [[0.0, 0.0, 0.0] for _ in chunks]

    monkeypatch.setattr(ingest_module.embedder, "encode", fake_encode)

    def fake_save(session_id, upload):
        dest_dir = tmp_path / session_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = upload.filename or "upload.txt"
        dest = dest_dir / filename
        with open(dest, "wb") as f:
            f.write(upload.file.read())
        upload.file.seek(0)
        return str(dest)

    monkeypatch.setattr(ingest_module, "save_upload_to_disk", fake_save)

    session_id = "audit-session"
    upload = SimpleNamespace(file=io.BytesIO(b"Audit log content"), filename="sample.txt")

    try:
        response = asyncio.run(ingest_module.ingest(session_id, files=[upload]))

        assert response["status"] == "ok"
        assert audit_path.exists()

        lines = audit_path.read_text(encoding="utf-8").splitlines()
        assert lines, "Audit log should contain at least one entry"

        entry = json.loads(lines[-1])
        assert entry["session_id"] == session_id
        assert entry["filename"] == "sample.txt"
        assert "timestamp" in entry
        assert "chunks" in entry
    finally:
        if not existed_before and logs_dir.exists():
            shutil.rmtree(logs_dir)

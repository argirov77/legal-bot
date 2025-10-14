import asyncio
import io
from pathlib import Path

import chromadb
from chromadb.config import Settings
from starlette.datastructures import UploadFile

import app.ingest as ingest_module


def test_ingest_and_chroma_persist(tmp_path):
    session_id = "test-session"
    persist_dir = tmp_path / "chroma"
    ingest_module.CHROMA_PERSIST_DIR = str(persist_dir)
    ingest_module.chroma_client = chromadb.Client(
        Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(persist_dir))
    )

    content = "This is a sample contract text. " * 10
    upload = UploadFile(filename="sample.txt", file=io.BytesIO(content.encode("utf-8")))

    result = asyncio.run(ingest_module.ingest(session_id, files=[upload]))

    assert result["status"] == "ok"
    assert result["added_chunks"] > 0
    assert any(Path(path).exists() for path in result["saved_files"])

    collection_name = f"session_{session_id}"
    collections = list(ingest_module.chroma_client.list_collections())
    assert any(collection.name == collection_name for collection in collections)
    collection = ingest_module.chroma_client.get_collection(name=collection_name)
    query = collection.query(query_texts=["sample"], n_results=1, include=["documents"])
    assert query["documents"][0]

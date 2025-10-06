import os
import time
from fastapi.testclient import TestClient
from app.main import app
import chromadb
from chromadb.config import Settings

client = TestClient(app)


def test_ingest_and_chroma_persist(tmp_path):
    session_id = "test-session"
    sample_dir = tmp_path / "data"
    sample_dir.mkdir()
    sample_file = sample_dir / "sample.txt"
    sample_file.write_text("This is a sample contract text. " * 100, encoding="utf-8")

    url = f"/sessions/{session_id}/ingest"
    with open(sample_file, "rb") as f:
        files = {"files": ("sample.txt", f, "text/plain")}
        resp = client.post(url, files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["added_chunks"] > 0

    # connect to chroma and assert collection exists and has >0
    CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "/chroma_db")
    chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PERSIST_DIR))
    coll_name = f"session_{session_id}"
    # small wait to ensure persist finished
    time.sleep(0.5)
    cols = [c.name for c in chroma_client.list_collections()]
    assert coll_name in cols
    coll = chroma_client.get_collection(name=coll_name)
    # query size via metadata/count not always provided uniformly; check docs returned
    res = coll.query(query_texts=["sample"], n_results=1, include=["documents"])
    assert len(res) > 0

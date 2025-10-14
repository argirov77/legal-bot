from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_ingest_and_query_roundtrip(tmp_path):
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

    query_resp = client.post(
        f"/sessions/{session_id}/query",
        json={"question": "What is the document about?", "top_k": 1},
    )
    assert query_resp.status_code == 200
    query_body = query_resp.json()
    assert query_body["answer"]
    assert query_body["sources"]

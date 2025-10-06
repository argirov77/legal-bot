from fastapi.testclient import TestClient

from app.main import app


def test_read_root_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == "ok"

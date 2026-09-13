from fastapi.testclient import TestClient

from app import main

def test_health_endpoint(monkeypatch):
    """Check that the API starts and its health endpoint responds."""
    monkeypatch.setattr(main.registry, "load_all", lambda: None)

    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

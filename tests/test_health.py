"""Health endpoint."""

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    # Reports the provider too, so "is this real data?" is answerable without a request.
    assert response.json() == {"status": "ok", "classifier": "mock"}

"""Tests for the X-API-Key gate.

The shared ``client`` fixture runs with ``VOCAB_ACCESS_KEY=disabled``, so these tests
build their own client after setting a real key — the gate reads the environment during
app startup, which happens when the TestClient is entered.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.security import access_control_enabled, require_access_key, validate_access_control

ACCESS_KEY = "test-access-key-not-a-real-secret"

PAYLOAD = {"text": "The accountant reviewed every invoice."}
EXTRACT_URL = "/api/v1/vocabulary/extract"


@pytest.fixture()
def guarded_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A client whose server requires ``X-API-Key: ACCESS_KEY``."""
    monkeypatch.setenv("VOCAB_ACCESS_KEY", ACCESS_KEY)
    with TestClient(app) as client:
        yield client


def test_request_without_the_header_is_rejected(guarded_client: TestClient) -> None:
    response = guarded_client.post(EXTRACT_URL, json=PAYLOAD)

    assert response.status_code == 401


def test_request_with_the_wrong_key_is_rejected(guarded_client: TestClient) -> None:
    response = guarded_client.post(
        EXTRACT_URL, json=PAYLOAD, headers={"X-API-Key": "wrong-key"}
    )

    assert response.status_code == 401


def test_non_ascii_key_is_rejected_not_a_server_error(guarded_client: TestClient) -> None:
    """compare_digest raises TypeError on non-ASCII str, which would surface as a 500.

    Sent as bytes because that is what a header is on the wire: the client refuses to
    encode a non-ASCII ``str``, while starlette decodes the received bytes as latin-1 and
    hands the dependency a ``str`` outside ASCII. Only reachable from a real socket.
    """
    response = guarded_client.post(
        EXTRACT_URL, json=PAYLOAD, headers={"X-API-Key": "khóa-sai".encode("latin-1")}
    )

    assert response.status_code == 401


def test_rejection_never_echoes_the_expected_key(guarded_client: TestClient) -> None:
    """A 401 that leaks the answer is worse than no gate at all."""
    response = guarded_client.post(
        EXTRACT_URL, json=PAYLOAD, headers={"X-API-Key": "wrong-key"}
    )

    assert ACCESS_KEY not in response.text


def test_request_with_the_right_key_is_served(guarded_client: TestClient) -> None:
    response = guarded_client.post(
        EXTRACT_URL, json=PAYLOAD, headers={"X-API-Key": ACCESS_KEY}
    )

    assert response.status_code == 200
    assert response.json()["classifier"] == "mock"


def test_health_needs_no_key(guarded_client: TestClient) -> None:
    """Render calls /health as its health check; gating it would block every deploy."""
    response = guarded_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_does_not_expose_either_key(guarded_client: TestClient) -> None:
    assert ACCESS_KEY not in guarded_client.get("/health").text


def test_disabled_serves_without_a_header(client: TestClient) -> None:
    """The shared fixture sets VOCAB_ACCESS_KEY=disabled, which opens the endpoint."""
    assert client.post(EXTRACT_URL, json=PAYLOAD).status_code == 200


def test_unset_access_key_refuses_to_start() -> None:
    """Deploying without deciding must be a startup error, not a silently open URL."""
    with pytest.raises(RuntimeError, match="VOCAB_ACCESS_KEY"):
        validate_access_control(Settings(classifier="mock"))


def test_disabled_starts_but_is_reported_as_off() -> None:
    settings = Settings(classifier="mock", vocab_access_key="disabled")

    validate_access_control(settings)

    assert access_control_enabled(settings) is False
    assert access_control_enabled(Settings(classifier="mock", vocab_access_key="k")) is True


def test_dependency_refuses_rather_than_allowing_when_unset() -> None:
    """Belt and braces: the gate must never read "no key configured" as "let them in"."""
    with pytest.raises(RuntimeError, match="VOCAB_ACCESS_KEY"):
        require_access_key(x_api_key=None, settings=Settings(classifier="mock"))


def test_access_key_is_not_exposed_in_repr() -> None:
    """Same guarantee as llm_api_key: Settings gets logged, so it must not carry secrets."""
    settings = Settings(classifier="mock", vocab_access_key=ACCESS_KEY)

    assert ACCESS_KEY not in repr(settings)

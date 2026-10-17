"""Shared test fixtures."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app

@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Run every test against a known configuration.

    Two separate isolations. First, a developer's exported vars must not change test
    outcomes — cleared from ``config.ENV_VARS`` rather than a list kept here, so a new
    variable cannot be added to the app and forgotten in the fixture. Second,
    ``load_settings`` now reads ``.env`` — so point it at a path that does not exist, or a
    real local ``.env`` would leak a live key into the suite.

    The two explicit values are set because neither has a default any more: without them
    every test touching ``client`` would get the startup refusal instead of a response.
    """
    for name in config.ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        config, "DOTENV_PATH", tmp_path_factory.getbasetemp() / "no-such.env"
    )
    monkeypatch.setenv("VOCAB_CLASSIFIER", "mock")
    # The existing tests are about the API, not the gate; test_access_key.py overrides
    # this to exercise the gate itself.
    monkeypatch.setenv("VOCAB_ACCESS_KEY", "disabled")


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client

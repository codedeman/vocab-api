"""Configuration and the provider seam.

These tests are what make the Phase 2 claim concrete: the API layer talks to whatever
``get_classifier`` returns, and the provider is chosen by environment variable.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config
from app.classifier import (
    ClassifierError,
    MockClassifier,
    OpenAICompatibleClassifier,
    get_classifier,
)
from app.config import load_settings
from app.main import app
from app.schemas import VocabularyItem
from app.taxonomy import MainCategory

EXTRACT_URL = "/api/v1/vocabulary/extract"


def test_unset_provider_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """There is no default. Silently returning the mock is the bug this prevents."""
    monkeypatch.delenv("VOCAB_CLASSIFIER", raising=False)

    with pytest.raises(ValueError) as excinfo:
        get_classifier()

    message = str(excinfo.value)
    assert "VOCAB_CLASSIFIER" in message
    assert "openai_compatible" in message
    assert "mock" in message


def test_explicit_mock_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOCAB_CLASSIFIER", "mock")

    assert isinstance(get_classifier(), MockClassifier)


def test_unknown_provider_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOCAB_CLASSIFIER", "bogus")

    with pytest.raises(ValueError) as excinfo:
        get_classifier()

    message = str(excinfo.value)
    assert "bogus" in message
    assert "mock" in message


def test_openai_compatible_provider_is_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOCAB_CLASSIFIER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "sk-test-value")

    assert isinstance(get_classifier(), OpenAICompatibleClassifier)


def test_openai_compatible_missing_config_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOCAB_CLASSIFIER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        get_classifier()


def test_settings_read_provider_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOCAB_CLASSIFIER", "mock")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    monkeypatch.setenv("LLM_API_KEY", "sk-secret-value")

    settings = load_settings()

    assert settings.classifier == "mock"
    assert settings.llm_base_url == "https://api.deepseek.com/v1"
    assert settings.llm_model == "deepseek-chat"
    assert settings.llm_api_key == "sk-secret-value"


def test_blank_env_values_are_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOCAB_CLASSIFIER", "   ")
    monkeypatch.setenv("LLM_BASE_URL", "")

    settings = load_settings()

    assert settings.classifier is None
    assert settings.llm_base_url is None


def test_dotenv_file_is_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The missing half of the fix: a correct .env used to be ignored entirely."""
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# a comment\n"
        "\n"
        "VOCAB_CLASSIFIER=openai_compatible\n"
        'LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"\n'
        "LLM_MODEL = gemini-3.8-flash \n"
        "LLM_API_KEY=key-from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "DOTENV_PATH", dotenv)
    for name in ("VOCAB_CLASSIFIER", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    settings = load_settings()

    assert settings.classifier == "openai_compatible"
    assert settings.llm_base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    # Whitespace around both the name and the value is stripped, quotes too.
    assert settings.llm_model == "gemini-3.8-flash"
    assert settings.llm_api_key == "key-from-file"


def test_real_env_wins_over_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """So `VOCAB_CLASSIFIER=mock uvicorn ...` still works with a Gemini .env on disk."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("VOCAB_CLASSIFIER=openai_compatible\n", encoding="utf-8")
    monkeypatch.setattr(config, "DOTENV_PATH", dotenv)
    monkeypatch.setenv("VOCAB_CLASSIFIER", "mock")

    assert load_settings().classifier == "mock"


def test_missing_dotenv_file_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "DOTENV_PATH", tmp_path / "absent.env")
    monkeypatch.setenv("VOCAB_CLASSIFIER", "mock")

    assert load_settings().classifier == "mock"


def test_gemini_configuration_builds_the_expected_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini's base URL ends in a slash; rstrip must not yield //chat/completions."""
    monkeypatch.setenv("VOCAB_CLASSIFIER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("LLM_API_KEY", "sk-test-value")

    classifier = get_classifier()

    assert isinstance(classifier, OpenAICompatibleClassifier)
    assert (
        f"{classifier._base_url}/chat/completions"
        == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )


def test_response_names_the_provider(client: TestClient) -> None:
    """The response is self-describing, so mock data can't be mistaken for real data."""
    body = client.post(EXTRACT_URL, json={"text": "She filed the invoice."}).json()

    assert body["classifier"] == "mock"


def test_api_key_is_not_exposed_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-secret-value")

    assert "sk-secret-value" not in repr(load_settings())


def test_route_uses_the_injected_classifier(client: TestClient) -> None:
    """Overriding the dependency changes the response, so Phase 2 needs no API change."""

    class FakeClassifier:
        def extract(self, text: str) -> list[VocabularyItem]:
            return [
                VocabularyItem(
                    word="fake",
                    part_of_speech="noun",
                    meaning_vi="giả",
                    cefr_level="C2",
                    main_category=MainCategory.SCIENCE_ENVIRONMENT,
                    sub_category="Injected",
                    example="A fake item from an injected provider.",
                    definition_en="A placeholder entry produced by an injected classifier.",
                    usage_vi="Chỉ dùng trong test để chứng minh route lấy item từ provider được inject.",
                )
            ]

    app.dependency_overrides[get_classifier] = FakeClassifier
    try:
        body = client.post(EXTRACT_URL, json={"text": "anything at all"}).json()
    finally:
        app.dependency_overrides.clear()

    assert body["count"] == 1
    assert body["items"][0]["word"] == "fake"
    assert body["items"][0]["sub_category"] == "Injected"


def test_provider_failure_is_a_readable_502(client: TestClient) -> None:
    """A provider failure used to be a 500 whose body was "Internal Server Error"."""
    detail = "provider returned 404 for model gemini-2.0-flash: no longer available"

    class BrokenClassifier:
        def extract(self, text: str) -> list[VocabularyItem]:
            raise ClassifierError(detail)

    app.dependency_overrides[get_classifier] = BrokenClassifier
    try:
        response = client.post(EXTRACT_URL, json={"text": "insecure"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert response.json() == {"detail": detail}

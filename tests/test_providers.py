"""Provider detection from the key: pasting a different LLM_API_KEY is enough to switch."""

import logging

import pytest
from fastapi.testclient import TestClient

from app.classifier import MockClassifier, OpenAICompatibleClassifier, get_classifier
from app.config import load_settings
from app.main import app
from app.providers import GEMINI, GROQ, OPENAI, OPENROUTER, detect_provider

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("AIzaSyTestValue", GEMINI),
        ("AQ.TestValue", GEMINI),
        ("sk-or-v1-test", OPENROUTER),
        ("gsk_test", GROQ),
        ("sk-proj-test", OPENAI),
        ("sk-test", OPENAI),
    ],
)
def test_key_prefix_picks_the_provider(key: str, expected: object) -> None:
    assert detect_provider(key) == expected


def test_unrecognised_key_is_not_guessed() -> None:
    assert detect_provider("ollama") is None
    assert detect_provider(None) is None


def test_anthropic_key_is_refused_by_name() -> None:
    """It starts with sk- too; read as OpenAI it would only surface as a 401 later."""
    with pytest.raises(ValueError, match="Anthropic"):
        detect_provider("sk-ant-api03-test")


def test_key_alone_configures_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOCAB_CLASSIFIER")
    monkeypatch.setenv("LLM_API_KEY", "sk-proj-test")

    settings = load_settings()

    assert settings.classifier == "openai_compatible"
    assert settings.llm_provider == "openai"
    assert settings.llm_base_url == "https://api.openai.com/v1"
    assert settings.llm_model == "gpt-4o-mini"
    # OpenAI rejects the field on a non-thinking model, so it must not be sent.
    assert settings.llm_reasoning_effort is None
    assert isinstance(get_classifier(), OpenAICompatibleClassifier)


def test_key_alone_configures_gemini_with_its_thinking_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOCAB_CLASSIFIER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "AQ.test")

    settings = load_settings()

    assert settings.llm_provider == "gemini"
    assert settings.llm_base_url == GEMINI_URL
    assert settings.llm_model == "gemini-3.8-flash"
    assert settings.llm_reasoning_effort == "low"


def test_explicit_model_overrides_the_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOCAB_CLASSIFIER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")

    settings = load_settings()

    assert settings.llm_model == "gpt-4.1-mini"
    assert settings.llm_base_url == "https://api.openai.com/v1"


def test_custom_endpoint_gets_no_guessed_model_or_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Gemini key behind a proxy must not get reasoning_effort sent unasked."""
    monkeypatch.setenv("VOCAB_CLASSIFIER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "AIzaTest")
    monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example.com/v1")

    settings = load_settings()

    assert settings.llm_base_url == "https://proxy.example.com/v1"
    assert settings.llm_model is None
    assert settings.llm_reasoning_effort is None


def test_mock_ignores_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOCAB_CLASSIFIER", "mock")
    monkeypatch.setenv("LLM_API_KEY", "sk-ant-would-be-refused")

    settings = load_settings()

    assert settings.llm_base_url is None
    assert isinstance(get_classifier(), MockClassifier)


def test_unrecognised_key_alone_still_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A recognised key may stand in for VOCAB_CLASSIFIER; an unknown one may not."""
    monkeypatch.delenv("VOCAB_CLASSIFIER")
    monkeypatch.setenv("LLM_API_KEY", "some-other-key")

    with pytest.raises(ValueError, match="VOCAB_CLASSIFIER"):
        get_classifier()


def test_key_wins_over_another_providers_leftover_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Render case: a new OpenAI key, with Gemini's URL, model and budget still set."""
    monkeypatch.setenv("VOCAB_CLASSIFIER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "sk-proj-test")
    monkeypatch.setenv("LLM_BASE_URL", GEMINI_URL)
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")

    settings = load_settings()

    assert settings.llm_provider == "openai"
    assert settings.llm_base_url == "https://api.openai.com/v1"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.llm_reasoning_effort is None
    assert settings.llm_ignored == ("LLM_BASE_URL", "LLM_MODEL", "LLM_REASONING_EFFORT")
    assert settings.llm_stale_host == "gemini"
    assert isinstance(get_classifier(), OpenAICompatibleClassifier)


def test_leftover_settings_are_named_in_a_startup_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("VOCAB_CLASSIFIER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "sk-proj-secret-value")
    monkeypatch.setenv("LLM_BASE_URL", GEMINI_URL)

    with caplog.at_level(logging.WARNING, logger="vocab-api"), TestClient(app):
        pass

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("ignoring LLM_BASE_URL left over from gemini" in w for w in warnings)
    assert all("sk-proj-secret-value" not in r.getMessage() for r in caplog.records)


def test_sk_key_on_deepseek_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOCAB_CLASSIFIER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")

    settings = load_settings()

    assert settings.llm_base_url == "https://api.deepseek.com/v1"
    assert settings.llm_model == "deepseek-chat"
    assert settings.llm_ignored == ()


def test_any_key_on_an_unknown_host_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOCAB_CLASSIFIER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "llama3.1")

    settings = load_settings()

    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_model == "llama3.1"
    assert settings.llm_ignored == ()

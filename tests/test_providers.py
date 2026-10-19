"""Provider detection from the key: pasting a different LLM_API_KEY is enough to switch."""

import pytest

from app.classifier import MockClassifier, OpenAICompatibleClassifier, get_classifier
from app.config import Settings, load_settings
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


def _classifier(key: str, base_url: str) -> OpenAICompatibleClassifier:
    return OpenAICompatibleClassifier(
        Settings(
            classifier="openai_compatible",
            llm_base_url=base_url,
            llm_model="any-model",
            llm_api_key=key,
        )
    )


def test_key_for_another_known_host_is_refused_at_startup() -> None:
    """The trap: a new OpenAI key pasted while the old Gemini URL is still set."""
    with pytest.raises(ValueError) as excinfo:
        _classifier("sk-proj-test", GEMINI_URL)

    message = str(excinfo.value)
    assert "openai" in message
    assert "gemini" in message
    assert "LLM_BASE_URL" in message
    assert "sk-proj-test" not in message


def test_sk_key_on_deepseek_is_allowed() -> None:
    _classifier("sk-test", "https://api.deepseek.com/v1")


def test_any_key_on_an_unknown_host_is_allowed() -> None:
    _classifier("sk-test", "http://localhost:11434/v1")
    _classifier("AIzaTest", "https://proxy.example.com/v1")

"""Tests for the OpenAI-compatible classifier, with HTTP mocked at _post."""

import pytest

from app.classifier import ClassifierError, FatalUpstreamError, OpenAICompatibleClassifier
from app.classifier.openai_compatible import (
    MAX_UPSTREAM_DETAIL,
    SYSTEM_PROMPT,
    TIMEOUT_SECONDS,
)
from app.config import Settings

VALID_CONTENT = (
    '{"items": [{"word": "invoice", "part_of_speech": "noun", "meaning_vi": "hóa đơn", '
    '"cefr_level": "B2", "main_category": "Finance", "sub_category": "Banking & Payments", '
    '"example": "The accountant reviewed every invoice.", '
    '"word_family": [{"word": "invoicing", "part_of_speech": "noun"}], '
    '"examples": ["Please send the invoice by Friday.", "They lost the invoice."], '
    '"definition_en": "A document listing goods sold and the money owed for them.", '
    '"usage_vi": "Dùng với send/issue/pay an invoice, phổ biến trong email thương mại."'
    "}], \"count\": 1}"
)

#: Everything except ``usage_vi``: the new fields are enforced at the validation boundary,
#: so a reply that omits one is rejected exactly like malformed JSON.
MISSING_USAGE_CONTENT = VALID_CONTENT.replace('"usage_vi"', '"usage_notes"')


def _settings() -> Settings:
    return Settings(
        classifier="openai_compatible",
        llm_base_url="https://example.com/v1",
        llm_model="test-model",
        llm_api_key="sk-test-value",
    )


def test_extract_returns_validated_items(monkeypatch: pytest.MonkeyPatch) -> None:
    classifier = OpenAICompatibleClassifier(_settings())
    monkeypatch.setattr(classifier, "_post", lambda messages: VALID_CONTENT)

    items = classifier.extract("The accountant reviewed every invoice.")

    assert len(items) == 1
    assert items[0].word == "invoice"
    assert items[0].meaning_vi == "hóa đơn"
    assert items[0].word_family[0].word == "invoicing"
    assert len(items[0].examples) == 2
    assert items[0].definition_en.startswith("A document")
    assert "invoice" in items[0].usage_vi


def test_invalid_then_valid_retries_once(monkeypatch: pytest.MonkeyPatch) -> None:
    classifier = OpenAICompatibleClassifier(_settings())
    calls = {"n": 0}

    def fake_post(messages: list) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"
        return VALID_CONTENT

    monkeypatch.setattr(classifier, "_post", fake_post)

    items = classifier.extract("The accountant reviewed every invoice.")

    assert calls["n"] == 2
    assert items[0].word == "invoice"


def test_double_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    classifier = OpenAICompatibleClassifier(_settings())
    monkeypatch.setattr(classifier, "_post", lambda messages: "still not json")

    with pytest.raises(ClassifierError, match="2 attempts"):
        classifier.extract("anything")


def _failing_post(classifier: OpenAICompatibleClassifier, status: int, body: str):
    """Replace ``_post`` with the upstream failure it would have raised, counting calls."""
    calls = {"n": 0}

    def fake_post(messages: list) -> str:
        calls["n"] += 1
        raise classifier._upstream_error(status, body)

    return calls, fake_post


RETIRED_MODEL_BODY = (
    '{"error": {"code": 404, "message": "This model models/gemini-2.0-flash is no longer '
    'available. Please update your code to use models/gemini-3.8-flash for the latest '
    'features and improvements.", "status": "NOT_FOUND"}}'
)


def test_retired_model_is_not_retried_and_keeps_googles_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real 500 this came from: a second attempt cannot un-retire a model."""
    classifier = OpenAICompatibleClassifier(_settings())
    calls, fake_post = _failing_post(classifier, 404, RETIRED_MODEL_BODY)
    monkeypatch.setattr(classifier, "_post", fake_post)

    with pytest.raises(ClassifierError) as excinfo:
        classifier.extract("insecure")

    assert calls["n"] == 1, "a 404 must not be retried"
    message = str(excinfo.value)
    assert "404" in message
    assert "test-model" in message
    # The actionable half of Google's reply has to survive all the way to the caller.
    assert "models/gemini-3.8-flash" in message


@pytest.mark.parametrize("status", [401, 400, 403, 404], ids=["auth", "bad-request", "forbidden", "not-found"])
def test_unfixable_statuses_fail_immediately(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    classifier = OpenAICompatibleClassifier(_settings())
    calls, fake_post = _failing_post(classifier, status, "nope")
    monkeypatch.setattr(classifier, "_post", fake_post)

    with pytest.raises(FatalUpstreamError):
        classifier.extract("anything")

    assert calls["n"] == 1


@pytest.mark.parametrize("status", [429, 408, 500, 503], ids=["rate-limit", "timeout", "server", "unavailable"])
def test_transient_statuses_are_retried(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    """Retrying these is the whole point of having a retry."""
    classifier = OpenAICompatibleClassifier(_settings())
    calls, fake_post = _failing_post(classifier, status, "try again")
    monkeypatch.setattr(classifier, "_post", fake_post)

    with pytest.raises(ClassifierError, match="2 attempts"):
        classifier.extract("anything")

    assert calls["n"] == 2
    assert not isinstance(
        classifier._upstream_error(status, "x"), FatalUpstreamError
    ), f"{status} must stay retryable"


def test_upstream_detail_is_truncated_and_flattened() -> None:
    classifier = OpenAICompatibleClassifier(_settings())

    error = classifier._upstream_error(500, "line one\n   line two\n" + "x" * 1000)

    message = str(error)
    assert "line one line two" in message, "newlines and runs of spaces collapse to one space"
    assert len(message) < MAX_UPSTREAM_DETAIL + 120, "the body is capped, not echoed whole"


def test_empty_upstream_body_still_says_something() -> None:
    classifier = OpenAICompatibleClassifier(_settings())

    assert "(empty response body)" in str(classifier._upstream_error(502, ""))


def test_timeout_leaves_room_for_a_full_length_extraction() -> None:
    """15 items measured at ~28s with reasoning_effort=low, so 30s was not survivable.

    The timeout is a safety net, not the fix: without ``LLM_REASONING_EFFORT`` the same
    request did not finish in 600s, which no ceiling can repair.
    """
    assert TIMEOUT_SECONDS >= 120


def test_reasoning_effort_is_omitted_unless_configured() -> None:
    """Sending it unconditionally would 400 on providers with no thinking mode."""
    classifier = OpenAICompatibleClassifier(_settings())

    assert "reasoning_effort" not in classifier._request_body([])


def test_reasoning_effort_is_sent_when_configured() -> None:
    """The measured fix for the 600s hang: Gemini thinks until told to stop."""
    settings = Settings(
        classifier="openai_compatible",
        llm_base_url="https://example.com/v1",
        llm_model="gemini-3.8-flash",
        llm_api_key="sk-test-value",
        llm_reasoning_effort="low",
    )

    body = OpenAICompatibleClassifier(settings)._request_body([])

    assert body["reasoning_effort"] == "low"
    # The rest of the payload is untouched, so JSON mode still applies.
    assert body["response_format"] == {"type": "json_object"}


def test_reply_missing_a_new_field_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    classifier = OpenAICompatibleClassifier(_settings())
    monkeypatch.setattr(classifier, "_post", lambda messages: MISSING_USAGE_CONTENT)

    with pytest.raises(ClassifierError, match="2 attempts"):
        classifier.extract("The accountant reviewed every invoice.")


def test_llm_reply_does_not_need_the_classifier_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """VALID_CONTENT has no "classifier" key, and must not need one.

    ``classifier`` is the service's own knowledge, so it lives on ``ExtractResponse``
    while the model is validated against ``ClassifierPayload``.
    """
    assert "classifier" not in VALID_CONTENT
    classifier = OpenAICompatibleClassifier(_settings())
    monkeypatch.setattr(classifier, "_post", lambda messages: VALID_CONTENT)

    assert classifier.extract("The accountant reviewed every invoice.")[0].word == "invoice"


def test_single_word_input_is_covered_by_the_prompt() -> None:
    """A bare word has no sentence to quote, so the prompt must say what to do instead."""
    assert "single word" in SYSTEM_PROMPT
    assert "instead of echoing the input" in SYSTEM_PROMPT


def test_missing_each_config_var_raises() -> None:
    base = {"classifier": "openai_compatible", "llm_base_url": "https://example.com/v1",
            "llm_model": "m", "llm_api_key": "k"}
    for missing in ("llm_base_url", "llm_model", "llm_api_key"):
        kwargs = {k: v for k, v in base.items() if k != missing}
        with pytest.raises(ValueError, match=missing.upper()):
            OpenAICompatibleClassifier(Settings(**kwargs))  # type: ignore[arg-type]

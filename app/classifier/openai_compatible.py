"""OpenAI-compatible chat completions classifier.

One adapter covers DeepSeek, OpenAI, Groq, Together, OpenRouter, Ollama, vLLM and
Gemini's OpenAI-compatible endpoint — only ``LLM_BASE_URL`` and ``LLM_MODEL`` change.
Sends ``response_format: {"type": "json_object"}`` (the common denominator DeepSeek
supports, which Gemini maps to ``responseMimeType: application/json``) and validates the
reply with ``ClassifierPayload``.

On a thinking model, set ``LLM_REASONING_EFFORT`` (Gemini: ``low``). Measured on
``gemini-3.8-flash``, a 15-item extraction takes ~28s with it and does not finish inside
600s without it — the model spends the whole budget thinking before it emits anything.

A malformed or schema-breaking reply is retried once, since a second sample often lands.
An upstream rejection that a retry cannot fix — a retired model, a bad key — is raised
straight away, carrying the provider's own message so the caller can act on it.
"""

import httpx2

from app.classifier.base import ClassifierError
from app.config import Settings
from app.providers import check_key_matches_url
from app.schemas import ClassifierPayload, VocabularyItem
from app.taxonomy import CEFR_LEVELS, MainCategory


#: Measured against gemini-3.8-flash with LLM_REASONING_EFFORT=low: 1 item ~10s,
#: 15 items ~28s. Without that setting the same 15-item request did not finish in 600s,
#: so the ceiling here is a safety net, not the thing that makes long input work.
TIMEOUT_SECONDS = 120.0
MAX_ATTEMPTS = 2

#: Upstream error bodies are quoted back to the caller, so cap them. The key lives in a
#: header and never appears in a response body.
MAX_UPSTREAM_DETAIL = 300

#: 4xx that a second identical request could actually fix. Everything else in the 4xx
#: range — a retired model, a bad key, a malformed request — fails the same way twice.
RETRYABLE_STATUSES = frozenset({408, 429})


class FatalUpstreamError(ClassifierError):
    """An upstream rejection that a retry cannot fix, so it is raised immediately."""

SYSTEM_PROMPT = """You extract English vocabulary for Vietnamese learners and classify each word.
Return ONLY a JSON object with exactly this shape, no markdown fences, no commentary:
{"items": [{"word": "...", "part_of_speech": "...", "meaning_vi": "...", "cefr_level": "...", "main_category": "...", "sub_category": "...", "example": "...", "word_family": [{"word": "...", "part_of_speech": "..."}], "examples": ["...", "..."], "definition_en": "...", "usage_vi": "..."}], "count": N}

Rules:
- Pick up to 20 words worth learning: skip function words, duplicates, proper nouns.
- part_of_speech: short label such as noun, verb, adjective, adverb (max 30 chars).
- meaning_vi: concise Vietnamese meaning (max 300 chars).
- cefr_level: exactly one of A1 A2 B1 B2 C1 C2.
- main_category: exactly one of Business | Working & Career | Technical & IT | Daily Life & Communication | Learning & Personal Development | Finance | Health & Wellbeing | Society & Politics | Science & Environment | Other.
- sub_category: a short topic label (1-60 chars); use Uncategorized only if truly unplaceable.
- example: a real sentence from the input containing the word (max 300 chars). If the input is a single word or a fragment with no usable sentence, write a natural sentence yourself instead of echoing the input.
- word_family: up to 8 real related forms of the same root (noun, verb, adjective, adverb derivations and common inflections), each with its own part_of_speech. Never invent a form that does not exist; use [] when the word has no common derived forms.
- examples: 2 or 3 NEW sentences that you write yourself, showing the word in natural use. Each must contain the word, each must show a different context, and none may repeat the "example" sentence from the input.
- definition_en: explain what the word means in simple English (B1-level wording, max 400 chars). Do not define the word using the word itself.
- usage_vi: one Vietnamese paragraph (max 600 chars) covering BOTH how to use the word (common collocations, prepositions it takes, which form is usual) AND the context it belongs to (formal or informal, spoken or written, which field or situation).
- count must equal the number of items."""


class OpenAICompatibleClassifier:
    """Classifier backed by any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(self, settings: Settings) -> None:
        missing = [
            name
            for name, value in (
                ("LLM_BASE_URL", settings.llm_base_url),
                ("LLM_MODEL", settings.llm_model),
                ("LLM_API_KEY", settings.llm_api_key),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"Missing required configuration for 'openai_compatible': "
                f"{', '.join(missing)}. Set the corresponding environment variables."
            )
        assert settings.llm_base_url is not None
        assert settings.llm_model is not None
        assert settings.llm_api_key is not None
        # Startup, not first request: a key pasted over a stale LLM_BASE_URL is otherwise
        # only discovered as a 401 once someone calls the API.
        check_key_matches_url(settings.llm_api_key, settings.llm_base_url)
        self._base_url = settings.llm_base_url.rstrip("/")
        self._model = settings.llm_model
        self._api_key = settings.llm_api_key
        self._reasoning_effort = settings.llm_reasoning_effort

    def extract(self, text: str) -> list[VocabularyItem]:
        """Send the passage to the model and return the validated items."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        last_error: Exception | None = None
        for _ in range(MAX_ATTEMPTS):
            try:
                content = self._post(messages)
                return ClassifierPayload.model_validate_json(content).items
            except FatalUpstreamError:
                # A retired model or a rejected key fails identically on attempt two, so
                # raise now instead of spending another call and another wait on it.
                raise
            except Exception as exc:  # noqa: BLE001 - retried once, then raised
                last_error = exc
        raise ClassifierError(
            f"{self._model} failed after {MAX_ATTEMPTS} attempts: {last_error}"
        )

    def _request_body(self, messages: list[dict]) -> dict:
        """Build the request payload, adding ``reasoning_effort`` only when configured.

        It is omitted by default on purpose. Providers without a thinking mode reject
        unknown fields, so sending it unconditionally would break every non-reasoning
        model this one adapter is supposed to serve.
        """
        body: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        if self._reasoning_effort:
            body["reasoning_effort"] = self._reasoning_effort
        return body

    def _post(self, messages: list[dict]) -> str:
        """POST one chat completion and return the assistant message content."""
        with httpx2.Client(timeout=TIMEOUT_SECONDS) as client:
            response = client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=self._request_body(messages),
            )
            if response.status_code >= 400:
                raise self._upstream_error(response.status_code, response.text)
            return response.json()["choices"][0]["message"]["content"]

    def _upstream_error(self, status: int, body: str) -> ClassifierError:
        """Build the error the caller will read, keeping what the provider actually said.

        Providers give genuinely useful messages — Gemini names the model to switch to
        when one is retired. Discarding that in favour of a generic "request failed" is
        what turns a one-line fix into a debugging session.
        """
        detail = " ".join(body.split())[:MAX_UPSTREAM_DETAIL] or "(empty response body)"
        message = f"provider returned {status} for model {self._model}: {detail}"
        retryable = status in RETRYABLE_STATUSES or status >= 500
        return ClassifierError(message) if retryable else FatalUpstreamError(message)

    def __repr__(self) -> str:
        categories = ", ".join(c.value for c in MainCategory)
        return (
            f"OpenAICompatibleClassifier(model={self._model!r}, "
            f"base_url={self._base_url!r}, cefr={CEFR_LEVELS!r}, categories=[{categories}])"
        )

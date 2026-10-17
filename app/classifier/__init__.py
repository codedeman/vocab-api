"""Classifier providers and the single factory the API depends on.

This module is the swap point. Adding a real provider means one new file plus one line in
``PROVIDERS`` — ``main.py``, ``api/vocabulary.py`` and ``schemas.py`` stay untouched.
"""

from collections.abc import Callable

from app.classifier.base import Classifier, ClassifierError
from app.classifier.mock import MockClassifier
from app.classifier.openai_compatible import FatalUpstreamError, OpenAICompatibleClassifier
from app.config import Settings, load_settings

#: Provider name -> factory. "openai_compatible" covers Gemini, DeepSeek, OpenAI, Groq,
#: Together, OpenRouter, Ollama and vLLM — same wire format, different base_url. Anthropic
#: would need its own adapter because its wire format differs.
PROVIDERS: dict[str, Callable[[Settings], Classifier]] = {
    "mock": lambda settings: MockClassifier(),
    "openai_compatible": lambda settings: OpenAICompatibleClassifier(settings),
}


def get_classifier() -> Classifier:
    """Build the classifier named by ``VOCAB_CLASSIFIER``.

    Raises:
        ValueError: if nothing is configured, or if the configured name is not
            registered in this build. There is no default on purpose: falling back to
            the mock silently served fake data from a service that looked healthy.
    """
    settings = load_settings()
    if not settings.classifier:
        raise ValueError(
            "VOCAB_CLASSIFIER is not set, so no classifier was chosen. "
            "For Gemini set VOCAB_CLASSIFIER=openai_compatible plus LLM_BASE_URL, "
            "LLM_MODEL and LLM_API_KEY (see .env.example). "
            "For the offline stand-in set VOCAB_CLASSIFIER=mock explicitly."
        )
    factory = PROVIDERS.get(settings.classifier)
    if factory is None:
        raise ValueError(
            f"Unknown VOCAB_CLASSIFIER {settings.classifier!r}. "
            f"Available in this build: {', '.join(sorted(PROVIDERS))}"
        )
    return factory(settings)


__all__ = [
    "PROVIDERS",
    "Classifier",
    "ClassifierError",
    "FatalUpstreamError",
    "MockClassifier",
    "OpenAICompatibleClassifier",
    "get_classifier",
]

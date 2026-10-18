"""Runtime configuration, read from ``.env`` and the environment.

Deliberately provider-neutral: the same ``LLM_*`` variables point the service at Gemini,
DeepSeek, OpenAI or a local model, so switching provider never means editing code.

Neither ``VOCAB_CLASSIFIER`` nor ``VOCAB_ACCESS_KEY`` has a default; both must be set
explicitly and the server refuses to start otherwise. For the classifier, a silent
fallback to the mock is what made the service return fake data while looking healthy.
For the access key, a silent fallback to "open" would publish an endpoint backed by a
real LLM key to anyone who found the URL.

Nothing here needs changing to deploy: real environment variables win over the file, and
a missing ``.env`` is not an error, so a host that injects env vars works as-is.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Project root: config.py lives in app/, so one level up.
DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"

ENV_VARS = (
    "VOCAB_CLASSIFIER",
    "VOCAB_ACCESS_KEY",
    "DATABASE_URL",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_API_KEY",
    "LLM_REASONING_EFFORT",
)


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the environment-provided configuration."""

    #: ``None`` means nobody chose a provider; that is an error, not "use the mock".
    classifier: str | None = None
    #: The key callers must send as ``X-API-Key`` to use this service. Not the same thing
    #: as ``llm_api_key``, which is this service's own key for the LLM provider.
    #: repr=False for the same reason as llm_api_key: it is a secret.
    vocab_access_key: str | None = field(default=None, repr=False)
    #: Postgres connection string for the word cache. Unset means the cache is off, which
    #: only makes lookups slower. repr=False: the string carries the database password.
    database_url: str | None = field(default=None, repr=False)
    llm_base_url: str | None = None
    llm_model: str | None = None
    # repr=False so a logged or echoed Settings can never leak the key.
    llm_api_key: str | None = field(default=None, repr=False)
    #: Thinking budget for reasoning models, e.g. "low". Sent only when set, because
    #: providers without a thinking mode reject the field outright.
    llm_reasoning_effort: str | None = None


def _load_dotenv(path: Path | None = None) -> None:
    """Copy ``KEY=value`` lines from ``.env`` into the environment.

    Hand-written instead of adding ``python-dotenv``: we need a handful of flat keys.
    A variable already present in the environment wins, so ``VOCAB_CLASSIFIER=mock
    uvicorn ...`` still overrides the file. A missing file is not an error.

    ``path`` is resolved at call time, not import time, so tests can point
    ``DOTENV_PATH`` elsewhere and not read the developer's own ``.env``.
    """
    path = path or DOTENV_PATH
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip("\"'")
        if name and name not in os.environ:
            os.environ[name] = value


def _clean(name: str) -> str | None:
    """Read an env var, treating empty/whitespace-only values as unset."""
    value = os.environ.get(name, "").strip()
    return value or None


def load_settings() -> Settings:
    """Build Settings from ``.env`` plus the current environment."""
    _load_dotenv()
    return Settings(
        classifier=_clean("VOCAB_CLASSIFIER"),
        vocab_access_key=_clean("VOCAB_ACCESS_KEY"),
        database_url=_clean("DATABASE_URL"),
        llm_base_url=_clean("LLM_BASE_URL"),
        llm_model=_clean("LLM_MODEL"),
        llm_api_key=_clean("LLM_API_KEY"),
        llm_reasoning_effort=_clean("LLM_REASONING_EFFORT"),
    )

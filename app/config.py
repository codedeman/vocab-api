"""Runtime configuration, read from ``.env`` and the environment.

Deliberately provider-neutral: the same ``LLM_*`` variables point the service at Gemini,
DeepSeek, OpenAI or a local model, so switching provider never means editing code. For a
host with a recognisable key (``app/providers.py``), ``LLM_API_KEY`` alone is enough: the
endpoint, model and thinking budget are filled in from the key, and any of them can still
be set explicitly to override.

``VOCAB_ACCESS_KEY`` has no default, and ``VOCAB_CLASSIFIER`` has none either except
``openai_compatible`` when a recognised LLM key is present — never the mock. The server
refuses to start otherwise. For the classifier, a silent
fallback to the mock is what made the service return fake data while looking healthy.
For the access key, a silent fallback to "open" would publish an endpoint backed by a
real LLM key to anyone who found the URL.

Nothing here needs changing to deploy: real environment variables win over the file, and
a missing ``.env`` is not an error, so a host that injects env vars works as-is.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.providers import detect_provider, stale_host

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
    #: Which known host the key was recognised as, e.g. "openai". ``None`` for a key we
    #: cannot place, which is fine as long as LLM_BASE_URL and LLM_MODEL are set.
    llm_provider: str | None = None
    #: LLM_* variables that were set but ignored because they belong to another known
    #: host (``llm_stale_host``) than the key does. Names only — never values.
    llm_ignored: tuple[str, ...] = ()
    llm_stale_host: str | None = None


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
    classifier = _clean("VOCAB_CLASSIFIER")
    api_key = _clean("LLM_API_KEY")
    base_url = _clean("LLM_BASE_URL")
    model = _clean("LLM_MODEL")
    reasoning_effort = _clean("LLM_REASONING_EFFORT")
    provider = None
    ignored: tuple[str, ...] = ()
    old_host = None

    # The mock never talks to a host, so its key (if any) is not interpreted at all.
    if classifier in (None, "openai_compatible"):
        preset = detect_provider(api_key)
        if preset is not None:
            provider = preset.name
            # A recognised key is an explicit choice of a real provider, so it may stand
            # in for VOCAB_CLASSIFIER. It can never select the mock.
            classifier = classifier or "openai_compatible"
            old_host = stale_host(preset, base_url) if base_url else None
            if old_host is not None:
                # The key was changed and the previous provider's settings were not. The
                # key is the newest, most deliberate choice, so it wins; the endpoint,
                # model and thinking budget all belonged to the old host and go with it.
                ignored = tuple(
                    name
                    for name, value in (
                        ("LLM_BASE_URL", base_url),
                        ("LLM_MODEL", model),
                        ("LLM_REASONING_EFFORT", reasoning_effort),
                    )
                    if value is not None
                )
                base_url = model = reasoning_effort = None
            if base_url is None:
                base_url = preset.base_url
                # Only alongside the preset's own endpoint: a custom LLM_BASE_URL may be a
                # host that rejects the field, so it gets no thinking budget unasked.
                if reasoning_effort is None:
                    reasoning_effort = preset.reasoning_effort
                # Likewise the model: the preset's default only makes sense on its host.
                if model is None:
                    model = preset.model

    return Settings(
        classifier=classifier,
        vocab_access_key=_clean("VOCAB_ACCESS_KEY"),
        database_url=_clean("DATABASE_URL"),
        llm_base_url=base_url,
        llm_model=model,
        llm_api_key=api_key,
        llm_reasoning_effort=reasoning_effort,
        llm_provider=provider,
        llm_ignored=ignored,
        llm_stale_host=old_host,
    )

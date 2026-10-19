"""Known LLM hosts, recognised from the shape of the API key.

This is what makes "paste a different key" enough to switch provider. Every host here
issues keys with a recognisable prefix, so the key alone says where it belongs; the
endpoint, a sensible default model and the thinking budget follow from that.

``LLM_BASE_URL``, ``LLM_MODEL`` and ``LLM_REASONING_EFFORT`` still win when set — that is
how you reach DeepSeek, Ollama, a proxy, or a different model on the same host.
"""

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class ProviderPreset:
    """Everything needed to talk to one host, given only its key."""

    name: str
    base_url: str
    model: str
    #: Only Gemini's default model thinks. Everyone else rejects the field outright.
    reasoning_effort: str | None = None


GEMINI = ProviderPreset(
    name="gemini",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    model="gemini-3.8-flash",
    # Measured: a 15-item extraction does not finish inside 600s without this.
    reasoning_effort="low",
)
OPENROUTER = ProviderPreset(
    name="openrouter",
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
)
GROQ = ProviderPreset(
    name="groq",
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
)
OPENAI = ProviderPreset(
    name="openai",
    base_url="https://api.openai.com/v1",
    model="gpt-4o-mini",
)

#: Checked in order, so a longer prefix must come before any shorter one it starts with:
#: ``sk-or-`` and ``sk-ant-`` would otherwise be read as OpenAI keys.
KEY_PREFIXES: tuple[tuple[str, ProviderPreset | None], ...] = (
    ("AIza", GEMINI),
    # Google's newer key format; AI Studio issues both.
    ("AQ.", GEMINI),
    ("sk-or-", OPENROUTER),
    ("gsk_", GROQ),
    # Recognised only to refuse it clearly: Anthropic's wire format is not
    # OpenAI-compatible, so there is no adapter for it in this build.
    ("sk-ant-", None),
    # DeepSeek keys also start with sk-. OpenAI is the default reading; DeepSeek users
    # set LLM_BASE_URL, which the mismatch check below accepts for an sk- key.
    ("sk-", OPENAI),
)

#: Hostname -> provider name, for every host we can recognise in LLM_BASE_URL. DeepSeek
#: has no preset of its own (see above) but its host is still known.
KNOWN_HOSTS: dict[str, str] = {
    urlparse(preset.base_url).hostname or "": preset.name
    for preset in (GEMINI, OPENROUTER, GROQ, OPENAI)
} | {"api.deepseek.com": "deepseek"}

#: Hosts that accept a key from a provider other than their own name.
COMPATIBLE_HOSTS: dict[str, frozenset[str]] = {"openai": frozenset({"deepseek"})}


def detect_provider(api_key: str | None) -> ProviderPreset | None:
    """Return the preset this key belongs to, or ``None`` if the key is not recognised.

    Raises:
        ValueError: for an Anthropic key, which is recognisable but unusable here. Saying
            so beats letting it fall through to OpenAI and come back as a 401.
    """
    if not api_key:
        return None
    for prefix, preset in KEY_PREFIXES:
        if api_key.startswith(prefix):
            if preset is None:
                raise ValueError(
                    "LLM_API_KEY is an Anthropic key, but this build has no Anthropic "
                    "adapter (its wire format is not OpenAI-compatible). Use a Gemini, "
                    "OpenAI, OpenRouter, Groq or DeepSeek key instead."
                )
            return preset
    return None


def host_provider(base_url: str) -> str | None:
    """Name the provider behind ``base_url``, or ``None`` for a host we don't know."""
    return KNOWN_HOSTS.get(urlparse(base_url).hostname or "")


def check_key_matches_url(api_key: str, base_url: str) -> None:
    """Refuse a key that clearly belongs to a different known host than ``base_url``.

    This is the trap the presets alone cannot close: an ``LLM_BASE_URL`` left over from
    the previous provider overrides the preset, so a freshly pasted key gets sent to the
    wrong host and the failure only shows up as a 401 on the first request. An unknown
    host — Ollama, vLLM, a proxy — is never refused, since any key may be right there.

    Raises:
        ValueError: naming both providers and the fix.
    """
    try:
        preset = detect_provider(api_key)
    except ValueError:
        return  # detect_provider already reported it where the settings were loaded.
    host = host_provider(base_url)
    if preset is None or host is None or host == preset.name:
        return
    if host in COMPATIBLE_HOSTS.get(preset.name, frozenset()):
        return
    raise ValueError(
        f"LLM_API_KEY looks like a {preset.name} key, but LLM_BASE_URL points at {host} "
        f"({base_url}). Remove LLM_BASE_URL and LLM_MODEL to configure {preset.name} "
        f"from the key automatically, or set them to match the key."
    )

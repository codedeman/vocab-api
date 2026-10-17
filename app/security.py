"""Access control for the public endpoints: a shared secret in the ``X-API-Key`` header.

This exists because deploying the service publishes a URL backed by a real LLM key. With
no gate, anyone who finds the URL spends this service's provider quota, and on a free
provider tier their text is also fed to the provider's training.

It is deliberately the smallest thing that works: one shared secret, compared in constant
time, no users, no sessions, no dependency added. Pointless over plain HTTP — the header
travels in cleartext — so only call a deployment that terminates TLS.
"""

import hmac
import logging

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, load_settings

logger = logging.getLogger("vocab-api")

#: ``VOCAB_ACCESS_KEY=disabled`` turns the gate off. A word, not an empty value, so that
#: an unset variable and a deliberately open service can never look the same.
DISABLED = "disabled"

_UNSET_MESSAGE = (
    "VOCAB_ACCESS_KEY is not set. Set it to the secret callers must send as the "
    "X-API-Key header, or to 'disabled' to serve without any access control "
    "(only sensible for local development)."
)


def validate_access_control(settings: Settings) -> None:
    """Refuse to start when nobody decided whether the service is public.

    Called from the app's lifespan, mirroring ``get_classifier()``: an unset variable is
    an error at startup rather than a surprise at runtime. Defaulting to "open" would be
    the worse failure — it looks healthy while serving everyone.
    """
    if not settings.vocab_access_key:
        raise RuntimeError(_UNSET_MESSAGE)
    if settings.vocab_access_key == DISABLED:
        logger.warning(
            "vocab-api: VOCAB_ACCESS_KEY=disabled — every endpoint is open to anyone "
            "who can reach this URL, and each request spends the configured LLM key."
        )


def access_control_enabled(settings: Settings) -> bool:
    """Whether a caller needs the header. Used for the startup log line."""
    return bool(settings.vocab_access_key) and settings.vocab_access_key != DISABLED


def require_access_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(load_settings),
) -> None:
    """Reject a request whose ``X-API-Key`` does not match ``VOCAB_ACCESS_KEY``.

    ``hmac.compare_digest`` rather than ``==`` so the comparison does not return early on
    the first wrong byte. Both sides are encoded first: given ``str`` it rejects non-ASCII
    with a ``TypeError``, which would turn an odd header into a 500 instead of a 401.

    The 401 detail says only that the key is missing or wrong: it never names the expected
    value, and never distinguishes "no header" from "wrong header" in a way that helps
    someone guess.
    """
    expected = settings.vocab_access_key
    if not expected:
        # Unreachable through the app, which refuses to start in this state. Guard anyway
        # so importing this dependency somewhere else can never mean "allow everything".
        raise RuntimeError(_UNSET_MESSAGE)
    if expected == DISABLED:
        return
    if x_api_key is None or not hmac.compare_digest(
        x_api_key.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
        )

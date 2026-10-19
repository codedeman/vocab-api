"""FastAPI application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import vocabulary
from app.cache import get_cache
from app.classifier import ClassifierError, get_classifier
from app.config import load_settings
from app.security import access_control_enabled, require_access_key, validate_access_control

logger = logging.getLogger("vocab-api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Check the configuration once at startup so misconfiguration fails here, not later.

    Without this the server starts happily and only breaks on the first request — or
    worse, quietly answers with mock data, or quietly serves everyone.
    """
    # uvicorn configures only its own loggers, leaving the root logger handler-less, so
    # without this the line below is silently discarded. A no-op if handlers exist.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
    settings = load_settings()
    get_classifier()
    validate_access_control(settings)
    cache = get_cache(settings)
    if cache is not None:
        # A DATABASE_URL that was set but cannot be reached is a misconfiguration: fail
        # here rather than discover it as a stream of per-request warnings.
        cache.init_schema()
    # No secret here, ever (neither key, nor the database URL): this line ends up in logs
    # and terminals. access_control and cache report on/OFF only — enough to spot an
    # accidentally open deployment, and nothing that helps anyone reading the logs get in.
    logger.info(
        "vocab-api: classifier=%s provider=%s model=%s base_url=%s access_control=%s cache=%s",
        settings.classifier,
        settings.llm_provider or "-",
        settings.llm_model or "-",
        settings.llm_base_url or "-",
        "on" if access_control_enabled(settings) else "OFF",
        "on" if cache is not None else "OFF",
    )
    yield


app = FastAPI(
    title="Vocabulary Classification API",
    version="0.2.0",
    description=(
        "Extracts English vocabulary and classifies each item against a fixed topic "
        "taxonomy. The provider is chosen by VOCAB_CLASSIFIER and reported in every "
        "response, so mock data can never be mistaken for real output."
    ),
    lifespan=lifespan,
)

# The gate goes on the router, not on the whole app: /health below must stay reachable
# without a key, because the deployment platform calls it as a health check and because it
# touches no provider. It reveals only the configured provider name.
app.include_router(
    vocabulary.router, prefix="/api/v1", dependencies=[Depends(require_access_key)]
)


@app.exception_handler(ClassifierError)
async def classifier_error_handler(request: Request, exc: ClassifierError) -> JSONResponse:
    """Report a provider failure as 502 with the provider's own words.

    Registered app-wide rather than caught in the route, so every future provider gets
    it for free. 502 and not 500: the upstream rejected us, this service is fine.
    """
    logger.warning("vocab-api: classifier failed: %s", exc)
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Liveness probe, including which provider is configured."""
    return {"status": "ok", "classifier": load_settings().classifier or "unset"}

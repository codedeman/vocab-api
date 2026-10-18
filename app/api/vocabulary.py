"""Vocabulary extraction endpoint."""

import logging
from typing import Literal

from fastapi import APIRouter, Depends

from app.cache import WordCache, get_cache, single_word
from app.classifier import Classifier, get_classifier
from app.config import Settings, load_settings
from app.schemas import ExtractRequest, ExtractResponse

logger = logging.getLogger("vocab-api")

router = APIRouter(prefix="/vocabulary", tags=["vocabulary"])


@router.post("/extract", response_model=ExtractResponse)
def extract_vocabulary(
    payload: ExtractRequest,
    classifier: Classifier = Depends(get_classifier),
    settings: Settings = Depends(load_settings),
    cache: WordCache | None = Depends(get_cache),
) -> ExtractResponse:
    """Extract vocabulary from an English passage and classify each item.

    A single word is answered from the cache when it is there, with no LLM call. A
    passage always goes to the LLM, since it picks the words, but every word it returns
    is stored so a later lookup of that word hits.
    """
    # get_classifier() already refused an unset/unknown provider, so this is a real name.
    provider = settings.classifier or "unknown"
    word = single_word(payload.text)
    status: Literal["hit", "miss", "off", "error"] = "off" if cache is None else "miss"

    if cache is not None and word:
        try:
            cached = cache.get(word)
        except Exception:  # noqa: BLE001 - a broken cache must not break the API
            logger.warning("vocab-api: cache read failed", exc_info=True)
            status = "error"
        else:
            if cached is not None:
                return ExtractResponse(
                    items=[cached], count=1, classifier=provider, cache="hit"
                )

    items = classifier.extract(payload.text)

    if cache is not None and status != "error":
        try:
            for item in items:
                keys = [item.word.strip().lower()]
                if word and len(items) == 1:
                    keys.append(word)
                cache.put(keys, item, settings.llm_model or provider)
        except Exception:  # noqa: BLE001
            logger.warning("vocab-api: cache write failed", exc_info=True)
            status = "error"

    return ExtractResponse(items=items, count=len(items), classifier=provider, cache=status)

"""Vocabulary extraction endpoint."""

from fastapi import APIRouter, Depends

from app.classifier import Classifier, get_classifier
from app.config import Settings, load_settings
from app.schemas import ExtractRequest, ExtractResponse

router = APIRouter(prefix="/vocabulary", tags=["vocabulary"])


@router.post("/extract", response_model=ExtractResponse)
def extract_vocabulary(
    payload: ExtractRequest,
    classifier: Classifier = Depends(get_classifier),
    settings: Settings = Depends(load_settings),
) -> ExtractResponse:
    """Extract vocabulary from an English passage and classify each item."""
    items = classifier.extract(payload.text)
    # get_classifier() already refused an unset/unknown provider, so this is a real name.
    return ExtractResponse(
        items=items, count=len(items), classifier=settings.classifier or "unknown"
    )

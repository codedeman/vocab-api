"""Request and response models.

``part_of_speech``, ``cefr_level``, ``meaning_vi``, ``main_category`` and ``sub_category``
are separate fields on purpose: they are independent classification dimensions, and the
Chrome extension filters on each one separately.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.taxonomy import MainCategory

CefrLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2"]


class ExtractRequest(BaseModel):
    """An English passage to extract vocabulary from."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="English text to extract vocabulary from.",
    )

    @field_validator("text")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must contain at least one non-whitespace character")
        return stripped


class WordFamilyMember(BaseModel):
    """One related form of the same root, with its own part of speech."""

    word: str = Field(..., min_length=1, max_length=80)
    part_of_speech: str = Field(..., min_length=1, max_length=30)


class VocabularyItem(BaseModel):
    """One classified vocabulary entry."""

    word: str = Field(..., min_length=1, max_length=80)
    part_of_speech: str = Field(..., min_length=1, max_length=30)
    meaning_vi: str = Field(..., min_length=1, max_length=300)
    cefr_level: CefrLevel
    main_category: MainCategory
    sub_category: str = Field(..., min_length=1, max_length=60)
    #: A real sentence taken from the submitted text.
    example: str = Field(..., min_length=1, max_length=300)
    #: Derived forms. Empty is valid data: plenty of words have no common derivations.
    word_family: list[WordFamilyMember] = Field(default_factory=list, max_length=8)
    #: Generated sentences, distinct from ``example``. Capped so a provider cannot flood it.
    examples: list[str] = Field(default_factory=list, max_length=5)
    #: Required: every word can be explained, so an absent value is a provider failure.
    definition_en: str = Field(..., min_length=1, max_length=400)
    usage_vi: str = Field(..., min_length=1, max_length=600)

    @field_validator("examples")
    @classmethod
    def _drop_blank_examples(cls, value: list[str]) -> list[str]:
        """Strip each sentence and drop empties, so ``["", " "]`` cannot slip through."""
        return [stripped for sentence in value if (stripped := sentence.strip())]


class ClassifierPayload(BaseModel):
    """Exactly what a provider must return — the LLM's output schema.

    Separate from ``ExtractResponse`` because ``classifier`` is the service's own
    knowledge: asking the model to report which provider it is would be absurd.
    """

    items: list[VocabularyItem]
    count: int = Field(..., ge=0, description="Number of items returned.")


class ExtractResponse(ClassifierPayload):
    """The extraction result as the API returns it."""

    classifier: str = Field(
        ...,
        min_length=1,
        description="Provider that produced these items, e.g. 'openai_compatible' or 'mock'.",
    )
    cache: Literal["hit", "miss", "off", "error"] = Field(
        ...,
        description=(
            "Where the items came from: 'hit' served from the word cache with no LLM call, "
            "'miss' freshly generated and stored, 'off' no cache configured, 'error' the "
            "cache failed and the LLM answered instead."
        ),
    )

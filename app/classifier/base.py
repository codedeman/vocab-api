"""The one contract the API layer depends on."""

from typing import Protocol

from app.schemas import VocabularyItem


class ClassifierError(RuntimeError):
    """A provider failed, and the caller deserves to know why.

    Part of the same contract as ``Classifier``: raise this instead of a bare
    ``RuntimeError`` and ``main.py`` turns it into a 502 carrying the message, rather
    than a 500 whose only content is the words "Internal Server Error".
    """


class Classifier(Protocol):
    """Turns English text into classified vocabulary items.

    A real provider implementation satisfies this same signature, so adding one touches
    nothing outside ``app/classifier/``.
    """

    def extract(self, text: str) -> list[VocabularyItem]:
        """Return the vocabulary items found in ``text``."""
        ...

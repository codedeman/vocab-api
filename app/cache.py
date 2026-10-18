"""Word cache in Postgres, so a word looked up once is not generated again.

Every call to the LLM costs ~10s and provider quota; a cached word costs one query. This
module is the only place that touches the database driver.

Failure policy differs by when it happens. A bad ``DATABASE_URL`` is a configuration
mistake, so ``init_schema`` raises at startup and the server refuses to run. A database
that goes away *later* is not allowed to take the API down with it: the route catches the
error, answers from the LLM, and reports ``cache: "error"``.
"""

import re

import psycopg
from fastapi import Depends
from psycopg.types.json import Jsonb

from app.config import Settings, load_settings
from app.schemas import VocabularyItem

#: A cache lookup must fail fast: waiting on a dead database would be slower than
#: just asking the LLM.
CONNECT_TIMEOUT_SECONDS = 5

#: One token of letters, with internal apostrophes or hyphens ("don't", "well-known").
_SINGLE_WORD = re.compile(r"[a-z]+(?:['-][a-z]+)*")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vocab_cache (
    word       text PRIMARY KEY,
    item       jsonb NOT NULL,
    model      text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
)
"""


def single_word(text: str) -> str | None:
    """Return the lowercased word if ``text`` is exactly one word, else ``None``.

    Only a single word can be answered from the cache: for a passage the LLM chooses
    which words to teach, so there is no way to know them before asking.
    """
    candidate = text.strip().lower()
    return candidate if _SINGLE_WORD.fullmatch(candidate) else None


class WordCache:
    """Read and write cached vocabulary items, keyed by lowercase word."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._database_url, connect_timeout=CONNECT_TIMEOUT_SECONDS)

    def init_schema(self) -> None:
        """Create the table if missing. Raises if the database cannot be reached."""
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def get(self, word: str) -> VocabularyItem | None:
        """Return the cached item for ``word``, or ``None`` on a miss."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT item FROM vocab_cache WHERE word = %s", (word,)
            ).fetchone()
        # Re-validated on the way out, so a row written under an older schema fails
        # loudly here instead of reaching a client in the wrong shape.
        return VocabularyItem.model_validate(row[0]) if row else None

    def put(self, words: list[str], item: VocabularyItem, model: str) -> None:
        """Store ``item`` under every key in ``words``, leaving existing rows alone.

        Several keys, one item: a user typing "postponed" gets back the headword
        "postpone", and both spellings should hit next time.
        """
        payload = Jsonb(item.model_dump(mode="json"))
        with self._connect() as conn:
            for word in dict.fromkeys(words):
                conn.execute(
                    "INSERT INTO vocab_cache (word, item, model) VALUES (%s, %s, %s) "
                    "ON CONFLICT (word) DO NOTHING",
                    (word, payload, model),
                )


def get_cache(settings: Settings = Depends(load_settings)) -> WordCache | None:
    """The cache, or ``None`` when it is off.

    Off when ``DATABASE_URL`` is unset, and always off for the mock classifier: its
    output is fake, and storing it would hand `(mock)` text to real lookups later.
    """
    if not settings.database_url or settings.classifier == "mock":
        return None
    return WordCache(settings.database_url)

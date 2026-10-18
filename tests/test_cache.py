"""Tests for the word cache, using an in-memory stand-in for the Postgres one.

The SQL in ``WordCache`` itself is not exercised here; see the last test.
"""

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.cache import WordCache, get_cache, single_word
from app.classifier import get_classifier
from app.config import Settings
from app.main import app
from app.schemas import VocabularyItem

URL = "/api/v1/vocabulary/extract"


def _item(word: str) -> VocabularyItem:
    return VocabularyItem(
        word=word, part_of_speech="verb", meaning_vi="hoãn lại", cefr_level="B1",
        main_category="Business", sub_category="Scheduling", example=f"They {word} it.",
        word_family=[], examples=[f"Do not {word} it."], definition_en="To delay.",
        usage_vi="Dùng với until.",
    )


class FakeCache:
    """Same interface as WordCache, backed by a dict."""

    def __init__(self) -> None:
        self.rows: dict[str, VocabularyItem] = {}
        self.fail = False

    def get(self, word: str) -> VocabularyItem | None:
        if self.fail:
            raise RuntimeError("db down")
        return self.rows.get(word)

    def put(self, words: list[str], item: VocabularyItem, model: str) -> None:
        if self.fail:
            raise RuntimeError("db down")
        for word in words:
            self.rows.setdefault(word, item)


class CountingClassifier:
    def __init__(self, words: list[str]) -> None:
        self.words = words
        self.calls = 0

    def extract(self, text: str) -> list[VocabularyItem]:
        self.calls += 1
        return [_item(w) for w in self.words]


@pytest.fixture()
def fake() -> FakeCache:
    return FakeCache()


def _client(cache: FakeCache | None, classifier: CountingClassifier) -> Iterator[TestClient]:
    app.dependency_overrides[get_cache] = lambda: cache
    app.dependency_overrides[get_classifier] = lambda: classifier
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def llm() -> CountingClassifier:
    return CountingClassifier(["postpone"])


@pytest.fixture()
def client(fake: FakeCache, llm: CountingClassifier) -> Iterator[TestClient]:
    yield from _client(fake, llm)


def test_single_word_miss_calls_llm_once_and_stores(client, fake, llm) -> None:
    body = client.post(URL, json={"text": "postpone"}).json()

    assert body["cache"] == "miss"
    assert llm.calls == 1
    assert "postpone" in fake.rows


def test_single_word_hit_does_not_call_the_llm(client, fake, llm) -> None:
    client.post(URL, json={"text": "postpone"})
    llm.calls = 0

    body = client.post(URL, json={"text": "postpone"}).json()

    assert body["cache"] == "hit"
    assert llm.calls == 0, "a cache hit must not spend an LLM call"
    assert body["items"][0]["word"] == "postpone"
    assert body["count"] == 1


def test_lookup_ignores_case_and_surrounding_space(client, llm) -> None:
    client.post(URL, json={"text": "postpone"})
    llm.calls = 0

    body = client.post(URL, json={"text": "  Postpone "}).json()

    assert body["cache"] == "hit"
    assert llm.calls == 0


def test_inflected_form_is_cached_under_what_the_user_typed(client, fake, llm) -> None:
    """Typing 'postponed' returns headword 'postpone'; both spellings must hit next time."""
    client.post(URL, json={"text": "postponed"})

    assert {"postponed", "postpone"} <= set(fake.rows)
    llm.calls = 0
    assert client.post(URL, json={"text": "postponed"}).json()["cache"] == "hit"
    assert llm.calls == 0


def test_passage_always_calls_llm_but_stores_every_word(fake) -> None:
    llm = CountingClassifier(["postpone", "negotiate"])
    for client in _client(fake, llm):
        text = {"text": "They postpone the meeting to negotiate."}
        first = client.post(URL, json=text).json()
        second = client.post(URL, json=text).json()

    assert first["cache"] == second["cache"] == "miss"
    assert llm.calls == 2, "a passage cannot be answered from the cache"
    assert {"postpone", "negotiate"} <= set(fake.rows)


def test_word_from_a_passage_hits_on_a_later_single_lookup(fake) -> None:
    llm = CountingClassifier(["postpone", "negotiate"])
    for client in _client(fake, llm):
        client.post(URL, json={"text": "They postpone the meeting to negotiate."})
        llm.calls = 0
        body = client.post(URL, json={"text": "negotiate"}).json()

    assert body["cache"] == "hit"
    assert llm.calls == 0


def test_cache_read_failure_falls_back_to_the_llm(client, fake, llm) -> None:
    fake.fail = True

    response = client.post(URL, json={"text": "postpone"})

    assert response.status_code == 200
    assert response.json()["cache"] == "error"
    assert llm.calls == 1


def test_cache_write_failure_still_returns_the_answer(fake, llm) -> None:
    class WriteFails(FakeCache):
        def put(self, words, item, model) -> None:
            raise RuntimeError("db down")

    for client in _client(WriteFails(), llm):
        response = client.post(URL, json={"text": "postpone"})

    assert response.status_code == 200
    assert response.json()["cache"] == "error"


def test_no_cache_is_reported_as_off(llm) -> None:
    for client in _client(None, llm):
        body = client.post(URL, json={"text": "postpone"}).json()

    assert body["cache"] == "off"


@pytest.mark.parametrize("text", ["two words", "end.", "a1", "", "well known foo", "x\ny"])
def test_only_a_bare_word_is_cacheable(text: str) -> None:
    assert single_word(text) is None


@pytest.mark.parametrize("text, key", [("Postpone", "postpone"), (" don't ", "don't"), ("well-known", "well-known")])
def test_single_word_is_normalised(text: str, key: str) -> None:
    assert single_word(text) == key


def test_mock_classifier_never_uses_the_cache() -> None:
    """Mock output is fake; caching it would later serve '(mock)' text as real data."""
    settings = Settings(classifier="mock", database_url="postgresql://u:p@h/d")

    assert get_cache(settings) is None


def test_unset_database_url_means_no_cache() -> None:
    assert get_cache(Settings(classifier="openai_compatible")) is None


def test_database_url_is_not_exposed_in_repr() -> None:
    settings = Settings(classifier="mock", database_url="postgresql://u:hunter2@h/d")

    assert "hunter2" not in repr(settings)


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="needs TEST_DATABASE_URL")
def test_postgres_roundtrip() -> None:
    cache = WordCache(os.environ["TEST_DATABASE_URL"])
    cache.init_schema()
    word = "zz-cache-roundtrip-test"

    cache.put([word], _item("postpone"), "test-model")

    assert cache.get(word) == _item("postpone")
    assert cache.get("zz-not-there") is None

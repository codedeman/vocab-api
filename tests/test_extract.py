"""The extraction endpoint and the mock classifier's guarantees."""

import pytest
from fastapi.testclient import TestClient

from app.classifier.mock import MAX_ITEMS, MockClassifier
from app.taxonomy import CEFR_LEVELS, MainCategory

EXTRACT_URL = "/api/v1/vocabulary/extract"

PARAGRAPH = (
    "The accountant reviewed every invoice before the team could deploy the new billing "
    "service. Our colleague mentioned that inflation affected the quarterly budget."
)

STRING_FIELDS = (
    "word",
    "part_of_speech",
    "meaning_vi",
    "cefr_level",
    "main_category",
    "sub_category",
    "example",
    "definition_en",
    "usage_vi",
)

LIST_FIELDS = ("word_family", "examples")

REQUIRED_FIELDS = STRING_FIELDS + LIST_FIELDS


def post_text(client: TestClient, text: str):
    return client.post(EXTRACT_URL, json={"text": text})


def test_extract_returns_items_with_every_field(client: TestClient) -> None:
    response = post_text(client, PARAGRAPH)

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(body["items"])
    assert body["items"], "expected at least one vocabulary item"

    for item in body["items"]:
        assert set(item) == set(REQUIRED_FIELDS)
        for field in STRING_FIELDS:
            assert isinstance(item[field], str) and item[field].strip()
        for field in LIST_FIELDS:
            assert isinstance(item[field], list)


def test_word_family_members_are_well_formed(client: TestClient) -> None:
    items = post_text(client, PARAGRAPH).json()["items"]

    assert items
    for item in items:
        assert len(item["word_family"]) <= 8
        for member in item["word_family"]:
            assert set(member) == {"word", "part_of_speech"}
            assert member["word"].strip() and member["part_of_speech"].strip()
            assert member["word"] != item["word"], "a family member must not repeat the word"


def test_generated_examples_contain_their_word(client: TestClient) -> None:
    """``examples`` carries the same contains-the-word guarantee as ``example``."""
    items = post_text(client, PARAGRAPH).json()["items"]

    assert items
    for item in items:
        assert 1 <= len(item["examples"]) <= 5
        for sentence in item["examples"]:
            assert sentence == sentence.strip() and sentence
            assert item["word"] in sentence.lower()


def test_every_item_has_a_definition_and_usage_note(client: TestClient) -> None:
    """Including fallback words — this is what proves the mock fills in all new fields."""
    items = post_text(client, "The zzqqvlx shipment arrived carefully.").json()["items"]

    assert items
    for item in items:
        assert item["definition_en"].strip()
        assert item["usage_vi"].strip()


def test_enriched_lexicon_word_has_real_learner_content(client: TestClient) -> None:
    items = post_text(client, "We deploy the service every Friday.").json()["items"]

    deploy = next(item for item in items if item["word"] == "deploy")
    assert {"word": "deployment", "part_of_speech": "noun"} in deploy["word_family"]
    assert "(mock)" not in deploy["definition_en"]
    assert "(mock)" not in deploy["usage_vi"]


def test_fallback_word_has_mock_prefixed_learner_content(client: TestClient) -> None:
    items = post_text(client, "The zzqqvlx appeared overnight.").json()["items"]

    unknown = next(item for item in items if item["word"] == "zzqqvlx")
    assert unknown["definition_en"].startswith("(mock)")
    assert unknown["usage_vi"].startswith("(mock)")
    # Plural of a word ending in -x takes -es, so the inflection rules are doing real work.
    assert unknown["word_family"] == [{"word": "zzqqvlxes", "part_of_speech": "noun"}]


def test_categories_and_levels_stay_inside_the_taxonomy(client: TestClient) -> None:
    allowed_categories = {category.value for category in MainCategory}

    items = post_text(client, PARAGRAPH).json()["items"]

    assert items
    for item in items:
        assert item["main_category"] in allowed_categories
        assert item["cefr_level"] in CEFR_LEVELS


def test_lexicon_word_gets_its_pinned_category(client: TestClient) -> None:
    items = post_text(client, "She filed the invoice yesterday.").json()["items"]

    invoice = next(item for item in items if item["word"] == "invoice")
    assert invoice["main_category"] == MainCategory.FINANCE.value
    assert invoice["sub_category"] == "Banking & Payments"
    assert invoice["meaning_vi"] == "hóa đơn"


def test_unknown_word_falls_back_to_other(client: TestClient) -> None:
    items = post_text(client, "The zzqqvlx appeared overnight.").json()["items"]

    unknown = next(item for item in items if item["word"] == "zzqqvlx")
    assert unknown["main_category"] == MainCategory.OTHER.value
    assert unknown["sub_category"] == "Uncategorized"
    assert unknown["meaning_vi"].startswith("(mock)")


def test_repeated_word_is_returned_once(client: TestClient) -> None:
    items = post_text(client, "Budget talks. The budget grew. Budget again.").json()["items"]

    words = [item["word"] for item in items]
    assert words.count("budget") == 1


def test_example_contains_its_own_word(client: TestClient) -> None:
    items = post_text(client, PARAGRAPH).json()["items"]

    assert items
    for item in items:
        assert item["word"] in item["example"].lower()


@pytest.mark.parametrize(
    "payload",
    [{}, {"text": ""}, {"text": "   "}, {"text": "\n\t "}],
    ids=["missing", "empty", "spaces", "whitespace"],
)
def test_invalid_text_is_rejected(client: TestClient, payload: dict) -> None:
    assert client.post(EXTRACT_URL, json=payload).status_code == 422


def test_text_over_the_length_limit_is_rejected(client: TestClient) -> None:
    assert post_text(client, "word " * 2000).status_code == 422


def test_identical_requests_produce_identical_responses(client: TestClient) -> None:
    first = post_text(client, PARAGRAPH).json()
    second = post_text(client, PARAGRAPH).json()

    assert first == second


@pytest.mark.parametrize(
    ("word", "expected_pos"),
    [
        ("reviewed", "verb"),
        ("shipping", "verb"),
        ("carefully", "adverb"),
        ("dangerous", "adjective"),
        ("shipment", "noun"),
    ],
)
def test_fallback_part_of_speech_follows_suffix_rules(word: str, expected_pos: str) -> None:
    items = MockClassifier().extract(f"They {word} it.")

    item = next(entry for entry in items if entry.word == word)
    assert item.part_of_speech == expected_pos


@pytest.mark.parametrize(
    "word",
    ["checks", "glasses", "shipping", "reviewed", "carefully"],
    ids=["plural-s", "already-plural", "participle", "past", "adverb"],
)
def test_ambiguous_forms_get_no_invented_word_family(word: str) -> None:
    """An empty family beats a made-up one: "checkses" is not a word."""
    items = MockClassifier().extract(f"They {word} it.")

    item = next(entry for entry in items if entry.word == word)
    assert item.word_family == []


def test_function_words_are_not_returned_as_vocabulary(client: TestClient) -> None:
    items = post_text(client, "They checked every invoice because another was missing.").json()["items"]

    words = {item["word"] for item in items}
    assert "invoice" in words
    assert words.isdisjoint({"every", "another", "because", "they", "was"})


def test_item_count_is_capped(client: TestClient) -> None:
    words = [f"zeta{a}{b}" for a in "abcde" for b in "abcdefgh"]
    assert len(words) > MAX_ITEMS

    body = post_text(client, " ".join(words) + ".").json()

    assert body["count"] == MAX_ITEMS
    assert len(body["items"]) == MAX_ITEMS

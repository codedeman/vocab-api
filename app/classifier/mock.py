"""A deterministic, offline stand-in for the real classifier.

No network, no randomness: the same text always produces the same items, which is what makes
the API testable before a model is wired up. Words in the seed lexicon get real values;
anything else falls back to suffix and length heuristics and the ``Other`` category, with
``meaning_vi`` prefixed ``(mock)`` so mock output is never mistaken for a real gloss.

The learner fields follow the same rule: ``word_family`` comes from conservative inflection
rules, while ``examples``, ``definition_en`` and ``usage_vi`` are ``(mock)``-prefixed
templates unless a lexicon entry supplies real content.
"""

import re
from typing import NamedTuple

from app.schemas import VocabularyItem, WordFamilyMember
from app.taxonomy import UNCATEGORIZED, MainCategory

#: Hard ceiling on items per request, so a long passage can't return hundreds of rows.
MAX_ITEMS = 20

#: Example sentences are excerpted to this length.
EXAMPLE_MAX_CHARS = 200

#: Tokens longer than this are skipped as noise rather than treated as vocabulary.
MAX_WORD_LENGTH = 45

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Function words carry no vocabulary value for a learner.
STOPWORDS = frozenset(
    """
    the and but for nor yet with from into onto upon that this these those they them their
    there here then than when while where which who whom whose what why how all any both each
    few more most other some such only own same too very can will just don should now have
    has had been being was were are its his her hers ours yours theirs you your she him not
    about above after again against because before below between during further once out
    over under until our also would could shall may might must did does doing
    every another either neither several enough myself yourself himself herself itself
    ourselves themselves
    """.split()
)

class LexEntry(NamedTuple):
    """One seed-lexicon row.

    The last three fields default to empty on purpose: an entry without hand-written learner
    content falls back to the same ``(mock)`` text as a word outside the lexicon, so enriching
    an entry is opt-in rather than 30 paragraphs of invented filler.
    """

    part_of_speech: str
    meaning_vi: str
    cefr_level: str
    main_category: MainCategory
    sub_category: str
    word_family: tuple[tuple[str, str], ...] = ()
    definition_en: str = ""
    usage_vi: str = ""


LEXICON: dict[str, LexEntry] = {
    # Business
    "revenue": LexEntry(
        "noun",
        "doanh thu",
        "B2",
        MainCategory.BUSINESS,
        "Sales & Revenue",
        word_family=(("revenues", "noun"),),
        definition_en=(
            "The total amount of money a company receives from selling its goods or "
            "services, before any costs are subtracted."
        ),
        usage_vi=(
            "Thường đi với generate/increase/report revenue và annual revenue. Là danh từ "
            "không đếm được khi nói chung, dùng số nhiều 'revenues' khi tách theo từng "
            "nguồn thu. Văn phong trang trọng, hay gặp trong báo cáo tài chính và báo chí "
            "kinh tế; trong hội thoại thường ngày người ta hay nói 'sales' hoặc 'income'."
        ),
    ),
    "stakeholder": LexEntry("noun", "bên liên quan", "C1", MainCategory.BUSINESS, "Corporate Strategy"),
    "negotiate": LexEntry("verb", "đàm phán", "B2", MainCategory.BUSINESS, "Negotiation"),
    # Working & Career
    "promotion": LexEntry("noun", "sự thăng chức", "B1", MainCategory.WORKING_CAREER, "Career Progression"),
    "colleague": LexEntry(
        "noun",
        "đồng nghiệp",
        "B1",
        MainCategory.WORKING_CAREER,
        "Workplace Relationships",
        word_family=(("colleagues", "noun"),),
        definition_en="A person you work with, especially someone in a similar position to you.",
        usage_vi=(
            "Dùng với giới từ 'at' cho nơi làm việc (a colleague at the agency) và 'from' "
            "cho bộ phận (a colleague from marketing). Trung tính và hơi trang trọng, phù "
            "hợp email công việc; văn nói Mỹ thường dùng 'coworker' thay thế."
        ),
    ),
    "recruiter": LexEntry("noun", "nhà tuyển dụng", "B2", MainCategory.WORKING_CAREER, "Job Applications"),
    # Technical & IT
    "deploy": LexEntry(
        "verb",
        "triển khai",
        "B2",
        MainCategory.TECHNICAL_IT,
        "Software Development",
        word_family=(
            ("deployment", "noun"),
            ("deploys", "verb"),
            ("deployed", "verb"),
            ("deploying", "verb"),
            ("redeploy", "verb"),
        ),
        definition_en=(
            "To put software onto the servers where people actually use it, making a new "
            "version live."
        ),
        usage_vi=(
            "Là ngoại động từ, đi với tân ngữ là thứ được đưa lên (deploy the service) và "
            "giới từ 'to' cho môi trường đích (deploy to production). Danh từ tương ứng là "
            "'deployment'. Rất phổ biến trong ngành phần mềm; ngoài IT từ này còn mang "
            "nghĩa điều động quân đội hoặc nhân lực."
        ),
    ),
    "database": LexEntry("noun", "cơ sở dữ liệu", "B1", MainCategory.TECHNICAL_IT, "Data & Storage"),
    "encryption": LexEntry("noun", "sự mã hóa", "C1", MainCategory.TECHNICAL_IT, "Security"),
    "server": LexEntry("noun", "máy chủ", "B1", MainCategory.TECHNICAL_IT, "Infrastructure"),
    # Daily Life & Communication
    "neighbor": LexEntry("noun", "người láng giềng", "A2", MainCategory.DAILY_LIFE, "Community & Neighbours"),
    "apologize": LexEntry("verb", "xin lỗi", "B1", MainCategory.DAILY_LIFE, "Social Interaction"),
    "groceries": LexEntry("noun", "hàng thực phẩm", "A2", MainCategory.DAILY_LIFE, "Shopping & Errands"),
    # Learning & Personal Development
    "curriculum": LexEntry("noun", "chương trình học", "B2", MainCategory.LEARNING, "Education & Courses"),
    "motivation": LexEntry("noun", "động lực", "B2", MainCategory.LEARNING, "Habits & Motivation"),
    "fluency": LexEntry("noun", "sự lưu loát", "B2", MainCategory.LEARNING, "Language Learning"),
    # Finance
    "invoice": LexEntry("noun", "hóa đơn", "B2", MainCategory.FINANCE, "Banking & Payments"),
    "mortgage": LexEntry("noun", "khoản vay thế chấp", "C1", MainCategory.FINANCE, "Loans & Credit"),
    "inflation": LexEntry("noun", "sự lạm phát", "B2", MainCategory.FINANCE, "Economy & Prices"),
    "budget": LexEntry("noun", "ngân sách", "B1", MainCategory.FINANCE, "Budgeting"),
    # Health & Wellbeing
    "symptom": LexEntry("noun", "triệu chứng", "B2", MainCategory.HEALTH, "Illness & Symptoms"),
    "nutrition": LexEntry("noun", "dinh dưỡng", "B2", MainCategory.HEALTH, "Nutrition & Diet"),
    "insomnia": LexEntry("noun", "chứng mất ngủ", "C1", MainCategory.HEALTH, "Sleep & Rest"),
    # Society & Politics
    "election": LexEntry("noun", "cuộc bầu cử", "B1", MainCategory.SOCIETY_POLITICS, "Elections & Voting"),
    "legislation": LexEntry("noun", "pháp luật", "C1", MainCategory.SOCIETY_POLITICS, "Law & Policy"),
    "inequality": LexEntry("noun", "sự bất bình đẳng", "C1", MainCategory.SOCIETY_POLITICS, "Social Issues"),
    # Science & Environment
    "emissions": LexEntry("noun", "khí thải", "B2", MainCategory.SCIENCE_ENVIRONMENT, "Climate & Pollution"),
    "ecosystem": LexEntry("noun", "hệ sinh thái", "B2", MainCategory.SCIENCE_ENVIRONMENT, "Ecology & Biodiversity"),
    "hypothesis": LexEntry("noun", "giả thuyết", "C1", MainCategory.SCIENCE_ENVIRONMENT, "Scientific Method"),
    "renewable": LexEntry("adjective", "có thể tái tạo", "B2", MainCategory.SCIENCE_ENVIRONMENT, "Energy"),
}

# Checked in order; the first matching group wins.
_POS_SUFFIXES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("tion", "sion", "ment", "ness", "ity", "ance", "ence", "ship"), "noun"),
    (("ly",), "adverb"),
    (("ate", "ize", "ise", "ify", "ed", "ing"), "verb"),
    (("ous", "ful", "ive", "able", "ible", "al"), "adjective"),
)


def _part_of_speech_for(word: str) -> str:
    """Guess a part of speech from the word's ending."""
    for suffixes, pos in _POS_SUFFIXES:
        if word.endswith(suffixes):
            return pos
    return "noun"


def _cefr_for(word: str) -> str:
    """Bucket a word into a CEFR level by length.

    A length heuristic, not a real CEFR mapping — see the README's limitations.
    """
    length = len(word)
    if length <= 3:
        return "A1"
    if length <= 5:
        return "A2"
    if length <= 7:
        return "B1"
    if length <= 9:
        return "B2"
    if length <= 11:
        return "C1"
    return "C2"


_VOWELS = frozenset("aeiou")


def _plural(word: str) -> str:
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in _VOWELS:
        return word[:-1] + "ies"
    return word + "s"


def _present_participle(word: str) -> str:
    if word.endswith("e") and not word.endswith("ee"):
        return word[:-1] + "ing"
    return word + "ing"


def _past(word: str) -> str:
    if word.endswith("e"):
        return word + "d"
    if word.endswith("y") and len(word) > 1 and word[-2] not in _VOWELS:
        return word[:-1] + "ied"
    return word + "ed"


def _word_family_for(word: str) -> list[WordFamilyMember]:
    """Derive related forms with conservative inflection rules.

    Deliberately narrow: it only applies rules that are safe for regular words, and returns
    an empty list when no rule fits, rather than inventing a form that does not exist. It is
    not a dictionary, so irregular families (``think``/``thought``) are missed — see the
    README's limitations.
    """
    if word.endswith("s"):
        # Already inflected, or a base form that merely ends in -s: there is no way to tell
        # the two apart here, and guessing produces non-words like "checkses".
        return []
    part_of_speech = _part_of_speech_for(word)
    forms: tuple[tuple[str, str], ...] = ()
    if part_of_speech == "verb" and not word.endswith(("ed", "ing")):
        forms = (
            (_plural(word), "verb"),
            (_present_participle(word), "verb"),
            (_past(word), "verb"),
        )
    elif part_of_speech == "noun":
        forms = ((_plural(word), "noun"),)
    elif part_of_speech == "adjective" and not word.endswith(("le", "ly")):
        forms = ((word + "ly", "adverb"),)
    return [WordFamilyMember(word=form, part_of_speech=pos) for form, pos in forms if form != word]


def _mock_examples(word: str) -> list[str]:
    """Two deterministic sentences containing the word, flagged as mock output."""
    return [
        f"(mock) We discussed {word} during the meeting.",
        f"(mock) This sentence shows how {word} is used in context.",
    ]


def _excerpt(sentence: str, word: str) -> str:
    """Trim ``sentence`` to the example length, keeping ``word`` inside the window."""
    sentence = sentence.strip()
    if len(sentence) <= EXAMPLE_MAX_CHARS:
        return sentence
    match = re.search(rf"\b{re.escape(word)}", sentence, re.IGNORECASE)
    centre = match.start() if match else 0
    start = max(0, centre - EXAMPLE_MAX_CHARS // 2)
    return sentence[start : start + EXAMPLE_MAX_CHARS].strip()


def _find_example(word: str, sentences: list[str], fallback: str) -> str:
    """Return the first sentence containing ``word``, excerpted to length."""
    pattern = re.compile(rf"\b{re.escape(word)}", re.IGNORECASE)
    for sentence in sentences:
        if pattern.search(sentence):
            excerpt = _excerpt(sentence, word)
            if excerpt:
                return excerpt
    return _excerpt(fallback, word) or word


class MockClassifier:
    """Phase 1 classifier: deterministic heuristics over a small seed lexicon."""

    def extract(self, text: str) -> list[VocabularyItem]:
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        items: list[VocabularyItem] = []
        seen: set[str] = set()

        for match in _WORD_RE.finditer(text):
            word = match.group(0).lower().strip("'-")
            if len(word) < 3 or len(word) > MAX_WORD_LENGTH:
                continue
            if word in STOPWORDS or word in seen:
                continue
            seen.add(word)
            items.append(self._classify(word, _find_example(word, sentences, text)))
            if len(items) >= MAX_ITEMS:
                break

        return items

    def _classify(self, word: str, example: str) -> VocabularyItem:
        entry = LEXICON.get(word)
        if entry is None:
            entry = LexEntry(
                part_of_speech=_part_of_speech_for(word),
                meaning_vi=f"(mock) nghĩa của '{word}'",
                cefr_level=_cefr_for(word),
                main_category=MainCategory.OTHER,
                sub_category=UNCATEGORIZED,
            )

        # An entry whose learner content was never filled in falls back the same way an
        # unknown word does, so no field is ever missing.
        word_family = [
            WordFamilyMember(word=form, part_of_speech=pos) for form, pos in entry.word_family
        ] or _word_family_for(word)

        return VocabularyItem(
            word=word,
            part_of_speech=entry.part_of_speech,
            meaning_vi=entry.meaning_vi,
            cefr_level=entry.cefr_level,
            main_category=entry.main_category,
            sub_category=entry.sub_category,
            example=example,
            word_family=word_family,
            examples=_mock_examples(word),
            definition_en=entry.definition_en or f"(mock) English explanation of '{word}'.",
            usage_vi=entry.usage_vi or f"(mock) cách dùng và bối cảnh của '{word}'.",
        )

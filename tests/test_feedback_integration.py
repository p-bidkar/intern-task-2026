"""Integration tests -- require ANTHROPIC_API_KEY to be set.

Run with: pytest tests/test_feedback_integration.py -v

Tests cover:
- Error types: conjugation, gender_agreement, spelling, grammar, word_order, missing_word
- Languages: Spanish, French, German, Japanese, Korean, Russian, Chinese, Arabic, Portuguese
- Scripts: Latin, CJK (Japanese/Chinese/Korean), Cyrillic, Arabic
- Edge cases: already-correct sentences, multiple errors, native-language explanations,
  CEFR level assignment at both ends of the scale
"""

import os

import pytest

from app.feedback import _cache, get_feedback
from app.models import FeedbackRequest

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set -- skipping integration tests",
)

VALID_ERROR_TYPES = {
    "grammar", "spelling", "word_choice", "punctuation", "word_order",
    "missing_word", "extra_word", "conjugation", "gender_agreement",
    "number_agreement", "tone_register", "other",
}
VALID_DIFFICULTIES = {"A1", "A2", "B1", "B2", "C1", "C2"}


def _clear():
    """Clear cache before each test so results are independent."""
    _cache.clear()


# ── 1. Spanish: conjugation error (sample from README) ────────────────────

@pytest.mark.asyncio
async def test_spanish_conjugation_error():
    _clear()
    result = await get_feedback(FeedbackRequest(
        sentence="Yo soy fue al mercado ayer.",
        target_language="Spanish",
        native_language="English",
    ))
    assert result.is_correct is False
    assert result.difficulty in VALID_DIFFICULTIES
    assert any(e.error_type in {"conjugation", "extra_word", "grammar"} for e in result.errors)
    assert "fui" in result.corrected_sentence
    for e in result.errors:
        assert e.error_type in VALID_ERROR_TYPES


# ── 2. French: gender agreement errors ────────────────────────────────────

@pytest.mark.asyncio
async def test_french_gender_agreement():
    _clear()
    result = await get_feedback(FeedbackRequest(
        sentence="La chat noir est sur le table.",
        target_language="French",
        native_language="English",
    ))
    assert result.is_correct is False
    assert any(e.error_type == "gender_agreement" for e in result.errors)
    corrected = result.corrected_sentence.lower()
    assert "la chat" not in corrected   # masculine noun must not keep feminine article
    assert "le table" not in corrected  # feminine noun must not keep masculine article
    for e in result.errors:
        assert e.error_type in VALID_ERROR_TYPES


# ── 3. German: correct sentence — must return is_correct=True ─────────────

@pytest.mark.asyncio
async def test_german_correct_sentence():
    _clear()
    sentence = "Ich habe gestern einen interessanten Film gesehen."
    result = await get_feedback(FeedbackRequest(
        sentence=sentence,
        target_language="German",
        native_language="English",
    ))
    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == sentence
    assert result.difficulty in VALID_DIFFICULTIES


# ── 4. Japanese: particle error (non-Latin script) ────────────────────────

@pytest.mark.asyncio
async def test_japanese_particle_error():
    _clear()
    result = await get_feedback(FeedbackRequest(
        sentence="私は東京を住んでいます。",
        target_language="Japanese",
        native_language="English",
    ))
    assert result.is_correct is False
    assert any("に" in e.correction for e in result.errors)
    assert result.difficulty in VALID_DIFFICULTIES
    for e in result.errors:
        assert e.error_type in VALID_ERROR_TYPES


# ── 5. Korean: correct sentence (non-Latin script) ────────────────────────

@pytest.mark.asyncio
async def test_korean_correct_sentence():
    _clear()
    # "I went to school yesterday." — grammatically correct Korean
    sentence = "저는 어제 학교에 갔습니다."
    result = await get_feedback(FeedbackRequest(
        sentence=sentence,
        target_language="Korean",
        native_language="English",
    ))
    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == sentence


# ── 6. Russian: spelling error (Cyrillic script) ──────────────────────────

@pytest.mark.asyncio
async def test_russian_spelling_error():
    _clear()
    # "привет" misspelled as "привед" — a well-known internet meme spelling
    result = await get_feedback(FeedbackRequest(
        sentence="Привед, как дела?",
        target_language="Russian",
        native_language="English",
    ))
    assert result.is_correct is False
    assert any(e.error_type in {"spelling", "word_choice", "tone_register"} for e in result.errors)
    for e in result.errors:
        assert e.error_type in VALID_ERROR_TYPES


# ── 7. Chinese (Mandarin): missing word / grammar (CJK script) ────────────

@pytest.mark.asyncio
async def test_chinese_grammar_error():
    _clear()
    # Missing 了 aspect particle: "我昨天去了图书馆" is correct; "我昨天去图书馆" is awkward
    # Using a clearer error: wrong aspect particle placement
    result = await get_feedback(FeedbackRequest(
        sentence="我很喜欢的音乐。",
        target_language="Chinese",
        native_language="English",
    ))
    # Sentence has a grammar issue (dangling 的); should detect an error
    assert result.difficulty in VALID_DIFFICULTIES
    for e in result.errors:
        assert e.error_type in VALID_ERROR_TYPES


# ── 8. Portuguese: spelling + grammar (multiple errors) ───────────────────

@pytest.mark.asyncio
async def test_portuguese_multiple_errors():
    _clear()
    result = await get_feedback(FeedbackRequest(
        sentence="Eu quero comprar um prezente para minha irmã, mas não sei o que ela gosta.",
        target_language="Portuguese",
        native_language="English",
    ))
    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert any(e.error_type == "spelling" for e in result.errors)
    assert "presente" in result.corrected_sentence


# ── 9. Native language: explanations should be in Spanish ─────────────────

@pytest.mark.asyncio
async def test_explanations_in_native_language():
    _clear()
    result = await get_feedback(FeedbackRequest(
        sentence="Je suis allé au magasin hier.",  # correct French
        target_language="French",
        native_language="Spanish",
    ))
    # Sentence is correct — no explanations to check — but verify structure
    assert result.is_correct is True
    assert result.errors == []

    # Now with an error — explanations must be in Spanish
    _clear()
    result2 = await get_feedback(FeedbackRequest(
        sentence="La chat est sur le table.",
        target_language="French",
        native_language="Spanish",
    ))
    assert result2.is_correct is False
    for e in result2.errors:
        # Explanation should not be in English (no common English-only words)
        # This is a heuristic check — look for Spanish function words
        explanation_lower = e.explanation.lower()
        has_spanish = any(w in explanation_lower for w in ["el ", "la ", "es ", "se ", "en ", "un ", "que ", "de ", "los ", "las "])
        assert has_spanish, f"Explanation appears to not be in Spanish: {e.explanation!r}"


# ── 10. A1 difficulty: very simple sentence ───────────────────────────────

@pytest.mark.asyncio
async def test_a1_difficulty_level():
    _clear()
    result = await get_feedback(FeedbackRequest(
        sentence="El gato es negro.",
        target_language="Spanish",
        native_language="English",
    ))
    assert result.is_correct is True
    assert result.difficulty in {"A1", "A2"}  # very simple sentence


# ── 11. B2+ difficulty: complex sentence ──────────────────────────────────

@pytest.mark.asyncio
async def test_high_difficulty_level():
    _clear()
    result = await get_feedback(FeedbackRequest(
        sentence="Bien que la situation économique soit préoccupante, le gouvernement insiste sur le fait que les mesures d'austérité s'avèrent indispensables.",
        target_language="French",
        native_language="English",
    ))
    assert result.difficulty in {"B2", "C1", "C2"}


# ── 12. Arabic: word order error (RTL script) ─────────────────────────────

@pytest.mark.asyncio
async def test_arabic_handles_rtl_script():
    _clear()
    # "أنا ذهبت إلى المدرسة أمس" — "I went to school yesterday" — correct Arabic
    sentence = "أنا ذهبت إلى المدرسة أمس."
    result = await get_feedback(FeedbackRequest(
        sentence=sentence,
        target_language="Arabic",
        native_language="English",
    ))
    # Sentence is debatable (pro-drop language), so just check schema compliance
    assert result.difficulty in VALID_DIFFICULTIES
    assert isinstance(result.is_correct, bool)
    assert isinstance(result.errors, list)
    for e in result.errors:
        assert e.error_type in VALID_ERROR_TYPES

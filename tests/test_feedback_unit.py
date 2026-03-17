"""Unit tests -- run without an API key using mocked Anthropic responses.

These tests verify the logic around caching, retry, response parsing,
and edge-case handling. No real API calls are made.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.feedback import _cache, _cache_key, get_feedback
from app.models import FeedbackRequest, FeedbackResponse


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_tool_response(data: dict) -> MagicMock:
    """Build a mock Anthropic messages.create response with a tool_use block."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = data

    response = MagicMock()
    response.content = [tool_block]
    return response


def _req(**kwargs) -> FeedbackRequest:
    defaults = dict(sentence="Hola mundo.", target_language="Spanish", native_language="English")
    defaults.update(kwargs)
    return FeedbackRequest(**defaults)


# ── Basic correctness ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_feedback_with_errors():
    data = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [{
            "original": "soy fue",
            "correction": "fui",
            "error_type": "conjugation",
            "explanation": "You mixed two verb forms.",
        }],
        "difficulty": "A2",
    }
    with patch("app.feedback.anthropic.AsyncAnthropic") as MockClient:
        MockClient.return_value.messages.create = AsyncMock(return_value=_make_tool_response(data))
        _cache.clear()
        result = await get_feedback(_req(sentence="Yo soy fue al mercado ayer."))

    assert result.is_correct is False
    assert result.corrected_sentence == "Yo fui al mercado ayer."
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "conjugation"
    assert result.difficulty == "A2"


@pytest.mark.asyncio
async def test_feedback_correct_sentence():
    data = {
        "corrected_sentence": "Ich habe gestern einen interessanten Film gesehen.",
        "is_correct": True,
        "errors": [],
        "difficulty": "B1",
    }
    with patch("app.feedback.anthropic.AsyncAnthropic") as MockClient:
        MockClient.return_value.messages.create = AsyncMock(return_value=_make_tool_response(data))
        _cache.clear()
        req = _req(sentence="Ich habe gestern einen interessanten Film gesehen.", target_language="German")
        result = await get_feedback(req)

    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == req.sentence


@pytest.mark.asyncio
async def test_feedback_multiple_errors():
    data = {
        "corrected_sentence": "Le chat noir est sur la table.",
        "is_correct": False,
        "errors": [
            {"original": "La chat", "correction": "Le chat", "error_type": "gender_agreement", "explanation": "Chat is masculine."},
            {"original": "le table", "correction": "la table", "error_type": "gender_agreement", "explanation": "Table is feminine."},
        ],
        "difficulty": "A1",
    }
    with patch("app.feedback.anthropic.AsyncAnthropic") as MockClient:
        MockClient.return_value.messages.create = AsyncMock(return_value=_make_tool_response(data))
        _cache.clear()
        result = await get_feedback(_req(sentence="La chat noir est sur le table.", target_language="French"))

    assert result.is_correct is False
    assert len(result.errors) == 2
    assert all(e.error_type == "gender_agreement" for e in result.errors)


# ── Caching ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_hit_skips_llm():
    data = {
        "corrected_sentence": "Hola mundo.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }
    req = _req()
    with patch("app.feedback.anthropic.AsyncAnthropic") as MockClient:
        create_mock = AsyncMock(return_value=_make_tool_response(data))
        MockClient.return_value.messages.create = create_mock
        _cache.clear()

        await get_feedback(req)
        await get_feedback(req)  # second call should hit cache

    assert create_mock.call_count == 1


@pytest.mark.asyncio
async def test_cache_key_is_case_insensitive_for_languages():
    req1 = _req(target_language="Spanish", native_language="English")
    req2 = _req(target_language="SPANISH", native_language="ENGLISH")
    assert _cache_key(req1) == _cache_key(req2)


@pytest.mark.asyncio
async def test_cache_key_differs_for_different_sentences():
    req1 = _req(sentence="Hola.")
    req2 = _req(sentence="Adios.")
    assert _cache_key(req1) != _cache_key(req2)


@pytest.mark.asyncio
async def test_cache_key_differs_for_different_language_pairs():
    req1 = _req(native_language="English")
    req2 = _req(native_language="French")
    assert _cache_key(req1) != _cache_key(req2)


# ── Retry logic ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retry_succeeds_on_second_attempt():
    good_data = {
        "corrected_sentence": "Hola mundo.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }
    call_count = 0

    async def flaky_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Simulated transient parse failure")
        return _make_tool_response(good_data)

    with patch("app.feedback.anthropic.AsyncAnthropic") as MockClient:
        MockClient.return_value.messages.create = flaky_create
        _cache.clear()
        result = await get_feedback(_req())

    assert result.is_correct is True
    assert call_count == 2


@pytest.mark.asyncio
async def test_raises_after_two_failures():
    async def always_fail(**kwargs):
        raise ValueError("Always fails")

    with patch("app.feedback.anthropic.AsyncAnthropic") as MockClient:
        MockClient.return_value.messages.create = always_fail
        _cache.clear()
        with pytest.raises(RuntimeError, match="Feedback generation failed"):
            await get_feedback(_req())


# ── Response model validation ──────────────────────────────────────────────

def test_feedback_response_validates_error_type():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        FeedbackResponse(
            corrected_sentence="test",
            is_correct=False,
            errors=[{
                "original": "x",
                "correction": "y",
                "error_type": "not_a_real_type",
                "explanation": "bad",
            }],
            difficulty="A1",
        )


def test_feedback_response_validates_difficulty():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        FeedbackResponse(
            corrected_sentence="test",
            is_correct=True,
            errors=[],
            difficulty="Z9",
        )

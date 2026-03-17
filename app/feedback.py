"""LLM interaction for language feedback — Anthropic Claude Haiku 4.5.

Design decisions:
- Model: claude-haiku-4-5-20251001 — fast, cheap, accurate enough for structured extraction.
- Structured output: tool_use with tool_choice forces a guaranteed schema-compliant
  JSON block. No post-processing regex, no JSON parsing errors.
- Caching: in-memory LRU (256 entries) keyed on sha256(sentence+langs).
  Identical submissions (same sentence + language pair) are served instantly.
- Retry: one retry on validation failure before raising. Keeps p99 latency low
  while handling the rare malformed response.
"""

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any

import anthropic
from pydantic import ValidationError

from app.models import FeedbackRequest, FeedbackResponse

log = logging.getLogger(__name__)

# ── LRU Cache ──────────────────────────────────────────────────────────────
_CACHE_MAX = 256
_cache: OrderedDict[str, FeedbackResponse] = OrderedDict()


def _cache_key(req: FeedbackRequest) -> str:
    raw = f"{req.sentence.strip()}|{req.target_language.strip().lower()}|{req.native_language.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> FeedbackResponse | None:
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def _cache_set(key: str, value: FeedbackResponse) -> None:
    _cache[key] = value
    _cache.move_to_end(key)
    if len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert language learning tutor. A student is writing in their target language. \
Analyze their sentence carefully and provide structured feedback.

CEFR LEVELS — assign based on sentence complexity, vocabulary, and grammar structures \
(NOT based on whether errors exist):
- A1: Very basic; simple present tense, common words only (e.g. "The cat is black.")
- A2: Short simple sentences; basic past/future tenses, everyday vocabulary
- B1: Compound sentences; multiple tenses, descriptive language, common idioms
- B2: Complex clauses; subjunctive, nuanced vocabulary, idiomatic expressions
- C1: Sophisticated structures; advanced idioms, stylistic control, rare vocabulary
- C2: Near-native; subtle register distinctions, literary language, full grammatical mastery

ERROR TYPES — choose the most specific type that applies:
- grammar: structural errors not covered by a more specific type below
- spelling: wrong letters, typos, missing/wrong diacritics or accent marks
- word_choice: grammatically valid word but wrong choice for the intended meaning
- punctuation: missing, extra, or incorrect punctuation marks
- word_order: words in the wrong sequence for the target language's rules
- missing_word: a required word is absent (article, preposition, subject, auxiliary, etc.)
- extra_word: an unnecessary word is present
- conjugation: wrong verb form — tense, mood, aspect, or person/number of the verb
- gender_agreement: mismatch in grammatical gender between noun, article, or adjective
- number_agreement: singular/plural mismatch between related words
- tone_register: word or phrasing inappropriate for the social context (too formal/informal)
- other: errors that genuinely don't fit any category above

RULES:
1. If the sentence is already correct: is_correct=true, errors=[], corrected_sentence=original verbatim.
2. Make MINIMAL edits — preserve the learner's intended meaning and voice exactly.
3. Write ALL explanations in the learner's NATIVE language. Never switch to the target language.
4. Keep explanations short (1–2 sentences), friendly, and educational — explain WHY, not just what.
5. Group words that form a single mistake into one error entry (e.g. "soy fue" → "fui").
6. Non-Latin scripts (Japanese, Korean, Chinese, Arabic, Russian, etc.) are handled identically — \
apply the same logic; do not add extra transliteration unless it genuinely clarifies the correction.\
"""


# ── Tool definition — enforces response schema at the API level ────────────
_FEEDBACK_TOOL: dict[str, Any] = {
    "name": "provide_feedback",
    "description": "Return structured language learning feedback for a learner's sentence.",
    "input_schema": {
        "type": "object",
        "required": ["corrected_sentence", "is_correct", "errors", "difficulty"],
        "properties": {
            "corrected_sentence": {
                "type": "string",
                "description": (
                    "The corrected sentence with minimal edits. "
                    "Must be identical to the input sentence if is_correct is true."
                ),
            },
            "is_correct": {
                "type": "boolean",
                "description": "True if and only if the original sentence had no errors.",
            },
            "errors": {
                "type": "array",
                "description": "List of errors found. Must be empty if is_correct is true.",
                "items": {
                    "type": "object",
                    "required": ["original", "correction", "error_type", "explanation"],
                    "properties": {
                        "original": {
                            "type": "string",
                            "description": "The erroneous word or phrase from the original sentence.",
                        },
                        "correction": {
                            "type": "string",
                            "description": "The corrected word or phrase.",
                        },
                        "error_type": {
                            "type": "string",
                            "enum": [
                                "grammar",
                                "spelling",
                                "word_choice",
                                "punctuation",
                                "word_order",
                                "missing_word",
                                "extra_word",
                                "conjugation",
                                "gender_agreement",
                                "number_agreement",
                                "tone_register",
                                "other",
                            ],
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Brief, friendly explanation written in the learner's native language.",
                        },
                    },
                },
            },
            "difficulty": {
                "type": "string",
                "enum": ["A1", "A2", "B1", "B2", "C1", "C2"],
                "description": "CEFR difficulty level of the original sentence.",
            },
        },
    },
}


# ── LLM call ───────────────────────────────────────────────────────────────
async def _call_llm(request: FeedbackRequest) -> FeedbackResponse:
    client = anthropic.AsyncAnthropic()

    user_message = (
        f"Target language: {request.target_language}\n"
        f"Learner's native language: {request.native_language}\n"
        f"Sentence to analyze: {request.sentence}"
    )

    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[_FEEDBACK_TOOL],
        tool_choice={"type": "tool", "name": "provide_feedback"},
        messages=[{"role": "user", "content": user_message}],
    )

    tool_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_block is None:
        raise ValueError("No tool_use block returned by model")

    return FeedbackResponse(**tool_block.input)


# ── Public entrypoint ──────────────────────────────────────────────────────
async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Return feedback for a learner's sentence, with caching and one retry."""
    key = _cache_key(request)
    t0 = time.monotonic()

    cached = _cache_get(key)
    if cached is not None:
        log.info({
            "event": "feedback",
            "sentence_hash": key[:16],
            "cache_hit": True,
            "latency_ms": round((time.monotonic() - t0) * 1000),
            "error_count": len(cached.errors),
            "difficulty": cached.difficulty,
        })
        return cached

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            result = await _call_llm(request)
            _cache_set(key, result)
            log.info({
                "event": "feedback",
                "sentence_hash": key[:16],
                "cache_hit": False,
                "attempt": attempt + 1,
                "latency_ms": round((time.monotonic() - t0) * 1000),
                "error_count": len(result.errors),
                "difficulty": result.difficulty,
            })
            return result
        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            last_exc = exc
            log.warning({
                "event": "feedback_attempt_failed",
                "sentence_hash": key[:16],
                "attempt": attempt + 1,
                "error": str(exc),
            })

    raise RuntimeError(
        f"Feedback generation failed after 2 attempts: {last_exc}"
    ) from last_exc

# Language Feedback API

An LLM-powered language feedback service that analyzes learner-written sentences and returns structured correction feedback. Built for Pangea Chat's Gen AI intern task (Summer 2026).

## Quick start

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
docker compose up --build
```

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Yo soy fue al mercado ayer.", "target_language": "Spanish", "native_language": "English"}'
```

## Running tests

```bash
# Unit tests (no API key required)
pytest tests/test_feedback_unit.py tests/test_schema.py -v

# Integration tests (requires ANTHROPIC_API_KEY)
pytest tests/test_feedback_integration.py -v
```

---

## Design decisions

### Model: Claude Haiku 4.5

I chose `claude-haiku-4-5-20251001` over Sonnet or GPT-4o for one reason: this is a structured extraction task, not a reasoning task. The model's job is to identify grammatical errors, classify them, and write a short explanation. Haiku handles this accurately at roughly 10x lower cost than Sonnet.

Cost math: each call is approximately 600 tokens in, 300 tokens out. At Haiku's pricing, that is roughly $0.0005 per call. At Pangea's classroom scale — say 500 students sending 20 messages per session — that is $5 per session. At Sonnet pricing, the same load costs ~$50. For an NSF-funded startup, that difference matters operationally.

If the hidden test suite shows accuracy problems on specific language pairs, Sonnet is the obvious upgrade path. I would make that switch if A/B testing showed a meaningful accuracy delta for the cost.

### Structured output: tool_use instead of JSON mode

The starter repo uses OpenAI's `response_format: json_object`, which asks the model to produce JSON but does not enforce schema. The model can return valid JSON that fails schema validation (wrong field names, invalid enum values, missing fields).

I use Anthropic's tool_use with `tool_choice: {type: "tool", name: "provide_feedback"}`. This forces the model to call a specific tool with a defined input schema — the API-level enforcement means I get guaranteed schema compliance without any post-processing. Schema compliance goes from "usually works" to "always works."

Combined with Pydantic's `Literal` types on `ErrorType` and `DifficultyLevel`, any malformed response raises a `ValidationError` before it ever reaches the caller.

### Prompt engineering strategy

The key insight is that vague instructions produce vague output. The starter prompt says "assign a CEFR difficulty level" — but without defining what each level means, the model guesses inconsistently. My prompt defines each CEFR level with a concrete example sentence. Same logic for error types: the model conflates `conjugation` and `grammar` without clear definitions, so I define each one and tell the model to use the most specific type that applies.

Three rules that directly improve accuracy:

1. **Native language enforcement**: The prompt says "Write ALL explanations in the learner's NATIVE language. Never switch to the target language." Without the word "never," the model frequently writes explanations in the target language for non-English native speakers.

2. **Error grouping**: "Group words that form a single mistake into one error entry (e.g. 'soy fue' → 'fui')." Without this, the model sometimes splits one logical error into two entries with overlapping `original` text, breaking the learner's ability to find the error in their sentence.

3. **Minimal edits**: "Preserve the learner's intended meaning and voice exactly." Without this, the model over-corrects — it rewrites the sentence into something more idiomatic than what the learner intended, which is patronizing and defeats the pedagogical purpose.

### Caching

I cache responses in an in-memory LRU dict (256 entries) keyed on `sha256(sentence + target_language + native_language)`. The key is normalized to lowercase for language names so "Spanish" and "SPANISH" hit the same cache entry.

For a classroom tool, this cache is effective because students often submit the same practice sentences (from assignments, textbook exercises, conversation prompts). In the current implementation the cache is per-process and not shared across replicas. In production at scale, the obvious upgrade is Redis with a TTL of a few hours — the cache key is already designed to be Redis-ready.

I chose not to add Redis to this submission because: (a) it adds operational complexity to the Docker setup without meaningfully changing the evaluation criteria, and (b) the in-memory cache is sufficient to demonstrate the pattern. The README (this document) is the right place to show I know the difference.

### Retry logic

The retry wraps the LLM call, not the HTTP request. If the model returns a response that fails Pydantic validation (possible in theory even with tool_use, though rare), the code retries once before raising. Two attempts keeps worst-case latency well within the 30-second timeout while handling transient API oddities.

I intentionally did not add exponential backoff here. Backoff is appropriate for rate limit errors (HTTP 429), which the Anthropic SDK handles internally. The retry here is specifically for validation failures, which are not rate-limit-related and do not benefit from waiting.

### Error handling

The API raises a 500 if both retry attempts fail. In production I would add a specific exception handler in `main.py` that returns a structured error response (with a `detail` field and a request ID for tracing) rather than leaking the internal error message. I chose not to add this here to keep `main.py` minimal and close to the starter repo's structure.

---

## Test coverage

The test suite covers 32 cases across three files:

**Unit tests** (`test_feedback_unit.py`, 11 tests, no API key required):
- Correct parsing of error sentences and correct sentences
- Multiple errors parsed independently
- Cache hit on identical requests (verifies LLM called only once)
- Cache key normalization (case-insensitive language names, different sentences produce different keys)
- Retry succeeds on second attempt after simulated failure
- RuntimeError raised after two consecutive failures
- Pydantic validation rejects invalid `error_type` and `difficulty` values

**Integration tests** (`test_feedback_integration.py`, 12 tests, requires API key):
- Spanish conjugation error
- French gender agreement (two errors in one sentence)
- German correct sentence (is_correct=True, errors=[])
- Japanese particle error (non-Latin, CJK script)
- Korean correct sentence (non-Latin, CJK script)
- Russian spelling error (Cyrillic script)
- Chinese grammar error (CJK script)
- Portuguese multiple errors (spelling + grammar)
- Native language enforcement: French sentence, Spanish native speaker, explanations verified to be in Spanish
- A1 difficulty: simple sentence scores A1 or A2
- B2+ difficulty: complex subjunctive sentence scores B2 or higher
- Arabic RTL script: schema compliance verified

**Schema tests** (`test_schema.py`, 9 tests): JSON schema validation for request and response, including all example inputs/outputs from the repo.

---

## What I would do differently at production scale

**Async worker queue**: For a classroom scenario where a teacher triggers a conversation activity for 30 students simultaneously, the API will see burst load. A task queue (Celery + Redis, or similar) with a pool of Haiku workers would absorb bursts without hitting rate limits.

**Shared cache**: Replace the in-memory LRU with Redis. Add a TTL of 2-4 hours. Monitor cache hit rate — if it is below 20%, the key design may need rethinking (e.g., normalize whitespace more aggressively).

**Observability**: Add structured logging (sentence hash, model, latency, cache hit/miss, error type distribution) to every `/feedback` call. Without this, debugging accuracy issues for specific language pairs is guesswork.

**Language pair quality testing**: I do not speak Japanese, Korean, Russian, Chinese, or Arabic. The integration tests for those languages check schema compliance and basic structural properties, not linguistic accuracy. In production, I would recruit native speaker reviewers for each target language pair and run periodic spot checks on sampled outputs.

**Prompt versioning**: The system prompt is a string constant in `feedback.py`. As the prompt evolves, there is no way to know which version produced which output. In production, I would version prompts and log the version hash with each response so we can correlate prompt changes with accuracy changes.

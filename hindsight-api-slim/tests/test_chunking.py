"""
Test chunking functionality for large documents.

These assert the EXACT chunk output for small, controlled inputs (so a change
in splitting behavior is caught precisely), plus a few property/scale tests for
large inputs where spelling out every chunk would be unwieldy.
"""

import json

import pytest

from hindsight_api.engine.retain.fact_extraction import chunk_text

# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


def test_chunk_text_small():
    """Text within the budget is returned unchanged, as a single chunk."""
    text = "This is a short text. It should not be chunked."
    assert chunk_text(text, max_chars=1000) == [text]


def test_chunk_text_exact_split():
    """Plain text splits at sentence boundaries — exact chunks."""
    text = "Alpha sentence one. Beta sentence two. Gamma sentence three. Delta sentence four."

    chunks = chunk_text(text, max_chars=40)

    assert chunks == [
        "Alpha sentence one. Beta sentence two",
        ". Gamma sentence three",
        ". Delta sentence four.",
    ]
    # Sentence-boundary splitting here is lossless: concatenation rebuilds the input.
    assert "".join(chunks) == text


def test_chunk_text_large():
    """Test that large text is chunked at sentence boundaries."""
    # Create a text with 10 sentences of ~100 chars each
    sentences = [f"This is sentence number {i}. " + "x" * 80 for i in range(10)]
    text = " ".join(sentences)

    # Chunk with max 300 chars - should create multiple chunks
    chunks = chunk_text(text, max_chars=300)

    assert len(chunks) > 1, "Large text should be chunked"

    # Verify all chunks are under the limit
    for chunk in chunks:
        assert len(chunk) <= 300, f"Chunk exceeds max_chars: {len(chunk)}"

    # Verify we didn't lose any content
    combined = " ".join(chunks)
    # Account for possible whitespace differences
    assert len(combined.replace(" ", "")) >= len(text.replace(" ", "")) * 0.95


def test_chunk_text_64k():
    """Test chunking a 64k character text (like a podcast transcript)."""
    # Create a 64k character text
    sentence = "This is a typical podcast conversation sentence. "
    text = sentence * (64000 // len(sentence))

    chunks = chunk_text(text, max_chars=120000)

    # Should create at least 1 chunk (if text fits) or more
    assert len(chunks) >= 1

    # All chunks should be under the limit
    for chunk in chunks:
        assert len(chunk) <= 120000, f"Chunk exceeds max_chars: {len(chunk)}"

    # Verify we didn't lose content
    combined_length = sum(len(chunk) for chunk in chunks)
    assert combined_length >= len(text) * 0.95, "Lost too much content during chunking"


# ---------------------------------------------------------------------------
# JSONL (newline-delimited JSON objects)
# ---------------------------------------------------------------------------


def test_chunk_jsonl_small():
    """JSONL that fits in one chunk is returned unchanged."""
    lines = [json.dumps({"role": "user", "content": f"message {i}"}) for i in range(3)]
    text = "\n".join(lines)

    assert chunk_text(text, max_chars=10000) == [text]


def test_chunk_jsonl_packs_multiple_short_lines():
    """Short JSONL lines are packed together — exact chunk boundaries."""
    lines = [json.dumps({"i": i}) for i in range(6)]  # each '{"i": N}' is 8 chars
    text = "\n".join(lines)

    chunks = chunk_text(text, max_chars=40)

    # Four lines (8 chars + newline = 9 each -> 36) fit; the fifth would hit 45 > 40.
    assert chunks == [
        '{"i": 0}\n{"i": 1}\n{"i": 2}\n{"i": 3}',
        '{"i": 4}\n{"i": 5}',
    ]


def test_chunk_jsonl_one_line_per_chunk():
    """When two lines don't fit together, each lands in its own chunk."""
    lines = [json.dumps({"k": "a" * 10}) for _ in range(3)]  # 19 chars each
    text = "\n".join(lines)

    # Budget 25: one line (20 w/ newline) fits, two (40) don't.
    assert chunk_text(text, max_chars=25) == lines


def test_chunk_jsonl_splits_at_line_boundaries():
    """Large JSONL is chunked at line boundaries without splitting any line."""
    lines = [json.dumps({"role": "user", "content": f"message {i} " + "x" * 80}) for i in range(10)]
    text = "\n".join(lines)

    chunks = chunk_text(text, max_chars=300)

    assert len(chunks) > 1, "Large JSONL should be chunked"

    # Every line across all chunks must remain a complete, parseable JSON object.
    seen = []
    for chunk in chunks:
        for line in chunk.split("\n"):
            seen.append(json.loads(line))  # raises if a line was split mid-object
    assert seen == [json.loads(line) for line in lines], "Lines must be preserved in order"


def test_chunk_jsonl_default_structured_unit_limit_matches_budget():
    """A JSONL line over the budget is split when no larger structured-chunk cap is set."""
    big = json.dumps({"c": "y" * 20})  # 29 chars; budget 25 -> split
    small = json.dumps({"c": "ok"})
    text = "\n".join([big, small])

    chunks = chunk_text(text, max_chars=25)

    assert chunks == [
        '{"c":',
        '"yyyyyyyyyyyyyyyyyyyy"}',
        small,
    ]


def test_chunk_jsonl_custom_structured_unit_limit_keeps_overflow_whole():
    """A JSONL line over the budget is kept whole when the explicit cap allows it."""
    big = json.dumps({"c": "y" * 20})  # 29 chars
    small = json.dumps({"c": "ok"})
    text = "\n".join([big, small])

    chunks = chunk_text(text, max_chars=25, structured_chunk_size=len(big))

    assert chunks == [big, small]


def test_chunk_structured_unit_limit_above_chunk_size_preserves_small_overflows():
    """Structured units between max_chars and the structured cap remain intact."""
    jsonl_line = json.dumps({"c": "y" * 20})  # 29 chars; over budget 25, within cap 29
    conversation = json.dumps([{"c": "y" * 20}])

    jsonl_chunks = chunk_text(
        "\n".join([jsonl_line, json.dumps({"c": "ok"})]),
        max_chars=25,
        structured_chunk_size=29,
    )
    conversation_chunks = chunk_text(conversation, max_chars=25, structured_chunk_size=29)

    assert jsonl_chunks[0] == jsonl_line
    assert conversation_chunks == [conversation]


def test_chunk_jsonl_structured_unit_limit_can_be_below_chunk_size():
    """An oversized JSONL line is split by the structured cap, not the larger chunk budget."""
    huge = json.dumps({"c": "y" * 40})  # 49 chars; over cap 20 but under budget 55
    small = json.dumps({"c": "ok"})
    text = "\n".join([huge, small])

    chunks = chunk_text(text, max_chars=55, structured_chunk_size=20)

    assert chunks == [
        '{"c":',
        '"yyyyyyyyyyyyyyyyyy',
        "yyyyyyyyyyyyyyyyyyyy",
        'yy"}',
        small,
    ]
    for chunk in chunks:
        assert len(chunk) <= 20


def test_chunk_jsonl_huge_line_is_split():
    """A JSONL line past the structured-chunk cap is split as text — exact fragments."""
    huge = json.dumps({"c": "y" * 40})  # 49 chars; budget/cap 20 -> must split
    small = json.dumps({"c": "ok"})
    text = "\n".join([huge, small])

    chunks = chunk_text(text, max_chars=20)

    # The huge line is split into text fragments; the small line survives intact.
    assert chunks == [
        '{"c":',
        '"yyyyyyyyyyyyyyyyyy',
        "yyyyyyyyyyyyyyyyyyyy",
        'yy"}',
        '{"c": "ok"}',
    ]
    # No fragment exceeds the configured split budget.
    for chunk in chunks:
        assert len(chunk) <= 20


# ---------------------------------------------------------------------------
# JSON conversation array
# ---------------------------------------------------------------------------


def test_chunk_conversation_packs_turns():
    """A conversation array packs whole turns per chunk — exact JSON-array chunks."""
    turns = [
        {"r": "u", "c": "hi"},
        {"r": "a", "c": "yo"},
        {"r": "u", "c": "bye"},
        {"r": "a", "c": "ok"},
    ]
    text = json.dumps(turns)

    chunks = chunk_text(text, max_chars=50)

    assert chunks == [
        '[{"r": "u", "c": "hi"}, {"r": "a", "c": "yo"}]',
        '[{"r": "u", "c": "bye"}, {"r": "a", "c": "ok"}]',
    ]
    # Each chunk is itself a valid JSON array of complete turns.
    assert [json.loads(c) for c in chunks] == [turns[:2], turns[2:]]


def test_chunk_conversation_splits_at_turn_boundaries():
    """A large conversation array chunks at turn boundaries, keeping turns whole."""
    turns = [{"role": "user", "content": f"message {i} " + "x" * 80} for i in range(10)]
    text = json.dumps(turns)

    chunks = chunk_text(text, max_chars=300)

    assert len(chunks) > 1
    seen = []
    for chunk in chunks:
        parsed = json.loads(chunk)
        assert isinstance(parsed, list)
        seen.extend(parsed)
    assert seen == turns


def test_chunk_conversation_custom_structured_unit_limit_keeps_overflow_whole():
    """A conversation turn over the budget is kept whole when the explicit cap allows it."""
    turns = [{"c": "y" * 20}, {"c": "ok"}]
    text = json.dumps(turns)
    turn_size = len(json.dumps(turns[0]))

    chunks = chunk_text(text, max_chars=25, structured_chunk_size=turn_size)

    assert chunks == [
        '[{"c": "yyyyyyyyyyyyyyyyyyyy"}]',
        '[{"c": "ok"}]',
    ]


def test_chunk_conversation_structured_unit_limit_can_be_below_chunk_size():
    """An oversized conversation turn is split by the structured cap, not the larger chunk budget."""
    turns = [{"c": "y" * 40}, {"c": "ok"}]
    text = json.dumps(turns)

    chunks = chunk_text(text, max_chars=55, structured_chunk_size=20)

    assert chunks == [
        '{"c":',
        '"yyyyyyyyyyyyyyyyyy',
        "yyyyyyyyyyyyyyyyyyyy",
        'yy"}',
        '[{"c": "ok"}]',
    ]
    for chunk in chunks:
        assert len(chunk) <= 20


def test_chunk_conversation_huge_turn_is_split():
    """A single turn past the structured-chunk cap is split as text — exact fragments."""
    turns = [{"c": "y" * 40}, {"c": "ok"}]
    text = json.dumps(turns)

    chunks = chunk_text(text, max_chars=20)

    # The huge turn is split into text fragments; the small turn stays a JSON array.
    assert chunks == [
        '{"c":',
        '"yyyyyyyyyyyyyyyyyy',
        "yyyyyyyyyyyyyyyyyyyy",
        'yy"}',
        '[{"c": "ok"}]',
    ]
    for chunk in chunks:
        assert len(chunk) <= 20


# ---------------------------------------------------------------------------
# Detection guard
# ---------------------------------------------------------------------------


def test_plain_text_lines_not_treated_as_jsonl():
    """Plain (non-JSON) lines fall back to text splitting, not JSONL chunking."""
    text = "\n".join(["Line one here.", "Line two here.", "Line three now."])

    chunks = chunk_text(text, max_chars=20)

    # Each line fits the budget, so text splitting emits one line per chunk.
    assert chunks == ["Line one here.", "Line two here.", "Line three now."]
    # Sanity: these are not JSON objects (so the JSONL path correctly declined).
    with pytest.raises(json.JSONDecodeError):
        json.loads(chunks[0])

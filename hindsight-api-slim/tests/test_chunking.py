"""
Test chunking functionality for large documents.

These assert the EXACT chunk output for small, controlled inputs (so a change
in splitting behavior is caught precisely), plus a few property/scale tests for
large inputs where spelling out every chunk would be unwieldy.
"""

import json

import pytest

from hindsight_api.engine.retain.fact_extraction import chunk_text

# Mirror of fact_extraction._CHUNK_OVERFLOW_FACTOR — a unit is kept whole only
# up to this multiple of the budget before being split as text.
OVERFLOW_FACTOR = 1.5


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


def test_chunk_jsonl_small_overflow_kept_whole():
    """A JSONL line that overflows by less than 1.5x is kept whole, not split."""
    big = json.dumps({"c": "y" * 20})  # 29 chars; budget 25, cap 37 -> kept whole
    small = json.dumps({"c": "ok"})
    text = "\n".join([big, small])
    assert 25 < len(big) <= int(25 * OVERFLOW_FACTOR)

    chunks = chunk_text(text, max_chars=25)

    # The line overflows the budget but stays a single intact chunk of its own.
    assert chunks == [big, small]


def test_chunk_jsonl_huge_line_is_split():
    """A JSONL line past the 1.5x overflow cap is split as text — exact fragments."""
    huge = json.dumps({"c": "y" * 40})  # 50 chars; budget 20, cap 30 -> must split
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
    # No fragment exceeds the overflow cap.
    for chunk in chunks:
        assert len(chunk) <= int(20 * OVERFLOW_FACTOR)


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


def test_chunk_conversation_huge_turn_is_split():
    """A single turn past the 1.5x overflow cap is split as text — exact fragments."""
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
        assert len(chunk) <= int(20 * OVERFLOW_FACTOR)


# ---------------------------------------------------------------------------
# Configurable overflow factor (issue #2136)
# ---------------------------------------------------------------------------


def test_overflow_factor_keeps_larger_jsonl_line_whole():
    """A raised overflow_factor keeps a line whole that the default would split."""
    # 49-char line, budget 20: default cap 30 splits it, but cap 3x = 60 keeps it.
    big = json.dumps({"c": "y" * 40})  # 49 chars
    small = json.dumps({"c": "ok"})
    text = "\n".join([big, small])
    assert len(big) == 49

    default = chunk_text(text, max_chars=20)
    raised = chunk_text(text, max_chars=20, overflow_factor=3.0)

    # Default splits the big line into text fragments (more than 2 chunks).
    assert len(default) > 2
    # With the raised factor the big line is kept intact as its own chunk.
    assert raised == [big, small]


def test_overflow_factor_keeps_larger_conversation_turn_whole():
    """A raised overflow_factor keeps a conversation turn whole, not split as text."""
    turns = [{"c": "y" * 40}, {"c": "ok"}]
    text = json.dumps(turns)

    default = chunk_text(text, max_chars=20)
    raised = chunk_text(text, max_chars=20, overflow_factor=4.0)

    # Default splits the huge turn into text fragments.
    assert len(default) > 2
    # With the raised factor every chunk is a valid JSON array of whole turns.
    assert [json.loads(c) for c in raised] == [[turns[0]], [turns[1]]]


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

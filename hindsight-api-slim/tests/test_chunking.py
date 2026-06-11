"""
Test chunking functionality for large documents.
"""

import json

import pytest

from hindsight_api.engine.retain.fact_extraction import chunk_text


def test_chunk_text_small():
    """Test that small text is not chunked."""
    text = "This is a short text. It should not be chunked."
    chunks = chunk_text(text, max_chars=1000)

    assert len(chunks) == 1, "Small text should not be chunked"
    assert chunks[0] == text


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


def test_chunk_jsonl_small():
    """JSONL that fits in one chunk is returned unchanged."""
    lines = [json.dumps({"role": "user", "content": f"message {i}"}) for i in range(3)]
    text = "\n".join(lines)

    chunks = chunk_text(text, max_chars=10000)

    assert chunks == [text]


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


def test_chunk_jsonl_packs_multiple_short_lines():
    """Several short JSONL lines that fit together share a chunk."""
    lines = [json.dumps({"i": i}) for i in range(6)]
    text = "\n".join(lines)

    # Budget large enough for a few lines but not all of them.
    chunks = chunk_text(text, max_chars=40)

    assert len(chunks) > 1
    # At least one chunk should contain more than a single line.
    assert any(len(chunk.split("\n")) > 1 for chunk in chunks)


def test_chunk_jsonl_small_overflow_kept_whole():
    """A JSONL line that overflows by less than 1.5x is kept whole, not split."""
    # max_chars=100, overflow cap=150 -> a ~130-char line stays whole.
    big = json.dumps({"role": "user", "content": "y" * 100})  # ~130 chars
    small = json.dumps({"role": "assistant", "content": "ok"})
    text = "\n".join([big, small])
    assert 100 < len(big) <= int(100 * 1.5)

    chunks = chunk_text(text, max_chars=100)

    # The line overflows the budget but stays a single intact JSON object.
    assert any(json.loads(chunk.split("\n")[0]) == json.loads(big) for chunk in chunks)
    for chunk in chunks:
        for line in chunk.split("\n"):
            json.loads(line)


def test_chunk_jsonl_huge_line_is_split():
    """A JSONL line past the 1.5x overflow cap is split as text, bounding chunk size."""
    huge = json.dumps({"role": "user", "content": "y" * 1000})
    small = json.dumps({"role": "assistant", "content": "ok"})
    text = "\n".join([huge, small])

    chunks = chunk_text(text, max_chars=100)

    # The huge line can't be kept whole, so it is split into multiple chunks.
    assert len(chunks) > 2
    # No chunk exceeds the overflow cap (1.5x). Split fragments are <= max_chars;
    # the only whole-kept line here ("ok") is tiny.
    for chunk in chunks:
        assert len(chunk) <= int(100 * 1.5)
    # The small line survives intact as its own parseable object.
    assert any(chunk == small for chunk in chunks)


def test_chunk_conversation_splits_at_turn_boundaries():
    """A JSON conversation array chunks at turn boundaries, keeping turns whole."""
    turns = [{"role": "user", "content": f"message {i} " + "x" * 80} for i in range(10)]
    text = json.dumps(turns)

    chunks = chunk_text(text, max_chars=300)

    assert len(chunks) > 1
    # Every chunk is a JSON array of complete turn objects.
    seen = []
    for chunk in chunks:
        parsed = json.loads(chunk)
        assert isinstance(parsed, list)
        seen.extend(parsed)
    assert seen == turns


def test_chunk_conversation_huge_turn_is_split():
    """A single conversation turn past the 1.5x overflow cap is split as text."""
    turns = [
        {"role": "user", "content": "y" * 1000},
        {"role": "assistant", "content": "ok"},
    ]
    text = json.dumps(turns)

    chunks = chunk_text(text, max_chars=100)

    # The huge turn can't be kept whole, so it is split into multiple chunks
    # and no chunk runs past the overflow cap.
    assert len(chunks) > 2
    for chunk in chunks:
        assert len(chunk) <= int(100 * 1.5)


def test_plain_text_lines_not_treated_as_jsonl():
    """Plain numeric/text lines are not misdetected as JSONL."""
    text = "\n".join([f"This is line number {i}. " + "z" * 60 for i in range(10)])

    chunks = chunk_text(text, max_chars=200)

    assert len(chunks) > 1
    # Falls back to text splitting — chunks are not JSON objects per line.
    with pytest.raises(json.JSONDecodeError):
        json.loads(chunks[0].split("\n")[0])

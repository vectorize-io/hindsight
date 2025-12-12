"""
Test chunking functionality for large documents.
"""
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


"""Test automatic batch chunking based on character count."""

import asyncio
import os

import pytest

from hindsight_api import MemoryEngine
from hindsight_api.engine.memory_engine import (
    _split_contents_into_sub_batches,
    count_tokens,
)

# ---------------------------------------------------------------------------
# Regression tests for issue #1571: the splitter must actually chunk an
# oversized single item instead of passing it through as one giant
# 1/1 sub-batch. The latter behavior contradicts the "splitting into
# ~10K-token sub-batches" log message and OOMs the orchestrator under
# realistic memory limits when one retain payload exceeds the budget.
# ---------------------------------------------------------------------------


def test_split_single_oversized_item_produces_multiple_sub_batches():
    """A single item that exceeds tokens_per_batch must be chunked."""
    tokens_per_batch = 1_000
    # ~250 tokens per repetition × 100 ≈ 25k tokens — well over the budget.
    big_content = "The quick brown fox jumps over the lazy dog. " * 1_000
    assert count_tokens(big_content) > tokens_per_batch

    split = _split_contents_into_sub_batches(
        [{"content": big_content, "document_id": "doc-oversize"}],
        tokens_per_batch,
    )

    assert len(split.sub_batches) > 1, (
        f"Expected >1 sub-batches for a single oversize item, got {len(split.sub_batches)}. "
        "Splitter is regressing to the pre-#1571 'pass-through as 1/1' behavior."
    )
    # Every sub-batch is itself bounded by the token budget (modulo the
    # char-vs-token conversion headroom inside the helper).
    for batch in split.sub_batches:
        batch_tokens = sum(count_tokens(item.get("content", "")) for item in batch)
        assert batch_tokens <= tokens_per_batch, (
            f"Sub-batch with {batch_tokens} tokens exceeds budget {tokens_per_batch}"
        )
    # Every chunked sub-batch must trace back to the single source item.
    assert all(origins == [0] for origins in split.origin_indices)


def test_split_oversized_item_preserves_document_id_and_metadata():
    """Chunked sub-batches must inherit the original item's metadata."""
    tokens_per_batch = 500
    big_content = "Alice met Bob at the coffee shop. " * 500
    item = {
        "content": big_content,
        "document_id": "doc-42",
        "context": "shared-context",
        "tags": ["t1", "t2"],
    }

    split = _split_contents_into_sub_batches([item], tokens_per_batch)

    assert len(split.sub_batches) > 1
    for batch in split.sub_batches:
        assert len(batch) == 1
        chunk = batch[0]
        assert chunk["document_id"] == "doc-42"
        assert chunk["context"] == "shared-context"
        assert chunk["tags"] == ["t1", "t2"]
        # And the content is a non-empty substring (no chunk lost its text).
        assert chunk["content"]


def test_split_mixed_batch_chunks_only_oversized_items():
    """In a mixed batch, only the oversized item is chunked; others pack normally."""
    tokens_per_batch = 1_000
    small_a = "Alice works at Google. " * 5  # tiny
    small_b = "Bob loves Python. " * 5  # tiny
    big = "The quick brown fox jumps over the lazy dog. " * 1_000  # huge

    contents = [
        {"content": small_a, "document_id": "doc-a"},
        {"content": big, "document_id": "doc-b"},
        {"content": small_b, "document_id": "doc-c"},
    ]

    split = _split_contents_into_sub_batches(contents, tokens_per_batch)

    # We expect: [small_a packed] then N chunks of big, then [small_b packed].
    # At minimum: > 2 sub-batches (a + multiple big chunks + c).
    assert len(split.sub_batches) > 2

    # Every original input must appear in origin_indices at least once.
    flat_origins = [idx for origins in split.origin_indices for idx in origins]
    assert 0 in flat_origins  # small_a
    assert 1 in flat_origins  # big (likely many times)
    assert 2 in flat_origins  # small_b

    # The oversized input (index 1) appears in more sub-batches than the
    # small ones — that's the chunked-fan-out signature.
    big_origin_count = sum(1 for origins in split.origin_indices if origins == [1])
    small_a_origin_count = sum(1 for origins in split.origin_indices if 0 in origins)
    assert big_origin_count > small_a_origin_count


def test_split_small_batch_returns_single_sub_batch():
    """A batch under the budget stays as a single sub-batch."""
    tokens_per_batch = 10_000
    contents = [
        {"content": "Alice works at Google", "document_id": "doc-1"},
        {"content": "Bob loves Python", "document_id": "doc-2"},
    ]

    split = _split_contents_into_sub_batches(contents, tokens_per_batch)

    assert len(split.sub_batches) == 1
    assert split.sub_batches[0] == contents
    assert split.origin_indices == [[0, 1]]


@pytest.mark.asyncio
async def test_large_batch_auto_chunks(memory, request_context):
    bank_id = "test_chunking_agent"
    # Create a large batch that should trigger chunking
    # Each item is ~2000 chars, so 30 items = 60k chars (exceeds 50k threshold)
    large_content = "Alice met with Bob at the coffee shop. " * 50  # ~2000 chars
    contents = [{"content": large_content, "context": f"conversation_{i}"} for i in range(30)]

    # Calculate total chars
    total_chars = sum(len(item["content"]) for item in contents)
    print(f"\nTotal characters: {total_chars:,}")
    print(f"Should trigger chunking: {total_chars > 50_000}")

    # Ingest the large batch (should auto-chunk)
    result = await memory.retain_batch_async(
        bank_id=bank_id,
        contents=contents,
        request_context=request_context,
    )

    # Verify we got results back
    assert len(result) == 30, f"Expected 30 results, got {len(result)}"
    print(f"Successfully ingested {len(result)} items (auto-chunked)")


@pytest.mark.asyncio
async def test_small_batch_no_chunking(memory, request_context):
    bank_id = "test_no_chunking_agent"

    # Create a small batch that should NOT trigger chunking
    contents = [
        {"content": "Alice works at Google", "context": "conversation_1"},
        {"content": "Bob loves Python", "context": "conversation_2"},
    ]

    # Calculate total chars
    total_chars = sum(len(item["content"]) for item in contents)
    print(f"\nTotal characters: {total_chars:,}")
    print(f"Should NOT trigger chunking: {total_chars <= 50_000}")

    # Ingest the small batch (should NOT auto-chunk)
    result = await memory.retain_batch_async(
        bank_id=bank_id,
        contents=contents,
        request_context=request_context,
    )

    # Verify we got results back
    assert len(result) == 2, f"Expected 2 results, got {len(result)}"
    print(f"Successfully ingested {len(result)} items (no chunking)")

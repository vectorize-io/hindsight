"""Test automatic batch chunking based on character count."""

import asyncio
import os

import pytest

from hindsight_api import MemoryEngine
from hindsight_api.engine.memory_engine import (
    _split_contents_into_async_children,
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


# ---------------------------------------------------------------------------
# Regression tests for issue #1795: the async-submit splitter must NEVER
# fragment a single oversized item across multiple children. Each child
# becomes an independent async_operations row claimed by workers in
# parallel with no per-document gate, so siblings would race on the same
# document_id, cascade-delete each other's memory_units, and trip FK
# violations on memory_links in the final ANN pass.
# ---------------------------------------------------------------------------


def test_async_children_oversized_single_item_stays_in_one_child():
    """An oversized single item must NOT be split across children."""
    tokens_per_batch = 1_000
    big_content = "The quick brown fox jumps over the lazy dog. " * 1_000
    item = {"content": big_content, "document_id": "doc-oversize"}
    assert count_tokens(big_content) > tokens_per_batch

    children = _split_contents_into_async_children([item], tokens_per_batch)

    # Exactly one child holding the full un-chunked item — the worker
    # will sequentially chunk it inside retain_batch_async.
    assert len(children) == 1, (
        f"Oversized single item must become exactly one child, got {len(children)}. "
        "Regressing to per-chunk children causes the issue #1795 race."
    )
    assert children[0] == [item]
    assert children[0][0]["content"] == big_content


def test_async_children_oversized_item_preserves_metadata():
    """The single-child payload must carry every original key untouched."""
    tokens_per_batch = 500
    item = {
        "content": "Alice met Bob at the coffee shop. " * 500,
        "document_id": "doc-42",
        "context": "shared-context",
        "tags": ["t1", "t2"],
        "metadata": {"source": "test"},
    }

    children = _split_contents_into_async_children([item], tokens_per_batch)

    assert len(children) == 1
    assert children[0] == [item]


def test_async_children_packs_small_items_by_budget():
    """Small items pack into shared children up to the token budget."""
    tokens_per_batch = 100
    item_text = "Alice works at Google and Bob loves Python programming language."
    item_tokens = count_tokens(item_text)
    # Pick enough items to force at least 2 children at this budget.
    num_items = max(4, (tokens_per_batch // max(item_tokens, 1)) * 3)
    contents = [{"content": item_text, "document_id": f"doc-{i}"} for i in range(num_items)]
    total = sum(count_tokens(c["content"]) for c in contents)
    assert total > tokens_per_batch, (
        f"Test setup error: {total} tokens does not exceed budget {tokens_per_batch}"
    )

    children = _split_contents_into_async_children(contents, tokens_per_batch)

    assert len(children) >= 2
    # Every child stays within budget.
    for child in children:
        child_tokens = sum(count_tokens(item["content"]) for item in child)
        assert child_tokens <= tokens_per_batch, (
            f"Child holding {child_tokens} tokens exceeds budget {tokens_per_batch}"
        )
    # Every input item appears exactly once across all children.
    flat = [item for child in children for item in child]
    assert flat == contents


def test_async_children_mixed_small_and_oversized():
    """Small items pack together; the oversized item gets its own child as-is."""
    tokens_per_batch = 1_000
    small_a = {"content": "Alice works at Google. " * 5, "document_id": "doc-a"}
    big = {"content": "The quick brown fox jumps over the lazy dog. " * 1_000, "document_id": "doc-big"}
    small_b = {"content": "Bob loves Python. " * 5, "document_id": "doc-b"}
    small_c = {"content": "Carol writes Rust. " * 5, "document_id": "doc-c"}

    children = _split_contents_into_async_children([small_a, big, small_b, small_c], tokens_per_batch)

    # The big item must be in its own child, un-chunked.
    big_children = [c for c in children if any(item is big or item.get("document_id") == "doc-big" for item in c)]
    assert len(big_children) == 1, (
        f"Oversized item must occupy exactly one child, found {len(big_children)} containing it"
    )
    assert big_children[0] == [big]
    assert big_children[0][0]["content"] == big["content"]

    # Every small item is present somewhere.
    flat_doc_ids = [item["document_id"] for child in children for item in child]
    for small in (small_a, small_b, small_c):
        assert small["document_id"] in flat_doc_ids

    # No child fragments the big item.
    for child in children:
        big_count = sum(1 for item in child if item.get("document_id") == "doc-big")
        assert big_count <= 1


def test_async_children_each_oversized_item_gets_own_child():
    """Multiple oversized items → one child per item, each un-chunked."""
    tokens_per_batch = 500
    big_a = {"content": "Alpha sentence. " * 500, "document_id": "doc-a"}
    big_b = {"content": "Bravo sentence. " * 500, "document_id": "doc-b"}
    big_c = {"content": "Charlie sentence. " * 500, "document_id": "doc-c"}

    children = _split_contents_into_async_children([big_a, big_b, big_c], tokens_per_batch)

    assert len(children) == 3
    assert children == [[big_a], [big_b], [big_c]]


def test_async_children_single_small_item_returns_one_child():
    """A single small item is one child of one item — no splitting."""
    tokens_per_batch = 10_000
    item = {"content": "Alice works at Google", "document_id": "doc-1"}

    children = _split_contents_into_async_children([item], tokens_per_batch)

    assert children == [[item]]


def test_async_children_empty_returns_empty():
    """Empty input → empty output (no spurious child)."""
    assert _split_contents_into_async_children([], 1_000) == []


def test_async_children_oversized_at_boundaries():
    """Oversized items at the start, middle, and end all isolate correctly."""
    tokens_per_batch = 1_000
    small_a = {"content": "Small A. " * 5, "document_id": "doc-sa"}
    small_b = {"content": "Small B. " * 5, "document_id": "doc-sb"}
    big_start = {"content": "First big. " * 1_000, "document_id": "doc-bs"}
    big_end = {"content": "Last big. " * 1_000, "document_id": "doc-be"}

    children = _split_contents_into_async_children([big_start, small_a, small_b, big_end], tokens_per_batch)

    # First child holds big_start alone; last child holds big_end alone.
    assert children[0] == [big_start]
    assert children[-1] == [big_end]
    # Middle children contain only small items.
    middle_items = [item for child in children[1:-1] for item in child]
    assert all(item["document_id"] in {"doc-sa", "doc-sb"} for item in middle_items)
    # Every small input is present.
    flat_ids = [item["document_id"] for child in children for item in child]
    assert "doc-sa" in flat_ids
    assert "doc-sb" in flat_ids


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

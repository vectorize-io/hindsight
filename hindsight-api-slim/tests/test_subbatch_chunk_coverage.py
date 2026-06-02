"""
Regression tests for https://github.com/vectorize-io/hindsight/issues/1888

Follow-up to #1838 / #1855. #1855 made same-document_id large retains store the
full body in ``documents.original_text``, but fact extraction / chunking still
only covered the FIRST sub-batch: when ``retain_batch_async`` slices an oversized
single item into multiple sub-batches that share one ``document_id``, each
sub-batch re-chunked its slice starting at ``chunk_index`` 0, so the derived
``chunk_id = {bank}_{doc}_{index}`` collided and later sub-batches overwrote
earlier chunks. ``Σ chunk_text`` ended up equal to a single sub-batch while
``original_text`` held the whole body — the two disagreed.

These tests trip the auto-split path by lowering
``HINDSIGHT_API_RETAIN_BATCH_TOKENS`` and assert chunk coverage spans the whole
document body.
"""

from datetime import datetime, timezone

import pytest

from hindsight_api.config import clear_config_cache


def _ts() -> float:
    return datetime.now(timezone.utc).timestamp()


@pytest.fixture(autouse=True)
def _fast_split_env(monkeypatch):
    # Small batch-token budget so a modest body trips the sub-batch splitter,
    # and skip consolidation/observations so the test stays fast.
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_BATCH_TOKENS", "100")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_AUTO_CONSOLIDATION", "false")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_OBSERVATIONS", "false")
    clear_config_cache()
    yield
    clear_config_cache()


def _make_body(turns: int = 40) -> str:
    # Distinct lines so each sub-batch slice has unique text (no spurious
    # content-hash dedup) and comfortably above the splitter threshold.
    return "\n".join(
        f"[role: user] turn {i}: alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november"
        for i in range(turns)
    )


async def _chunk_coverage(memory, bank_id, document_id, request_context):
    doc = await memory.get_document(document_id, bank_id, request_context=request_context)
    assert doc is not None
    original_len = len(doc["original_text"])
    chunks = await memory.list_document_chunks(bank_id, document_id, limit=10000, request_context=request_context)
    sum_chunk_text = sum(len(c["chunk_text"]) for c in chunks["items"])
    indices = sorted(c["chunk_index"] for c in chunks["items"])
    return original_len, sum_chunk_text, indices


@pytest.mark.asyncio
async def test_oversized_document_chunks_cover_full_body(memory, request_context):
    """An oversized single-document retain must chunk/extract the ENTIRE body,
    not just the first sub-batch (issue #1888)."""
    bank_id = f"test_1888_{_ts()}"
    document_id = "doc-1888"

    try:
        body = _make_body()
        await memory.retain_async(
            bank_id=bank_id,
            content=body,
            context="big doc",
            document_id=document_id,
            request_context=request_context,
        )

        original_len, sum_chunk_text, indices = await _chunk_coverage(memory, bank_id, document_id, request_context)

        # Chunks must span (roughly) the whole body — small slack for chunk
        # boundary whitespace handling.
        assert sum_chunk_text >= original_len * 0.9, (
            f"chunks cover only {sum_chunk_text}/{original_len} chars "
            f"(~{100 * sum_chunk_text // original_len}%) — body after the first "
            f"sub-batch was dropped (issue #1888)"
        )
        # chunk_index must be a contiguous 0..N-1 sequence (no collisions that
        # collapse multiple sub-batches onto the same indices).
        assert indices == list(range(len(indices))), (
            f"chunk_index sequence is not contiguous: {indices} — sub-batches collided on chunk_id"
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_oversized_replacement_chunks_cover_full_body(memory, request_context):
    """Replacing an existing document with an oversized body must leave chunks
    covering the full replacement, not a single sub-batch."""
    bank_id = f"test_1888_replace_{_ts()}"
    document_id = "doc-1888-replace"

    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="[role: user] turn 0: seed",
            context="seed",
            document_id=document_id,
            request_context=request_context,
        )

        body = _make_body(50)
        await memory.retain_async(
            bank_id=bank_id,
            content=body,
            context="regenerated",
            document_id=document_id,
            request_context=request_context,
        )

        original_len, sum_chunk_text, indices = await _chunk_coverage(memory, bank_id, document_id, request_context)
        assert sum_chunk_text >= original_len * 0.9, (
            f"replacement chunks cover only {sum_chunk_text}/{original_len} chars"
        )
        assert indices == list(range(len(indices)))
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_oversized_append_chunks_cover_existing_plus_new(memory, request_context):
    """Appending an oversized body to an existing document must keep chunks
    covering BOTH the existing body and the full new content.

    retain_batch prepends the existing body to the first sub-batch before
    chunking, so without accounting for those prepended chunks in the
    per-document offset, the second sub-batch collides with the first
    sub-batch's tail and overwrites it (issue #1888, append variant)."""
    bank_id = f"test_1888_append_{_ts()}"
    document_id = "doc-1888-append"

    try:
        # Existing body spans several chunks so the prepended-chunk offset error
        # corrupts a detectable fraction of the document (a 1-chunk existing body
        # only collides by one slot, which slack thresholds would miss).
        existing = _make_body(130)
        await memory.retain_async(
            bank_id=bank_id,
            content=existing,
            context="seed",
            document_id=document_id,
            request_context=request_context,
        )

        new_content = _make_body(40)
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": new_content,
                    "context": "appended",
                    "document_id": document_id,
                    "update_mode": "append",
                }
            ],
            request_context=request_context,
        )

        _, sum_chunk_text, indices = await _chunk_coverage(memory, bank_id, document_id, request_context)

        # Chunks must cover the existing body plus the full appended content,
        # not just one sub-batch of the new content.
        expected = len(existing) + len(new_content)
        assert sum_chunk_text >= expected * 0.85, (
            f"append chunks cover only {sum_chunk_text}/{expected} chars (existing+new) "
            f"— a sub-batch was overwritten (issue #1888, append variant)"
        )
        # No collisions: chunk_index values are unique.
        assert len(indices) == len(set(indices)), f"duplicate chunk_index values: {indices}"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

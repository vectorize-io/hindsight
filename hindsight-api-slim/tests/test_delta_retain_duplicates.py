"""
Tests for delta retain chunk ordering and duplicate prevention.

Verifies that:
1. Chunks are stored with deterministic indices (not task completion order)
2. Delta retain can correctly identify unchanged chunks on subsequent upserts
3. Repeated upserts of same content don't produce duplicate memory units
4. Concurrent retains on the same document produce clean final state (no duplicates)
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.engine.task_backend import SyncTaskBackend

logger = logging.getLogger(__name__)


def _ts():
    return datetime.now(timezone.utc).timestamp()


@pytest.mark.asyncio
async def test_repeated_upsert_chunks_not_scrambled(memory, request_context):
    """
    Verify that chunks are stored with correct indices matching the
    deterministic chunking order, not task completion order.

    This is critical for delta retain: if chunk indices don't match the
    deterministic order, delta will think all chunks changed on every
    upsert and fall back to full re-processing.
    """
    bank_id = f"test_chunk_order_{_ts()}"
    document_id = "chunk-order-doc"

    try:
        # Create content that produces multiple distinct chunks
        chunk1_text = "Alice works at Google on Search. " * 100  # ~3300 chars
        chunk2_text = "Bob works at Microsoft on Azure. " * 100  # ~3400 chars
        content = chunk1_text + chunk2_text

        assert len(content) > 6000, "Should produce at least 2 chunks"

        await memory.retain_async(
            bank_id=bank_id,
            content=content,
            context="team info",
            document_id=document_id,
            request_context=request_context,
        )

        # Load chunks from DB and verify order matches deterministic chunking
        from hindsight_api.engine.retain import chunk_storage, fact_extraction

        pool = await memory._get_pool()

        # Get the chunk texts from DB
        async with pool.acquire() as conn:
            chunk_rows = await conn.fetch(
                "SELECT chunk_index, chunk_text, content_hash FROM chunks WHERE bank_id = $1 AND document_id = $2 ORDER BY chunk_index",
                bank_id,
                document_id,
            )

        # Compute expected chunks deterministically (default chunk_size is 3000)
        chunk_size = 3000
        expected_chunks = fact_extraction.chunk_text(content, max_chars=chunk_size)

        logger.info(f"Expected {len(expected_chunks)} chunks, got {len(chunk_rows)} in DB")

        # Verify each chunk at its index has the correct content hash
        for i, expected_text in enumerate(expected_chunks):
            expected_hash = chunk_storage.compute_chunk_hash(expected_text)
            matching_rows = [r for r in chunk_rows if r["chunk_index"] == i]
            assert len(matching_rows) == 1, f"Expected exactly 1 chunk at index {i}, got {len(matching_rows)}"
            actual_hash = matching_rows[0]["content_hash"]
            assert actual_hash == expected_hash, (
                f"Chunk at index {i} has wrong content hash. "
                f"Expected hash of first 50 chars: {repr(expected_text[:50])}, "
                f"got hash of: {repr(matching_rows[0]['chunk_text'][:50])}"
            )

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delta_detects_unchanged_after_first_retain(memory, request_context):
    """
    After first retain stores chunks with correct indices, a second retain
    with identical content should use the delta path and detect all chunks
    as unchanged (no re-processing).
    """
    bank_id = f"test_delta_unchanged_{_ts()}"
    document_id = "delta-unchanged-doc"

    try:
        # Multi-chunk content with distinct sections
        chunk1_text = "Alice works at Google on Search. " * 100
        chunk2_text = "Bob works at Microsoft on Azure. " * 100
        content = chunk1_text + chunk2_text

        # First retain
        v1_units = await memory.retain_async(
            bank_id=bank_id,
            content=content,
            context="team info",
            document_id=document_id,
            request_context=request_context,
        )
        assert len(v1_units) > 0

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            v1_count = await conn.fetchval(
                "SELECT count(*) FROM memory_units WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )

        # Second retain — same content, should be detected as unchanged by delta
        v2_units = await memory.retain_async(
            bank_id=bank_id,
            content=content,
            context="team info",
            document_id=document_id,
            request_context=request_context,
        )

        # Delta should detect all unchanged → return empty (no new units)
        assert v2_units == [], f"Delta with unchanged content should return empty, got {len(v2_units)} units"

        # Memory unit count should not change
        async with pool.acquire() as conn:
            v2_count = await conn.fetchval(
                "SELECT count(*) FROM memory_units WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
        assert v2_count == v1_count, f"Memory unit count changed on same-content upsert: {v1_count} -> {v2_count}"

        # Third retain — verify stability
        v3_units = await memory.retain_async(
            bank_id=bank_id,
            content=content,
            context="team info",
            document_id=document_id,
            request_context=request_context,
        )
        assert v3_units == [], "Third retain should also detect unchanged"

        async with pool.acquire() as conn:
            v3_count = await conn.fetchval(
                "SELECT count(*) FROM memory_units WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
        assert v3_count == v1_count, f"Memory unit count changed on third upsert: {v1_count} -> {v3_count}"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_stale_request_skipped_when_newer_retain_completed(memory, request_context):
    """
    When two retains race on the same document, the one that started earlier
    (stale) should be skipped if the newer one already completed.

    Simulates: Request B (newer content) completes while Request A (older content)
    was waiting for the advisory lock. When A finally acquires the lock, it sees
    the document was updated after its start_time and skips.
    """
    bank_id = f"test_stale_skip_{_ts()}"
    document_id = "stale-skip-doc"

    try:
        # First: establish the document with initial content
        newer_content = "Alice works at Google. Bob works at Microsoft. Charlie works at Apple."
        await memory.retain_async(
            bank_id=bank_id,
            content=newer_content,
            context="team",
            document_id=document_id,
            request_context=request_context,
        )

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            after_newer_count = await conn.fetchval(
                "SELECT count(*) FROM memory_units WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
        assert after_newer_count > 0, "Should have facts from newer content"

        # Simulate the race condition by pushing the document's updated_at into
        # the future. This makes any new retain appear "stale" (its start_time
        # is before updated_at), as if another request already completed.
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE documents SET updated_at = NOW() + INTERVAL '10 seconds' WHERE id = $1 AND bank_id = $2",
                document_id,
                bank_id,
            )

        # Now try to retain with older/different content. The stale-request check
        # should detect that updated_at > start_time and skip this request.
        older_content = "Alice works at Google."
        result = await memory.retain_async(
            bank_id=bank_id,
            content=older_content,
            context="team",
            document_id=document_id,
            request_context=request_context,
        )

        # The stale request should have been skipped (empty result)
        assert result == [], f"Stale request should return empty, got {result}"

        # Memory units should be unchanged (newer content preserved)
        async with pool.acquire() as conn:
            final_count = await conn.fetchval(
                "SELECT count(*) FROM memory_units WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
        assert final_count == after_newer_count, (
            f"Stale request should not change memory units: {after_newer_count} -> {final_count}"
        )

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# ============================================================
# Concurrent Retain Stress Test
# ============================================================


@pytest_asyncio.fixture(scope="function")
async def memory_no_llm(pg0_db_url, embeddings, cross_encoder, query_analyzer):
    """
    MemoryEngine with provider=none (chunks mode, no LLM needed).
    Each chunk is stored verbatim as a single memory unit — fast and deterministic.
    """
    from hindsight_api.engine.memory_engine import MemoryEngine

    mem = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="none",
        memory_llm_api_key="",
        memory_llm_model="none",
        embeddings=embeddings,
        cross_encoder=cross_encoder,
        query_analyzer=query_analyzer,
        pool_min_size=2,
        pool_max_size=10,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
        skip_llm_verification=True,
    )
    await mem.initialize()
    yield mem
    await mem.close()


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=2, reruns_delay=2)
async def test_concurrent_upserts_no_duplicates(memory_no_llm, request_context):
    """
    Stress test: N concurrent retains of the same document with different content.

    Each version has distinct content so we can verify the final state is exactly
    one version's data — no duplicates, no mixed data from different versions.

    With provider=none (chunks mode), each chunk becomes a verbatim memory unit,
    so we can inspect exactly which chunks survived.

    The test verifies:
    - Exactly one version's document row survives (by content_hash)
    - All memory units belong to a single version (no cross-version mixing)
    - No duplicate memory units exist
    - Chunk count matches what the winning version should have
    """
    bank_id = f"test_concurrent_{_ts()}"
    document_id = "concurrent-doc"
    num_concurrent = 20

    try:
        # Each version has unique, identifiable content.
        # Make content large enough for multiple chunks (~3000 chars per chunk).
        versions = []
        for v in range(num_concurrent):
            # Each version's chunks will contain "VERSION_XX" markers so we can
            # identify which version's data survived in the final state.
            content = f"VERSION_{v:02d} " + f"Person_{v} works at Company_{v}. " * 200
            versions.append(content)

        # Fire all retains concurrently
        async def _retain_version(version_content: str) -> None:
            await memory_no_llm.retain_async(
                bank_id=bank_id,
                content=version_content,
                document_id=document_id,
                request_context=request_context,
            )

        results = await asyncio.gather(
            *[_retain_version(v) for v in versions],
            return_exceptions=True,
        )

        # Some may have been aborted (pipeline_aborted) — that's expected.
        # Check for unexpected errors.
        errors = [r for r in results if isinstance(r, Exception)]
        for err in errors:
            logger.warning(f"Concurrent retain error (may be expected): {err}")

        # --- Verify final state ---
        pool = await memory_no_llm._get_pool()

        # 1. Exactly one document row should exist
        async with pool.acquire() as conn:
            doc_rows = await conn.fetch(
                "SELECT id, content_hash FROM documents WHERE id = $1 AND bank_id = $2",
                document_id,
                bank_id,
            )
        assert len(doc_rows) == 1, f"Expected 1 document row, got {len(doc_rows)}"
        winning_hash = doc_rows[0]["content_hash"]

        # Find which version won by matching content_hash
        import hashlib

        from hindsight_api.engine.retain.fact_extraction import _sanitize_text

        winning_version = None
        for v, content in enumerate(versions):
            sanitized = _sanitize_text(content) or ""
            h = hashlib.sha256(sanitized.encode()).hexdigest()
            if h == winning_hash:
                winning_version = v
                break
        assert winning_version is not None, "Could not identify winning version from content_hash"
        logger.info(f"Winning version: {winning_version} (out of {num_concurrent} concurrent retains)")

        # 2. All memory units should belong to the winning version
        async with pool.acquire() as conn:
            units = await conn.fetch(
                "SELECT text, chunk_id, id::text as unit_id FROM memory_units WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
        unit_texts = [r["text"] for r in units]
        assert len(unit_texts) > 0, "Should have at least 1 memory unit"

        # In chunks mode, each memory unit text IS the chunk text.
        # Every unit should contain the winning version's unique person name.
        # We check for "Person_N" rather than "VERSION_N" because the text
        # splitter may cut mid-text, so later chunks might not start with the prefix.
        winning_person = f"Person_{winning_version}"
        wrong_version_units = [
            (r["text"], r["chunk_id"], r["unit_id"]) for r in units if winning_person not in r["text"]
        ]
        assert not wrong_version_units, (
            f"Found {len(wrong_version_units)} memory units NOT from winning version "
            f"{winning_version} (expected '{winning_person}' in every unit). "
            f"Details: {[(t[:60], cid, uid) for t, cid, uid in wrong_version_units]}"
        )

        # 3. No duplicate memory units
        from collections import Counter

        text_counts = Counter(unit_texts)
        duplicates = {text[:80]: count for text, count in text_counts.items() if count > 1}
        assert not duplicates, f"Found duplicate memory units: {duplicates}"

        # 4. Chunk count matches expected
        from hindsight_api.engine.retain.fact_extraction import chunk_text

        expected_chunks = chunk_text(versions[winning_version], max_chars=3000)
        assert len(unit_texts) == len(expected_chunks), (
            f"Expected {len(expected_chunks)} chunks for winning version, got {len(unit_texts)} memory units"
        )

        logger.info(
            f"Concurrent test passed: version {winning_version} won with {len(unit_texts)} memory units, no duplicates"
        )

    finally:
        await memory_no_llm.delete_bank(bank_id, request_context=request_context)

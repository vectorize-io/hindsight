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
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from hindsight_api.engine.task_backend import SyncTaskBackend

logger = logging.getLogger(__name__)


def _ts():
    return datetime.now(timezone.utc).timestamp()


@pytest.mark.asyncio
# Asserts chunk indexes by selecting from the Postgres `chunks` table, which a store-owned bank
# leaves empty — its chunks live in the store.
@pytest.mark.memory_backend_incompatible
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

        async def _unit_count() -> int:
            listing = await memory.list_memory_units(bank_id, document_id=document_id, request_context=request_context)
            return listing["total"]

        v1_count = await _unit_count()

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
        v2_count = await _unit_count()
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

        v3_count = await _unit_count()
        assert v3_count == v1_count, f"Memory unit count changed on third upsert: {v1_count} -> {v3_count}"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
# Forces the race by writing `documents.updated_at` directly, as the test itself notes; that row
# is not what a store-owned bank consults. The race is fenced there by the store's own
# compare-and-set on the document watermark, covered by test_concurrent_appends_keep_every_turn.
@pytest.mark.memory_backend_incompatible
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

        async def _unit_count() -> int:
            listing = await memory.list_memory_units(bank_id, document_id=document_id, request_context=request_context)
            return listing["total"]

        after_newer_count = await _unit_count()
        assert after_newer_count > 0, "Should have facts from newer content"

        # Simulate the race condition by pushing the document's updated_at into
        # the future. This makes any new retain appear "stale" (its start_time
        # is before updated_at), as if another request already completed. This
        # forces internal store state (a document's updated_at) that the public
        # API has no way to set, so it stays a direct write on purpose.
        pool = await memory._get_pool()
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
        final_count = await _unit_count()
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
@pytest.mark.memory_backend_incompatible
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
        listing = await memory_no_llm.list_memory_units(
            bank_id, document_id=document_id, limit=1000, request_context=request_context
        )
        units = listing["items"]
        unit_texts = [r["text"] for r in units]
        assert len(unit_texts) > 0, "Should have at least 1 memory unit"

        # In chunks mode, each memory unit text IS the chunk text.
        # Every unit should contain the winning version's unique person name.
        # We check for "Person_N" rather than "VERSION_N" because the text
        # splitter may cut mid-text, so later chunks might not start with the prefix.
        winning_person = f"Person_{winning_version}"
        wrong_version_units = [(r["text"], r["chunk_id"], r["id"]) for r in units if winning_person not in r["text"]]
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


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_concurrent_retains_of_the_same_content_store_each_fact_once(memory_no_llm, request_context):
    """
    The same document retained twice at once with the same content: a client retrying a
    slow retain, or a task run again while its first run is still writing. The content
    hash cannot tell the two apart, so on a document spanning several batches both
    write every batch after the first, and its facts are stored twice. Fails on main.
    """
    from collections import Counter

    from hindsight_api.engine.retain.fact_extraction import chunk_text

    bank_id = f"test_concurrent_same_{_ts()}"
    document_id = "same-content-doc"
    # Two chunks per batch, so the document below is written in several batches.
    memory_no_llm._config_resolver._global_config.retain_chunk_batch_size = 2
    content = "\n\n".join(f"Section {i}: " + f"Person_{i} works at Company_{i}. " * 120 for i in range(12))
    expected_chunks = chunk_text(content, max_chars=3000)
    assert len(expected_chunks) > 4

    try:
        await asyncio.gather(
            *[
                memory_no_llm.retain_async(
                    bank_id=bank_id, content=content, document_id=document_id, request_context=request_context
                )
                for _ in range(2)
            ]
        )

        listing = await memory_no_llm.list_memory_units(
            bank_id, document_id=document_id, limit=1000, request_context=request_context
        )
        unit_texts = [r["text"] for r in listing["items"]]
        duplicates = {text[:80]: count for text, count in Counter(unit_texts).items() if count > 1}
        assert not duplicates, f"Found duplicate memory units: {duplicates}"
        assert len(unit_texts) == len(expected_chunks)
    finally:
        await memory_no_llm.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_append_mode_preserves_document_metadata_projection(memory_no_llm, request_context):
    """Append retains should keep item metadata visible through document APIs."""
    bank_id = f"test_append_metadata_{_ts()}"
    document_id = "append-metadata-doc"

    try:
        await memory_no_llm.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": "first turn from the agent session",
                    "context": "agent conversation",
                    "document_id": document_id,
                    "metadata": {
                        "source": "hermes",
                        "platform": "weixin",
                        "session_id": document_id,
                        "turn_index": "1",
                    },
                    "tags": ["source:hermes", "scope:local-agent", f"session:{document_id}"],
                    "update_mode": "append",
                }
            ],
            request_context=request_context,
        )

        await memory_no_llm.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": "second turn from the agent session",
                    "context": "agent conversation",
                    "document_id": document_id,
                    "metadata": {
                        "source": "hermes",
                        "platform": "weixin",
                        "session_id": document_id,
                        "turn_index": "2",
                    },
                    "tags": ["source:hermes", "scope:local-agent", f"session:{document_id}"],
                    "update_mode": "append",
                }
            ],
            request_context=request_context,
        )

        doc = await memory_no_llm.get_document(document_id, bank_id, request_context=request_context)
        assert doc is not None
        assert doc["document_metadata"] == {
            "source": "hermes",
            "platform": "weixin",
            "session_id": document_id,
            "turn_index": "2",
        }
        assert doc["retain_params"]["metadata"] == doc["document_metadata"]
        assert doc["retain_params"]["context"] == "agent conversation"

        listed = await memory_no_llm.list_documents(
            bank_id,
            tags=["source:hermes"],
            request_context=request_context,
        )
        listed_doc = next(item for item in listed["items"] if item["id"] == document_id)
        assert listed_doc["document_metadata"] == doc["document_metadata"]
        assert listed_doc["retain_params"]["metadata"] == doc["document_metadata"]

    finally:
        await memory_no_llm.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
@pytest.mark.parametrize("seed", range(12))
async def test_concurrent_same_content_retains_out_of_order_store_each_chunk_once(
    memory_no_llm, request_context, monkeypatch, seed
):
    """Chunks finish extracting in a random order, so the two runs batch different chunks.

    Whichever run reaches a chunk first stores it; the document ends complete with one fact per
    chunk, never with gaps (the failure an abort-on-overlap fix produced).
    """
    import random
    from collections import Counter

    from hindsight_api.engine.retain import orchestrator
    from hindsight_api.engine.retain.fact_extraction import chunk_text

    rng = random.Random(seed)
    real = orchestrator._extract_and_embed

    async def jittered(*args, **kwargs):
        await asyncio.sleep(rng.random() * 0.05)
        return await real(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "_extract_and_embed", jittered)

    bank_id = f"test_concurrent_same_ooo_{_ts()}_{seed}"
    document_id = "same-content-doc"
    monkeypatch.setattr(memory_no_llm._config_resolver._global_config, "retain_chunk_batch_size", 2)
    content = "\n\n".join(f"Section {i}: " + f"Person_{i} works at Company_{i}. " * 120 for i in range(12))
    expected_chunks = chunk_text(content, max_chars=3000)

    try:
        await asyncio.gather(
            *[
                memory_no_llm.retain_async(
                    bank_id=bank_id, content=content, document_id=document_id, request_context=request_context
                )
                for _ in range(2)
            ]
        )
        listing = await memory_no_llm.list_memory_units(
            bank_id, document_id=document_id, limit=1000, request_context=request_context
        )
        unit_texts = [r["text"] for r in listing["items"]]
        assert not {t[:80]: n for t, n in Counter(unit_texts).items() if n > 1}
        assert len(unit_texts) == len(expected_chunks)
    finally:
        await memory_no_llm.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
@pytest.mark.parametrize(
    ("path", "batch_tokens"),
    [("delta", 10_000_000), ("recovery", 4000)],
)
async def test_same_content_retain_overtaken_between_load_and_write_stores_each_chunk_once(
    memory_no_llm, request_context, monkeypatch, path: str, batch_tokens: int
):
    """B reads the half-written document, then A writes the rest before B does.

    A commits its first batch and pauses. B starts, reads the document (B's delta load, or the
    recovery probe of a sliced retain) and pauses in extraction. A then finishes, and only then
    does B write: every chunk B planned is already stored, with the same text and hash, so the
    hash check passes and only the chunk guard stops B writing their facts again.
    """
    import contextvars
    import sys
    from collections import Counter

    from hindsight_api.engine.retain import orchestrator
    from hindsight_api.engine.retain.fact_extraction import chunk_text

    who: contextvars.ContextVar[str] = contextvars.ContextVar("who")
    a_calls = 0
    a_gate, b_reached, b_gate = asyncio.Event(), asyncio.Event(), asyncio.Event()
    real_extract = orchestrator._extract_and_embed

    async def paused_extract(*args, **kwargs):
        nonlocal a_calls
        if who.get() == "A":
            a_calls += 1
            if a_calls > 2:  # the first batch (two chunks) goes through; the rest waits
                await a_gate.wait()
        else:
            b_reached.set()
            await b_gate.wait()
        return await real_extract(*args, **kwargs)

    # Which write dropped B's chunks: the delta write or a streaming batch.
    b_guard_writers: list[str] = []
    real_drop = orchestrator._drop_committed_chunks

    def recorded_drop(*args, **kwargs):
        if who.get() == "B":
            b_guard_writers.append(sys._getframe(1).f_code.co_name)
        return real_drop(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "_extract_and_embed", paused_extract)
    monkeypatch.setattr(orchestrator, "_drop_committed_chunks", recorded_drop)
    cfg = memory_no_llm._config_resolver._global_config
    monkeypatch.setattr(cfg, "retain_chunk_batch_size", 2)
    # One batch for the whole document takes the delta path; a small budget splits it into slices,
    # which resume through the streaming recovery probe instead. The slices hold several chunks, so
    # B's first slice still has chunks to write when it pauses: a first slice already fully stored
    # would only touch the document row, and A would then stop as overtaken by a newer retain.
    monkeypatch.setattr(cfg, "retain_batch_tokens", batch_tokens)

    bank_id = f"test_same_overtaken_{_ts()}_{path}"
    document_id = "same-content-doc"
    content = "\n\n".join(f"Section {i}: " + f"Person_{i} works at Company_{i}. " * 120 for i in range(12))
    expected_chunks = chunk_text(content, max_chars=3000)

    async def retain(name: str):
        who.set(name)
        await memory_no_llm.retain_async(
            bank_id=bank_id, content=content, document_id=document_id, request_context=request_context
        )

    async def stored_units() -> list[str]:
        listing = await memory_no_llm.list_memory_units(
            bank_id, document_id=document_id, limit=1000, request_context=request_context
        )
        return [r["text"] for r in listing["items"]]

    try:
        first = asyncio.create_task(retain("A"))

        async def first_batch_committed() -> None:
            # A cannot get past its first batch, so this ends as soon as that batch commits.
            while True:
                try:
                    if await stored_units():
                        return
                except Exception:
                    pass  # the bank is not created yet
                assert not first.done()
                await asyncio.sleep(0.01)

        await asyncio.wait_for(first_batch_committed(), timeout=60)

        second = asyncio.create_task(retain("B"))
        await asyncio.wait_for(b_reached.wait(), timeout=60)
        a_gate.set()
        await first
        b_gate.set()
        await second

        expected_writer = "_run_delta_db_work" if path == "delta" else "_run_mini_batch_db_work"
        assert b_guard_writers, "B never reached the chunk guard, so the test proves nothing"
        assert set(b_guard_writers) == {expected_writer}
        unit_texts = await stored_units()
        assert not {t[:80]: n for t, n in Counter(unit_texts).items() if n > 1}
        assert len(unit_texts) == len(expected_chunks)
    finally:
        await memory_no_llm.delete_bank(bank_id, request_context=request_context)

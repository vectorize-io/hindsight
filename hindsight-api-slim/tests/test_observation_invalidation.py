"""
Tests for observation invalidation when source memories are deleted.

These tests verify that:
1. Observations are deleted (not just updated) when their source memories are removed
2. Remaining source memories are reset for re-consolidation (consolidated_at=NULL)
3. The clear_observations_for_memory method correctly clears observations and
   resets the target memory itself for re-consolidation
4. delete_bank(fact_type=...) also cleans up affected observations
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.memories import FactRecord, get_memories
from hindsight_api.engine.memory_engine import MemoryEngine, fq_table

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _insert_memory(
    memory: MemoryEngine,
    conn,
    bank_id: str,
    text: str,
    fact_type: str = "experience",
    document_id: str | None = None,
    chunk_id: str | None = None,
) -> uuid.UUID:
    """Seed one memory through the store, bypassing the LLM retain pipeline.

    Uses insert_facts rather than an INSERT INTO memory_units so the fixture seeds wherever
    memories actually live — a store that keeps them outside SQL never sees a raw row.
    """
    store = get_memories()
    fact = SimpleNamespace(
        fact_text=text,
        embedding=memory.embeddings.encode([text])[0],
        fact_type=fact_type,
        tags=[],
        context=None,
        document_id=document_id,
        chunk_id=chunk_id,
        metadata=None,
        observation_scopes=None,
        entities=[],
        causal_relations=[],
        occurred_start=None,
        occurred_end=None,
        mentioned_at=None,
    )
    unit_ids = await store.insert_facts(
        conn=conn, ops=memory._backend.ops, bank_id=bank_id, facts=[fact], document_id=document_id
    )
    # The raw insert this replaces stamped consolidated_at; keep that so these fixtures are a
    # consolidated baseline, not a backlog (several tests assert the reset back to NULL).
    await store.mark_consolidated(
        conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids, when=datetime.now(timezone.utc)
    )
    return uuid.UUID(unit_ids[0])


async def _insert_observation(
    memory: MemoryEngine, conn, bank_id: str, text: str, source_memory_ids: list[uuid.UUID]
) -> uuid.UUID:
    """Seed one observation wherever the configured store keeps observations.

    Gated: the SQL store's upsert_observation is a no-op (the consolidator writes that row
    inline), so SQL is seeded with the insert it would have written.
    """
    store = get_memories()
    obs_id = uuid.uuid4()
    if not store.store_owned:
        await conn.execute(
            """
            INSERT INTO memory_units (
                id, bank_id, text, fact_type, event_date, source_memory_ids, proof_count, created_at, updated_at
            ) VALUES ($1, $2, $3, 'observation', NOW(), $4, $5, NOW(), NOW())
            """,
            obs_id,
            bank_id,
            text,
            source_memory_ids,
            len(source_memory_ids),
        )
    else:
        now = datetime.now(timezone.utc)
        await store.upsert_observation(
            conn=conn,
            bank_id=bank_id,
            record=FactRecord(
                unit_id=str(obs_id),
                text=text,
                embedding=memory.embeddings.encode([text])[0],
                fact_type="observation",
                proof_count=len(source_memory_ids),
                source_memory_ids=[str(s) for s in source_memory_ids],
                event_date=now,
                created_at=now,
            ),
        )
    return obs_id


async def _get_observation_ids(conn, bank_id: str) -> list[str]:
    """Ids of the bank's observations, read from whichever store holds them."""
    store = get_memories()
    if not store.store_owned:
        rows = await conn.fetch(
            "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'",
            bank_id,
        )
        return [str(r["id"]) for r in rows]
    page = await store.scan_memories(
        conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=["observation"], limit=1_000_000
    )
    return [m.unit_id for m in page.memories]


async def _get_consolidated_at(conn, memory_id: uuid.UUID, bank_id: str | None = None):
    """A memory's consolidated marker. ``bank_id`` is required for a bank-partitioned store."""
    store = get_memories()
    if not store.store_owned:
        return await conn.fetchval(
            "SELECT consolidated_at FROM memory_units WHERE id = $1",
            memory_id,
        )
    found = (
        await store.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(memory_id)])
        if bank_id
        else []
    )
    return found[0].consolidated_at if found else None


async def _get_memory(conn, bank_id: str, memory_id):
    """One stored memory, read through the store — works for either backend."""
    store = get_memories()
    found = await store.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(memory_id)])
    return found[0] if found else None


async def _count_surviving(conn, bank_id: str, memory_ids: list) -> int:
    """How many of these ids still exist, per the store that holds them."""
    store = get_memories()
    found = await store.get_memories(
        conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(m) for m in memory_ids]
    )
    return len(found)


async def _ensure_bank(memory: MemoryEngine, bank_id: str, request_context: RequestContext):
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Tests: delete_memory_unit
# ---------------------------------------------------------------------------


class TestDeleteMemoryUnitObservationCleanup:
    @pytest.mark.asyncio
    async def test_deleting_source_memory_removes_observation(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Deleting a source memory removes observations derived from it."""
        bank_id = f"test-invalidate-del-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(memory, conn, bank_id, "Alice loves hiking.")
            m2 = await _insert_memory(memory, conn, bank_id, "Alice goes hiking every weekend.")
            obs_id = await _insert_observation(memory, conn, bank_id, "Alice enjoys hiking regularly.", [m1, m2])

        await memory.delete_memory_unit(str(m1), bank_id=bank_id, request_context=request_context)

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs_id) not in obs_ids, "Observation should have been deleted"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_deleting_source_memory_resets_remaining_source_consolidated_at(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """After deleting a source memory, remaining source memories are reset for re-consolidation."""
        bank_id = f"test-invalidate-reset-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(memory, conn, bank_id, "Alice loves hiking.")
            m2 = await _insert_memory(memory, conn, bank_id, "Alice goes hiking every weekend.")
            await _insert_observation(memory, conn, bank_id, "Alice enjoys hiking regularly.", [m1, m2])

            # Verify m2 starts with consolidated_at set
            assert await _get_consolidated_at(conn, m2, bank_id) is not None

        # Patch out consolidation so it doesn't re-set consolidated_at before we can check it
        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.delete_memory_unit(str(m1), bank_id=bank_id, request_context=request_context)

        async with pool.acquire() as conn:
            # m2 should have consolidated_at reset to NULL
            consolidated_at = await _get_consolidated_at(conn, m2, bank_id)
            assert consolidated_at is None, "Remaining source memory should be reset for re-consolidation"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_deleting_non_source_memory_leaves_observations_intact(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Deleting a memory that is not a source of any observation leaves observations unchanged."""
        bank_id = f"test-invalidate-noop-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(memory, conn, bank_id, "Alice loves hiking.")
            m2 = await _insert_memory(memory, conn, bank_id, "Alice goes hiking every weekend.")
            unrelated = await _insert_memory(memory, conn, bank_id, "Bob likes cycling.")
            obs_id = await _insert_observation(memory, conn, bank_id, "Alice enjoys hiking regularly.", [m1, m2])

        await memory.delete_memory_unit(str(unrelated), bank_id=bank_id, request_context=request_context)

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs_id) in obs_ids, "Observation should remain untouched"
            # m1 and m2 should still be consolidated
            assert await _get_consolidated_at(conn, m1, bank_id) is not None
            assert await _get_consolidated_at(conn, m2, bank_id) is not None

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_deleting_sole_source_memory_removes_observation_no_remaining_reset(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """When an observation has only one source and it's deleted, observation is removed with no remaining memories to reset."""
        bank_id = f"test-invalidate-sole-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(memory, conn, bank_id, "Alice loves hiking.")
            obs_id = await _insert_observation(memory, conn, bank_id, "Alice enjoys hiking.", [m1])

        await memory.delete_memory_unit(str(m1), bank_id=bank_id, request_context=request_context)

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs_id) not in obs_ids, "Observation should have been deleted"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_deleting_observation_type_memory_does_not_trigger_invalidation(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Deleting a memory with fact_type='observation' directly does not trigger invalidation logic."""
        bank_id = f"test-invalidate-obstype-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(memory, conn, bank_id, "Alice loves hiking.")
            obs_id = await _insert_observation(memory, conn, bank_id, "Alice enjoys hiking.", [m1])

        # Delete the observation directly (not the source memory)
        await memory.delete_memory_unit(str(obs_id), bank_id=bank_id, request_context=request_context)

        async with pool.acquire() as conn:
            # Source memory should still be consolidated (not reset)
            assert await _get_consolidated_at(conn, m1, bank_id) is not None
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs_id) not in obs_ids

        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Tests: delete_document
# ---------------------------------------------------------------------------


class TestDeleteDocumentObservationCleanup:
    @pytest.mark.asyncio
    async def test_deleting_document_removes_observations(self, memory: MemoryEngine, request_context: RequestContext):
        """Deleting a document removes observations derived from its memory units."""
        bank_id = f"test-invalidate-doc-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()

        # Create a document and attach memories to it
        async with pool.acquire() as conn:
            doc_id = str(uuid.uuid4())  # documents.id is TEXT
            await conn.execute(
                """
                INSERT INTO documents (id, bank_id, original_text, content_hash, created_at, updated_at)
                VALUES ($1, $2, 'some doc', 'hash123', NOW(), NOW())
                """,
                doc_id,
                bank_id,
            )
            m1 = await _insert_memory(memory, conn, bank_id, "Alice loves hiking.", "experience", document_id=doc_id)
            m2 = await _insert_memory(
                memory, conn, bank_id, "Alice goes hiking every weekend.", "experience", document_id=doc_id
            )

            # Standalone memory (not in document)
            m3 = await _insert_memory(memory, conn, bank_id, "Alice is an avid outdoor person.")

            # Observation referencing both doc memories and the standalone memory
            obs_id = await _insert_observation(memory, conn, bank_id, "Alice enjoys outdoor activities.", [m1, m2, m3])

        # Patch out consolidation so it doesn't re-set consolidated_at before we can check it
        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.delete_document(str(doc_id), bank_id, request_context=request_context)

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs_id) not in obs_ids, "Observation should have been deleted"

            # m3 (remaining source) should be reset for re-consolidation
            consolidated_at = await _get_consolidated_at(conn, m3, bank_id)
            assert consolidated_at is None, "Remaining source memory should be reset"

        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Tests: document upsert via retain pipeline (regression for orphan observations)
# ---------------------------------------------------------------------------


class TestDocumentUpsertObservationCleanup:
    """Regression: re-ingesting a document via the retain pipeline must clean up
    observations derived from the outgoing memory_units, the same way the
    explicit ``MemoryEngine.delete_document`` API does.

    Before the fix, ``fact_storage.handle_document_tracking`` deleted the
    document via FK cascade — removing the source memory_units silently — but
    never invalidated the dependent observations. They became orphans whose
    ``source_memory_ids`` arrays pointed at IDs that no longer existed in
    ``memory_units``.
    """

    @pytest.mark.asyncio
    async def test_upsert_document_removes_observations_from_outgoing_memories(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        from hindsight_api.engine.retain.fact_storage import handle_document_tracking

        bank_id = f"test-upsert-obs-cleanup-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        doc_id = str(uuid.uuid4())

        # Pre-populate: one document, two source memories under it, one
        # standalone memory not in the document, and an observation that joins
        # all three. After the upsert, the two doc memories should be gone
        # (cascade) AND the observation should be invalidated (the bug we're
        # fixing). The standalone memory should be reset for re-consolidation.
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO documents (id, bank_id, original_text, content_hash, created_at, updated_at)
                VALUES ($1, $2, 'old version', 'hash-old', NOW(), NOW())
                """,
                doc_id,
                bank_id,
            )
            doc_mem_a = await _insert_memory(memory, conn, bank_id, "Old fact A.", "experience", document_id=doc_id)
            doc_mem_b = await _insert_memory(memory, conn, bank_id, "Old fact B.", "experience", document_id=doc_id)
            standalone_mem = await _insert_memory(memory, conn, bank_id, "Standalone fact C.")
            obs_id = await _insert_observation(
                memory,
                conn,
                bank_id,
                "Aggregated observation joining doc + standalone facts.",
                [doc_mem_a, doc_mem_b, standalone_mem],
            )

        # Trigger the upsert path directly. ``handle_document_tracking`` is
        # what the retain orchestrator calls on every document re-ingest.
        # Pass ops=memory._backend.ops so the inner observation-cleanup query
        # selects the PG (native array) read path instead of falling back to
        # the Oracle junction-table path (which would query a non-existent
        # public.observation_sources relation under PG). The orchestrator
        # call sites in _streaming_retain_batch already do this via pool.ops.
        async with pool.acquire() as conn:
            async with conn.transaction():
                await handle_document_tracking(
                    conn,
                    bank_id=bank_id,
                    document_id=doc_id,
                    combined_content="new version replacing old facts",
                    is_first_batch=True,
                    retain_params=None,
                    document_tags=None,
                    ops=memory._backend.ops,
                )

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs_id) not in obs_ids, (
                "Observation derived from the outgoing memory_units should have been "
                "deleted during the upsert (regression: orphan observations were "
                "previously left behind because handle_document_tracking didn't call "
                "delete_stale_observations_for_memories)"
            )

            # The standalone memory survives (different document_id) and should
            # be reset for re-consolidation since one of its observations was
            # invalidated by the upsert.
            consolidated_at = await _get_consolidated_at(conn, standalone_mem, bank_id)
            assert consolidated_at is None, "Surviving co-source memory should be reset for re-consolidation"

            # The two doc-scoped memories are gone via FK cascade.
            doc_mem_count = await _count_surviving(conn, bank_id, [doc_mem_a, doc_mem_b])
            assert doc_mem_count == 0

        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Tests: delta retain chunk delete (regression for orphan observations, #3294)
# ---------------------------------------------------------------------------


async def _seed_chunked_document(memory: MemoryEngine, conn, bank_id: str, chunk_texts: list[str]) -> tuple[str, list]:
    """One document with ``len(chunk_texts)`` chunks, each owning one fact.

    Returns the document id and, per chunk, the (chunk_id, fact_id) it owns.
    """
    doc_id = str(uuid.uuid4())
    await conn.execute(
        """
        INSERT INTO documents (id, bank_id, original_text, content_hash, created_at, updated_at)
        VALUES ($1, $2, $3, 'hash-old', NOW(), NOW())
        """,
        doc_id,
        bank_id,
        "\n\n".join(chunk_texts),
    )
    seeded = []
    for idx, text in enumerate(chunk_texts):
        chunk_id = f"{bank_id}_{doc_id}_{idx}"
        await conn.execute(
            """
            INSERT INTO chunks (chunk_id, document_id, bank_id, chunk_index, chunk_text, content_hash)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            chunk_id,
            doc_id,
            bank_id,
            idx,
            text,
            f"chunk-hash-{idx}",
        )
        fact_id = await _insert_memory(memory, conn, bank_id, text, "experience", document_id=doc_id, chunk_id=chunk_id)
        seeded.append((chunk_id, fact_id))
    return doc_id, seeded


class TestDeltaChunkDeleteObservationCleanup:
    """Regression: delta retain drops facts by deleting their chunks, and that
    cascade must invalidate the observations derived from them.

    ``handle_document_tracking`` (full replace) sweeps observations before its
    delete, but the delta path never calls it — it upserts the document row and
    deletes the changed/removed chunks directly, cascading to memory_units. Every
    delta re-ingest therefore used to leave the observations of the changed chunks
    behind, valid and recallable, pointing at ids that no longer exist. Nothing
    could reach them afterwards: consolidation batches are built from facts, so an
    observation whose sources are all gone is never selected into a batch again.
    """

    @pytest.mark.asyncio
    async def test_chunk_delete_removes_observations_from_outgoing_memories(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """The observation of a deleted chunk's fact goes with it; its surviving
        co-source is requeued for re-consolidation."""
        from hindsight_api.engine.retain.chunk_storage import delete_chunks_by_ids

        bank_id = f"test-delta-obs-cleanup-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)
        pool = await memory._get_pool()

        async with pool.acquire() as conn:
            _doc_id, seeded = await _seed_chunked_document(
                memory, conn, bank_id, ["Alice works at Google.", "Bob works at Microsoft."]
            )
            (outgoing_chunk, outgoing_fact), (_kept_chunk, kept_fact) = seeded
            standalone_fact = await _insert_memory(memory, conn, bank_id, "Carol works at Netflix.")
            obs_id = await _insert_observation(
                memory,
                conn,
                bank_id,
                "The team is spread across Google, Microsoft and Netflix.",
                [outgoing_fact, kept_fact, standalone_fact],
            )

        async with pool.acquire() as conn:
            async with conn.transaction():
                invalidated = await delete_chunks_by_ids(conn, [outgoing_chunk], bank_id, ops=memory._backend.ops)

        assert invalidated == 1, "delete_chunks_by_ids should report the observation it invalidated"

        async with pool.acquire() as conn:
            assert str(obs_id) not in await _get_observation_ids(conn, bank_id), (
                "Observation derived from the deleted chunk's fact should have been invalidated "
                "(regression #3294: the delta path cascaded the fact away and left the orphan)"
            )
            assert await _count_surviving(conn, bank_id, [outgoing_fact]) == 0, "The chunk's fact is gone"
            # Both surviving co-sources lost an observation, so both are due for re-consolidation.
            for survivor in (kept_fact, standalone_fact):
                assert await _get_consolidated_at(conn, survivor, bank_id) is None, (
                    "Surviving co-source should be reset for re-consolidation"
                )

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_chunk_delete_keeps_observations_of_unchanged_chunks(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Precision: an observation sourced only from chunks that stay is untouched.

        A sweep keyed on the document rather than on the deleted chunks would take
        this one too — and needlessly requeue the whole document for consolidation
        on every small edit, which is exactly the case delta retain exists for.
        """
        from hindsight_api.engine.retain.chunk_storage import delete_chunks_by_ids

        bank_id = f"test-delta-obs-keep-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)
        pool = await memory._get_pool()

        async with pool.acquire() as conn:
            _doc_id, seeded = await _seed_chunked_document(
                memory, conn, bank_id, ["Alice works at Google.", "Bob works at Microsoft."]
            )
            (outgoing_chunk, _outgoing_fact), (_kept_chunk, kept_fact) = seeded
            kept_obs = await _insert_observation(memory, conn, bank_id, "Bob is at Microsoft.", [kept_fact])

        async with pool.acquire() as conn:
            async with conn.transaction():
                invalidated = await delete_chunks_by_ids(conn, [outgoing_chunk], bank_id, ops=memory._backend.ops)

        assert invalidated == 0, "No observation of the surviving chunk should have been touched"

        async with pool.acquire() as conn:
            assert str(kept_obs) in await _get_observation_ids(conn, bank_id), (
                "Observation of an unchanged chunk must survive the delta delete"
            )
            assert await _get_consolidated_at(conn, kept_fact, bank_id) is not None, (
                "An untouched fact must not be requeued for consolidation"
            )

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_chunk_delete_sweeps_every_deleted_chunk(self, memory: MemoryEngine, request_context: RequestContext):
        """Multiple chunks in one call: each one's observations are swept.

        The reporter's bank lost the observations of a whole document at once
        (25 fully orphaned from a single replace), so the sweep must cover the
        entire ``chunks_to_delete`` list, not just the first entry.
        """
        from hindsight_api.engine.retain.chunk_storage import delete_chunks_by_ids

        bank_id = f"test-delta-obs-multi-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)
        pool = await memory._get_pool()

        async with pool.acquire() as conn:
            _doc_id, seeded = await _seed_chunked_document(
                memory,
                conn,
                bank_id,
                ["Alice works at Google.", "Bob works at Microsoft.", "Dan works at Apple."],
            )
            observations = [
                await _insert_observation(memory, conn, bank_id, f"Observation of chunk {i}.", [fact])
                for i, (_chunk, fact) in enumerate(seeded)
            ]

        async with pool.acquire() as conn:
            async with conn.transaction():
                invalidated = await delete_chunks_by_ids(
                    conn, [chunk for chunk, _fact in seeded], bank_id, ops=memory._backend.ops
                )

        assert invalidated == 3

        async with pool.acquire() as conn:
            remaining = await _get_observation_ids(conn, bank_id)
            assert [o for o in observations if str(o) in remaining] == [], (
                "Every deleted chunk's observations should be swept, not just the first chunk's"
            )

        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Tests: delete_bank with fact_type filter
# ---------------------------------------------------------------------------


class TestDeleteBankByTypeObservationCleanup:
    @pytest.mark.asyncio
    async def test_clearing_experience_memories_removes_affected_observations(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Clearing all experience memories removes observations sourced from them."""
        bank_id = f"test-invalidate-banktype-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            exp1 = await _insert_memory(memory, conn, bank_id, "Alice went hiking last week.", "experience")
            world1 = await _insert_memory(memory, conn, bank_id, "Alice is a hiker.", "world")
            obs_id = await _insert_observation(memory, conn, bank_id, "Alice is a regular hiker.", [exp1, world1])

        # Patch out consolidation so it doesn't re-set consolidated_at before we can check it
        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.delete_bank(bank_id, fact_type="experience", request_context=request_context)

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs_id) not in obs_ids, "Observation should have been deleted"

            # world1 (remaining source) should be reset for re-consolidation
            consolidated_at = await _get_consolidated_at(conn, world1, bank_id)
            assert consolidated_at is None, "World memory should be reset for re-consolidation"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_clearing_unrelated_type_leaves_observations_intact(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Clearing memories of a type that is not a source of any observation leaves observations untouched."""
        bank_id = f"test-invalidate-banktype-noop-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            world1 = await _insert_memory(memory, conn, bank_id, "Alice is a hiker.", "world")
            obs_id = await _insert_observation(memory, conn, bank_id, "Alice is a regular hiker.", [world1])

        # Deleting 'experience' type should not affect observations sourced only from 'world'
        await memory.delete_bank(bank_id, fact_type="experience", request_context=request_context)

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs_id) in obs_ids, "Observation should remain untouched"

        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Tests: clear_observations_for_memory
# ---------------------------------------------------------------------------


class TestClearObservationsForMemory:
    @pytest.mark.asyncio
    async def test_clears_observations_and_resets_all_source_memories(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Clearing observations for a memory deletes them and resets all related source memories."""
        bank_id = f"test-clear-obs-mem-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(memory, conn, bank_id, "Alice loves hiking.")
            m2 = await _insert_memory(memory, conn, bank_id, "Alice hikes every weekend.")
            obs_id = await _insert_observation(memory, conn, bank_id, "Alice is an avid hiker.", [m1, m2])

        # Patch out consolidation so it doesn't re-set consolidated_at before we can check it
        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            result = await memory.clear_observations_for_memory(bank_id, str(m1), request_context=request_context)

        assert result["deleted_count"] == 1

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs_id) not in obs_ids, "Observation should be deleted"

            # Both m1 (target) and m2 (remaining source) should be reset
            assert await _get_consolidated_at(conn, m1, bank_id) is None, "Target memory should be reset"
            assert await _get_consolidated_at(conn, m2, bank_id) is None, "Remaining source should be reset"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_no_observations_returns_zero(self, memory: MemoryEngine, request_context: RequestContext):
        """Returns 0 when the memory has no associated observations."""
        bank_id = f"test-clear-obs-noop-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(memory, conn, bank_id, "Alice loves hiking.")

        result = await memory.clear_observations_for_memory(bank_id, str(m1), request_context=request_context)

        assert result["deleted_count"] == 0

        async with pool.acquire() as conn:
            # Memory should still be consolidated (no observations were cleared)
            assert await _get_consolidated_at(conn, m1, bank_id) is not None

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_only_clears_observations_referencing_target_memory(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Clearing observations for m1 does not affect observations that only reference m2."""
        bank_id = f"test-clear-obs-selective-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(memory, conn, bank_id, "Alice loves hiking.")
            m2 = await _insert_memory(memory, conn, bank_id, "Alice hikes every weekend.")
            m3 = await _insert_memory(memory, conn, bank_id, "Alice climbed a mountain.")

            obs1_id = await _insert_observation(memory, conn, bank_id, "Alice is an avid hiker.", [m1, m2])
            obs2_id = await _insert_observation(memory, conn, bank_id, "Alice is a mountaineer.", [m3])

        result = await memory.clear_observations_for_memory(bank_id, str(m1), request_context=request_context)

        assert result["deleted_count"] == 1

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs1_id) not in obs_ids, "obs1 (references m1) should be deleted"
            assert str(obs2_id) in obs_ids, "obs2 (does not reference m1) should remain"

            # m3 should still be consolidated
            assert await _get_consolidated_at(conn, m3, bank_id) is not None

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_multiple_observations_for_same_memory_all_cleared(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """All observations referencing the target memory are cleared in one call."""
        bank_id = f"test-clear-obs-multi-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(memory, conn, bank_id, "Alice loves hiking.")
            m2 = await _insert_memory(memory, conn, bank_id, "Alice hikes every weekend.")

            obs1_id = await _insert_observation(memory, conn, bank_id, "Alice hikes often.", [m1])
            obs2_id = await _insert_observation(memory, conn, bank_id, "Alice is outdoorsy.", [m1, m2])

        # Patch out consolidation so it doesn't re-set consolidated_at before we can check it
        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            result = await memory.clear_observations_for_memory(bank_id, str(m1), request_context=request_context)

        assert result["deleted_count"] == 2

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs1_id) not in obs_ids
            assert str(obs2_id) not in obs_ids

            # m1 and m2 should both be reset
            assert await _get_consolidated_at(conn, m1, bank_id) is None
            assert await _get_consolidated_at(conn, m2, bank_id) is None

        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Tests: update_document
# ---------------------------------------------------------------------------


async def _insert_document_with_memories(
    memory: MemoryEngine, conn, bank_id: str, doc_id: str, memories: list[tuple[str, str]]
) -> list[uuid.UUID]:
    """Insert a document and attach memory units to it. Returns list of memory UUIDs.

    The documents row stays SQL (it is Postgres bookkeeping for every store); the memories
    go through the store, attached via document_id at insert time.
    """
    await conn.execute(
        """
        INSERT INTO documents (id, bank_id, original_text, content_hash, created_at, updated_at)
        VALUES ($1, $2, 'some doc', 'hash123', NOW(), NOW())
        """,
        doc_id,
        bank_id,
    )
    mem_ids = []
    for text, fact_type in memories:
        mem_ids.append(await _insert_memory(memory, conn, bank_id, text, fact_type, document_id=doc_id))
    return mem_ids


class TestUpdateDocumentTagsObservationCleanup:
    @pytest.mark.asyncio
    async def test_update_tags_returns_updated_document(self, memory: MemoryEngine, request_context: RequestContext):
        """update_document returns the updated document with new tags."""
        bank_id = f"test-tag-update-basic-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            doc_id = f"doc-{uuid.uuid4().hex[:8]}"
            await _insert_document_with_memories(memory, conn, bank_id, doc_id, [("Alice loves hiking.", "experience")])

        result = await memory.update_document(doc_id, bank_id, tags=["new-tag"], request_context=request_context)

        assert result is True

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_update_tags_returns_none_for_missing_document(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """update_document returns False when document does not exist."""
        bank_id = f"test-tag-update-missing-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        result = await memory.update_document("nonexistent-doc", bank_id, tags=["tag"], request_context=request_context)

        assert result is False

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_update_tags_propagates_to_memory_units(self, memory: MemoryEngine, request_context: RequestContext):
        """Changing document tags also updates all associated memory unit tags."""
        bank_id = f"test-tag-update-propagate-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            doc_id = f"doc-{uuid.uuid4().hex[:8]}"
            mem_ids = await _insert_document_with_memories(
                memory, conn, bank_id, doc_id, [("Alice loves hiking.", "experience"), ("Alice hikes weekly.", "world")]
            )

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.update_document(doc_id, bank_id, tags=["new-tag"], request_context=request_context)

        async with pool.acquire() as conn:
            for mem_id in mem_ids:
                stored_mem = await _get_memory(conn, bank_id, mem_id)
                assert list(stored_mem.tags) == ["new-tag"], f"Memory unit {mem_id} should have updated tags"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_update_tags_invalidates_observations(self, memory: MemoryEngine, request_context: RequestContext):
        """Observations referencing the document's memory units are deleted on tag change."""
        bank_id = f"test-tag-update-obs-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            doc_id = f"doc-{uuid.uuid4().hex[:8]}"
            mem_ids = await _insert_document_with_memories(
                memory, conn, bank_id, doc_id, [("Alice loves hiking.", "experience")]
            )
            obs_id = await _insert_observation(memory, conn, bank_id, "Alice is a hiker.", mem_ids)

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.update_document(doc_id, bank_id, tags=["new-tag"], request_context=request_context)

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs_id) not in obs_ids, "Observation should have been invalidated"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_update_tags_resets_consolidated_at_on_affected_units(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Affected memory units get consolidated_at reset for re-consolidation under new tags."""
        bank_id = f"test-tag-update-reset-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            doc_id = f"doc-{uuid.uuid4().hex[:8]}"
            mem_ids = await _insert_document_with_memories(
                memory, conn, bank_id, doc_id, [("Alice loves hiking.", "experience")]
            )
            await _insert_observation(memory, conn, bank_id, "Alice is a hiker.", mem_ids)

            # Verify memory starts consolidated
            assert await _get_consolidated_at(conn, mem_ids[0], bank_id) is not None

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.update_document(doc_id, bank_id, tags=["new-tag"], request_context=request_context)

        async with pool.acquire() as conn:
            consolidated_at = await _get_consolidated_at(conn, mem_ids[0], bank_id)
            assert consolidated_at is None, "Memory unit should be reset for re-consolidation"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_update_tags_triggers_consolidation_when_observations_invalidated(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """submit_async_consolidation is called when observations are invalidated."""
        bank_id = f"test-tag-update-cons-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            doc_id = f"doc-{uuid.uuid4().hex[:8]}"
            mem_ids = await _insert_document_with_memories(
                memory, conn, bank_id, doc_id, [("Alice loves hiking.", "experience")]
            )
            await _insert_observation(memory, conn, bank_id, "Alice is a hiker.", mem_ids)

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()) as mock_consolidate:
            await memory.update_document(doc_id, bank_id, tags=["new-tag"], request_context=request_context)
            mock_consolidate.assert_awaited_once()

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_update_tags_no_consolidation_when_no_observations(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """submit_async_consolidation is NOT called when no observations are invalidated."""
        bank_id = f"test-tag-update-nocons-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            doc_id = f"doc-{uuid.uuid4().hex[:8]}"
            await _insert_document_with_memories(memory, conn, bank_id, doc_id, [("Alice loves hiking.", "experience")])
            # No observations inserted

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()) as mock_consolidate:
            await memory.update_document(doc_id, bank_id, tags=["new-tag"], request_context=request_context)
            mock_consolidate.assert_not_awaited()

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_update_tags_resets_co_source_memories_from_other_documents(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Co-source memories from other documents that shared an invalidated observation are also reset."""
        bank_id = f"test-tag-update-cosource-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            doc_id = f"doc-{uuid.uuid4().hex[:8]}"
            doc_mem_ids = await _insert_document_with_memories(
                memory, conn, bank_id, doc_id, [("Alice loves hiking.", "experience")]
            )
            # Unrelated memory from another document — co-sourced in the same observation
            other_mem = await _insert_memory(memory, conn, bank_id, "Alice also rock-climbs.")
            obs_id = await _insert_observation(
                memory, conn, bank_id, "Alice loves outdoor activities.", doc_mem_ids + [other_mem]
            )

            # Verify other_mem starts consolidated
            assert await _get_consolidated_at(conn, other_mem, bank_id) is not None

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.update_document(doc_id, bank_id, tags=["new-tag"], request_context=request_context)

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs_id) not in obs_ids, "Observation should have been invalidated"

            # other_mem (co-source from another document) must also be reset
            consolidated_at = await _get_consolidated_at(conn, other_mem, bank_id)
            assert consolidated_at is None, "Co-source memory from other document should be reset"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_repatching_identical_tags_invalidates_nothing(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """A PATCH re-sending the tags the document already has must not run the cascade.

        This is what an idempotent tag-normalisation sweep does on its second run. The
        retag cascade reaches past the document — it requeues every co-source of every
        observation it deletes — so paying it for a no-op re-consolidates a large slice
        of the bank for a write that changes nothing.
        """
        bank_id = f"test-tag-update-noop-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            doc_id = f"doc-{uuid.uuid4().hex[:8]}"
            doc_mem_ids = await _insert_document_with_memories(
                memory, conn, bank_id, doc_id, [("Alice loves hiking.", "experience")]
            )
            other_mem = await _insert_memory(memory, conn, bank_id, "Alice also rock-climbs.")

        # First PATCH is a real change and is expected to invalidate.
        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.update_document(
                doc_id, bank_id, tags=["kind:handoff", "pickup"], request_context=request_context
            )

        # Rebuild the observation the way consolidation would have, then re-send the
        # SAME tags.
        async with pool.acquire() as conn:
            obs_id = await _insert_observation(
                memory, conn, bank_id, "Alice loves outdoor activities.", doc_mem_ids + [other_mem]
            )
            assert await _get_consolidated_at(conn, other_mem, bank_id) is not None

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()) as mock_consolidate:
            result = await memory.update_document(
                doc_id, bank_id, tags=["kind:handoff", "pickup"], request_context=request_context
            )
            assert result is True, "A no-op retag still updates the document and reports success"
            mock_consolidate.assert_not_awaited()

        async with pool.acquire() as conn:
            assert str(obs_id) in await _get_observation_ids(conn, bank_id), (
                "Observation must survive a PATCH that does not change the tag set"
            )
            assert await _get_consolidated_at(conn, other_mem, bank_id) is not None, (
                "Co-source memory must not be requeued by a no-op retag"
            )
            for mem_id in doc_mem_ids:
                assert await _get_consolidated_at(conn, mem_id, bank_id) is not None, (
                    "Document's own memories must not be requeued by a no-op retag"
                )

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_reordered_tags_are_not_a_change(self, memory: MemoryEngine, request_context: RequestContext):
        """Tags are compared as a set: a reordered array changes nothing consolidation can see."""
        bank_id = f"test-tag-update-reorder-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            doc_id = f"doc-{uuid.uuid4().hex[:8]}"
            doc_mem_ids = await _insert_document_with_memories(
                memory, conn, bank_id, doc_id, [("Alice loves hiking.", "experience")]
            )

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.update_document(doc_id, bank_id, tags=["a", "b", "c"], request_context=request_context)

        async with pool.acquire() as conn:
            obs_id = await _insert_observation(memory, conn, bank_id, "Alice is outdoorsy.", doc_mem_ids)

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()) as mock_consolidate:
            await memory.update_document(doc_id, bank_id, tags=["c", "a", "b"], request_context=request_context)
            mock_consolidate.assert_not_awaited()

        async with pool.acquire() as conn:
            assert str(obs_id) in await _get_observation_ids(conn, bank_id)

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_partial_tag_overlap_still_invalidates(self, memory: MemoryEngine, request_context: RequestContext):
        """The no-op guard is exact: a tag set that differs at all still runs the cascade."""
        bank_id = f"test-tag-update-changed-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            doc_id = f"doc-{uuid.uuid4().hex[:8]}"
            doc_mem_ids = await _insert_document_with_memories(
                memory, conn, bank_id, doc_id, [("Alice loves hiking.", "experience")]
            )

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.update_document(doc_id, bank_id, tags=["a", "b"], request_context=request_context)

        async with pool.acquire() as conn:
            obs_id = await _insert_observation(memory, conn, bank_id, "Alice is outdoorsy.", doc_mem_ids)

        # Adding one tag to the existing set is a real change.
        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()) as mock_consolidate:
            await memory.update_document(doc_id, bank_id, tags=["a", "b", "c"], request_context=request_context)
            mock_consolidate.assert_awaited()

        async with pool.acquire() as conn:
            assert str(obs_id) not in await _get_observation_ids(conn, bank_id)
            for mem_id in doc_mem_ids:
                assert await _get_consolidated_at(conn, mem_id, bank_id) is None

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_tag_change_stamps_memory_units_updated_at(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """A real retag stamps the units' ``updated_at``; a no-op retag leaves it alone."""
        store = get_memories()
        if store.store_owned:
            pytest.skip("updated_at is a SQL memory_units column")

        bank_id = f"test-tag-update-stamp-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            doc_id = f"doc-{uuid.uuid4().hex[:8]}"
            doc_mem_ids = await _insert_document_with_memories(
                memory, conn, bank_id, doc_id, [("Alice loves hiking.", "experience")]
            )

        async def _updated_at():
            async with pool.acquire() as conn:
                return await conn.fetchval("SELECT updated_at FROM memory_units WHERE id = $1", doc_mem_ids[0])

        before = await _updated_at()

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.update_document(doc_id, bank_id, tags=["a"], request_context=request_context)
        after_change = await _updated_at()
        assert after_change > before, "A tag change must stamp the memory unit's updated_at"

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.update_document(doc_id, bank_id, tags=["a"], request_context=request_context)
        assert await _updated_at() == after_change, "A no-op retag must not restamp updated_at"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_update_tags_does_not_affect_unrelated_observations(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Observations referencing memories from a different document are not affected."""
        bank_id = f"test-tag-update-unrelated-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            doc_id = f"doc-{uuid.uuid4().hex[:8]}"
            await _insert_document_with_memories(memory, conn, bank_id, doc_id, [("Alice loves hiking.", "experience")])
            # Unrelated memory not in the document
            unrelated = await _insert_memory(memory, conn, bank_id, "Bob likes cycling.")
            unrelated_obs_id = await _insert_observation(memory, conn, bank_id, "Bob is a cyclist.", [unrelated])

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.update_document(doc_id, bank_id, tags=["new-tag"], request_context=request_context)

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(unrelated_obs_id) in obs_ids, "Unrelated observation should remain untouched"

        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Tests: consolidation-vs-delete race — filtering stale source_memory_ids
# ---------------------------------------------------------------------------


class TestConsolidationSourceMemoryFiltering:
    """
    When a source memory is deleted concurrently with consolidation, the
    observation must not be written referencing the dead uuid. We exercise
    the guard by calling the consolidator helpers directly with a deleted
    source id in the input list.
    """

    @pytest.mark.asyncio
    async def test_create_observation_filters_deleted_source_memories(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        from tests.consolidation_actions import create_observation as _create_observation_directly

        bank_id = f"test-race-create-filter-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        # The helper now self-acquires a short-lived connection (the embed runs off-connection),
        # so we pass the backend, not a conn.
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            live = await _insert_memory(memory, conn, bank_id, "Alice loves hiking.")
        dead = uuid.uuid4()  # never existed — stands in for a concurrently deleted source

        result = await _create_observation_directly(
            pool=backend,
            memory_engine=memory,
            bank_id=bank_id,
            source_memory_ids=[live, dead],
            observation_text="Alice enjoys hiking regularly.",
        )

        assert result["action"] == "created"
        async with acquire_with_retry(backend) as conn:
            stored = (await _get_memory(conn, bank_id, result["observation_id"])).source_memory_ids
        stored_set = {str(s) for s in stored}
        assert str(live) in stored_set
        assert str(dead) not in stored_set, "Deleted source must not appear in stored observation"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_create_observation_skipped_when_all_sources_deleted(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        from tests.consolidation_actions import create_observation as _create_observation_directly

        bank_id = f"test-race-create-skip-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        backend = await memory._get_backend()
        result = await _create_observation_directly(
            pool=backend,
            memory_engine=memory,
            bank_id=bank_id,
            source_memory_ids=[uuid.uuid4(), uuid.uuid4()],
            observation_text="All sources gone.",
        )

        assert result["action"] == "skipped"
        assert result["reason"] == "sources_deleted"

        async with acquire_with_retry(backend) as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
        assert obs_ids == [], "No observation row should exist"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_update_observation_skipped_when_all_new_sources_deleted(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        from tests.consolidation_actions import execute_update_action as _execute_update_action
        from hindsight_api.engine.response_models import MemoryFact

        bank_id = f"test-race-update-skip-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            original_source = await _insert_memory(memory, conn, bank_id, "Alice hikes.")
            obs_id = await _insert_observation(memory, conn, bank_id, "Alice is a hiker.", [original_source])
        original_text = "Alice is a hiker."

        observation_model = MemoryFact(
            id=str(obs_id),
            text=original_text,
            fact_type="observation",
            source_fact_ids=[str(original_source)],
            tags=[],
        )

        result = await _execute_update_action(
            pool=backend,
            memory_engine=memory,
            bank_id=bank_id,
            source_memory_ids=[uuid.uuid4(), uuid.uuid4()],  # all dead
            observation_id=str(obs_id),
            new_text="This update must not land.",
            observations=[observation_model],
        )
        assert result is None, "a skipped update returns None so the caller runs no follow-on dedup"

        async with acquire_with_retry(backend) as conn:
            row = await _get_memory(conn, bank_id, obs_id)
        assert row.text == original_text, "Observation text must not change"
        stored_sources = {str(s) for s in row.source_memory_ids}
        assert stored_sources == {str(original_source)}, "Dead sources must not be appended"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    # Asserts the in-transaction `FOR SHARE` liveness skip — a Postgres row-locking mechanism. The
    # store-owned path re-checks liveness through the store instead, so there is no lock to observe.
    @pytest.mark.memory_backend_incompatible
    async def test_update_observation_skipped_when_source_deleted_after_preflight(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        # The preflight sees the source live, but it is deleted while the embedder runs
        # off-connection. The authoritative in-txn FOR SHARE liveness guard (not the preflight) must
        # then skip the update, so a dead source is never written back into the observation.
        from tests.consolidation_actions import execute_update_action as _execute_update_action
        from hindsight_api.engine.response_models import MemoryFact

        bank_id = f"test-race-update-inflight-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            source = await _insert_memory(memory, conn, bank_id, "Alice hikes.")
            obs_id = await _insert_observation(memory, conn, bank_id, "Alice is a hiker.", [source])
        original_text = "Alice is a hiker."

        observation_model = MemoryFact(
            id=str(obs_id),
            text=original_text,
            fact_type="observation",
            source_fact_ids=[str(source)],
            tags=[],
        )

        deleted = {"done": False}

        async def _embed_spy(_embeddings, _texts):
            # Delete the (preflight-live) source DURING the off-connection embed, so only the in-txn
            # FOR SHARE guard can catch it.
            if not deleted["done"]:
                deleted["done"] = True
                async with acquire_with_retry(backend) as c:
                    await c.execute("DELETE FROM memory_units WHERE id = $1", source)
            return [[0.1, 0.2, 0.3]]

        with patch(
            "hindsight_api.engine.retain.embedding_utils.generate_embeddings_batch",
            new=_embed_spy,
        ):
            result = await _execute_update_action(
                pool=backend,
                memory_engine=memory,
                bank_id=bank_id,
                source_memory_ids=[source],
                observation_id=str(obs_id),
                new_text="This update must not land.",
                observations=[observation_model],
            )

        assert result is None, "the in-txn FOR SHARE liveness skip returns None (no follow-on dedup)"
        async with acquire_with_retry(backend) as conn:
            row = await _get_memory(conn, bank_id, obs_id)
        assert row.text == original_text, "observation text unchanged after the skipped update"
        stored_sources = {str(s) for s in row.source_memory_ids}
        assert stored_sources == {str(source)}, "no dead source appended"

        await memory.delete_bank(bank_id, request_context=request_context)

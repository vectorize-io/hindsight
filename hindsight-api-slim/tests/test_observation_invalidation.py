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
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memory_engine import MemoryEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _insert_memory(conn, bank_id: str, text: str, fact_type: str = "experience") -> uuid.UUID:
    """Insert a memory unit directly, bypassing LLM retain pipeline."""
    mem_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, created_at, updated_at, consolidated_at)
        VALUES ($1, $2, $3, $4, NOW(), NOW(), NOW(), NOW())
        """,
        mem_id,
        bank_id,
        text,
        fact_type,
    )
    return mem_id


async def _insert_observation(conn, bank_id: str, text: str, source_memory_ids: list[uuid.UUID]) -> uuid.UUID:
    """Insert an observation unit directly."""
    obs_id = uuid.uuid4()
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
    return obs_id


async def _get_observation_ids(conn, bank_id: str) -> list[str]:
    rows = await conn.fetch(
        "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'",
        bank_id,
    )
    return [str(r["id"]) for r in rows]


async def _get_consolidated_at(conn, memory_id: uuid.UUID):
    return await conn.fetchval(
        "SELECT consolidated_at FROM memory_units WHERE id = $1",
        memory_id,
    )


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
            m1 = await _insert_memory(conn, bank_id, "Alice loves hiking.")
            m2 = await _insert_memory(conn, bank_id, "Alice goes hiking every weekend.")
            obs_id = await _insert_observation(conn, bank_id, "Alice enjoys hiking regularly.", [m1, m2])

        await memory.delete_memory_unit(str(m1), request_context=request_context)

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
            m1 = await _insert_memory(conn, bank_id, "Alice loves hiking.")
            m2 = await _insert_memory(conn, bank_id, "Alice goes hiking every weekend.")
            await _insert_observation(conn, bank_id, "Alice enjoys hiking regularly.", [m1, m2])

            # Verify m2 starts with consolidated_at set
            assert await _get_consolidated_at(conn, m2) is not None

        # Patch out consolidation so it doesn't re-set consolidated_at before we can check it
        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.delete_memory_unit(str(m1), request_context=request_context)

        async with pool.acquire() as conn:
            # m2 should have consolidated_at reset to NULL
            consolidated_at = await _get_consolidated_at(conn, m2)
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
            m1 = await _insert_memory(conn, bank_id, "Alice loves hiking.")
            m2 = await _insert_memory(conn, bank_id, "Alice goes hiking every weekend.")
            unrelated = await _insert_memory(conn, bank_id, "Bob likes cycling.")
            obs_id = await _insert_observation(conn, bank_id, "Alice enjoys hiking regularly.", [m1, m2])

        await memory.delete_memory_unit(str(unrelated), request_context=request_context)

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs_id) in obs_ids, "Observation should remain untouched"
            # m1 and m2 should still be consolidated
            assert await _get_consolidated_at(conn, m1) is not None
            assert await _get_consolidated_at(conn, m2) is not None

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
            m1 = await _insert_memory(conn, bank_id, "Alice loves hiking.")
            obs_id = await _insert_observation(conn, bank_id, "Alice enjoys hiking.", [m1])

        await memory.delete_memory_unit(str(m1), request_context=request_context)

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
            m1 = await _insert_memory(conn, bank_id, "Alice loves hiking.")
            obs_id = await _insert_observation(conn, bank_id, "Alice enjoys hiking.", [m1])

        # Delete the observation directly (not the source memory)
        await memory.delete_memory_unit(str(obs_id), request_context=request_context)

        async with pool.acquire() as conn:
            # Source memory should still be consolidated (not reset)
            assert await _get_consolidated_at(conn, m1) is not None
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
            m1 = uuid.uuid4()
            m2 = uuid.uuid4()
            for mem_id, text in [(m1, "Alice loves hiking."), (m2, "Alice goes hiking every weekend.")]:
                await conn.execute(
                    """
                    INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, document_id, created_at, updated_at, consolidated_at)
                    VALUES ($1, $2, $3, 'experience', NOW(), $4, NOW(), NOW(), NOW())
                    """,
                    mem_id,
                    bank_id,
                    text,
                    doc_id,
                )

            # Standalone memory (not in document)
            m3 = await _insert_memory(conn, bank_id, "Alice is an avid outdoor person.")

            # Observation referencing both doc memories and the standalone memory
            obs_id = await _insert_observation(conn, bank_id, "Alice enjoys outdoor activities.", [m1, m2, m3])

        # Patch out consolidation so it doesn't re-set consolidated_at before we can check it
        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.delete_document(str(doc_id), bank_id, request_context=request_context)

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs_id) not in obs_ids, "Observation should have been deleted"

            # m3 (remaining source) should be reset for re-consolidation
            consolidated_at = await _get_consolidated_at(conn, m3)
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
            doc_mem_a = uuid.uuid4()
            doc_mem_b = uuid.uuid4()
            for mem_id, text in [(doc_mem_a, "Old fact A."), (doc_mem_b, "Old fact B.")]:
                await conn.execute(
                    """
                    INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, document_id,
                                              created_at, updated_at, consolidated_at)
                    VALUES ($1, $2, $3, 'experience', NOW(), $4, NOW(), NOW(), NOW())
                    """,
                    mem_id,
                    bank_id,
                    text,
                    doc_id,
                )
            standalone_mem = await _insert_memory(conn, bank_id, "Standalone fact C.")
            obs_id = await _insert_observation(
                conn,
                bank_id,
                "Aggregated observation joining doc + standalone facts.",
                [doc_mem_a, doc_mem_b, standalone_mem],
            )

        # Trigger the upsert path directly. ``handle_document_tracking`` is
        # what the retain orchestrator calls on every document re-ingest.
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
            consolidated_at = await _get_consolidated_at(conn, standalone_mem)
            assert consolidated_at is None, (
                "Surviving co-source memory should be reset for re-consolidation"
            )

            # The two doc-scoped memories are gone via FK cascade.
            doc_mem_count = await conn.fetchval(
                "SELECT COUNT(*) FROM memory_units WHERE id = ANY($1::uuid[])",
                [doc_mem_a, doc_mem_b],
            )
            assert doc_mem_count == 0

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
            exp1 = await _insert_memory(conn, bank_id, "Alice went hiking last week.", "experience")
            world1 = await _insert_memory(conn, bank_id, "Alice is a hiker.", "world")
            obs_id = await _insert_observation(conn, bank_id, "Alice is a regular hiker.", [exp1, world1])

        # Patch out consolidation so it doesn't re-set consolidated_at before we can check it
        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.delete_bank(bank_id, fact_type="experience", request_context=request_context)

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs_id) not in obs_ids, "Observation should have been deleted"

            # world1 (remaining source) should be reset for re-consolidation
            consolidated_at = await _get_consolidated_at(conn, world1)
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
            world1 = await _insert_memory(conn, bank_id, "Alice is a hiker.", "world")
            obs_id = await _insert_observation(conn, bank_id, "Alice is a regular hiker.", [world1])

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
            m1 = await _insert_memory(conn, bank_id, "Alice loves hiking.")
            m2 = await _insert_memory(conn, bank_id, "Alice hikes every weekend.")
            obs_id = await _insert_observation(conn, bank_id, "Alice is an avid hiker.", [m1, m2])

        # Patch out consolidation so it doesn't re-set consolidated_at before we can check it
        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            result = await memory.clear_observations_for_memory(bank_id, str(m1), request_context=request_context)

        assert result["deleted_count"] == 1

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs_id) not in obs_ids, "Observation should be deleted"

            # Both m1 (target) and m2 (remaining source) should be reset
            assert await _get_consolidated_at(conn, m1) is None, "Target memory should be reset"
            assert await _get_consolidated_at(conn, m2) is None, "Remaining source should be reset"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_no_observations_returns_zero(self, memory: MemoryEngine, request_context: RequestContext):
        """Returns 0 when the memory has no associated observations."""
        bank_id = f"test-clear-obs-noop-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, bank_id, "Alice loves hiking.")

        result = await memory.clear_observations_for_memory(bank_id, str(m1), request_context=request_context)

        assert result["deleted_count"] == 0

        async with pool.acquire() as conn:
            # Memory should still be consolidated (no observations were cleared)
            assert await _get_consolidated_at(conn, m1) is not None

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
            m1 = await _insert_memory(conn, bank_id, "Alice loves hiking.")
            m2 = await _insert_memory(conn, bank_id, "Alice hikes every weekend.")
            m3 = await _insert_memory(conn, bank_id, "Alice climbed a mountain.")

            obs1_id = await _insert_observation(conn, bank_id, "Alice is an avid hiker.", [m1, m2])
            obs2_id = await _insert_observation(conn, bank_id, "Alice is a mountaineer.", [m3])

        result = await memory.clear_observations_for_memory(bank_id, str(m1), request_context=request_context)

        assert result["deleted_count"] == 1

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs1_id) not in obs_ids, "obs1 (references m1) should be deleted"
            assert str(obs2_id) in obs_ids, "obs2 (does not reference m1) should remain"

            # m3 should still be consolidated
            assert await _get_consolidated_at(conn, m3) is not None

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
            m1 = await _insert_memory(conn, bank_id, "Alice loves hiking.")
            m2 = await _insert_memory(conn, bank_id, "Alice hikes every weekend.")

            obs1_id = await _insert_observation(conn, bank_id, "Alice hikes often.", [m1])
            obs2_id = await _insert_observation(conn, bank_id, "Alice is outdoorsy.", [m1, m2])

        # Patch out consolidation so it doesn't re-set consolidated_at before we can check it
        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            result = await memory.clear_observations_for_memory(bank_id, str(m1), request_context=request_context)

        assert result["deleted_count"] == 2

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs1_id) not in obs_ids
            assert str(obs2_id) not in obs_ids

            # m1 and m2 should both be reset
            assert await _get_consolidated_at(conn, m1) is None
            assert await _get_consolidated_at(conn, m2) is None

        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Tests: update_document
# ---------------------------------------------------------------------------


async def _insert_document_with_memories(
    conn, bank_id: str, doc_id: str, memories: list[tuple[str, str]]
) -> list[uuid.UUID]:
    """Insert a document and attach memory units to it. Returns list of memory UUIDs."""
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
        mem_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, document_id, created_at, updated_at, consolidated_at)
            VALUES ($1, $2, $3, $4, NOW(), $5, NOW(), NOW(), NOW())
            """,
            mem_id,
            bank_id,
            text,
            fact_type,
            doc_id,
        )
        mem_ids.append(mem_id)
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
            await _insert_document_with_memories(conn, bank_id, doc_id, [("Alice loves hiking.", "experience")])

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
                conn, bank_id, doc_id, [("Alice loves hiking.", "experience"), ("Alice hikes weekly.", "world")]
            )

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.update_document(doc_id, bank_id, tags=["new-tag"], request_context=request_context)

        async with pool.acquire() as conn:
            for mem_id in mem_ids:
                tags = await conn.fetchval("SELECT tags FROM memory_units WHERE id = $1", mem_id)
                assert list(tags) == ["new-tag"], f"Memory unit {mem_id} should have updated tags"

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
                conn, bank_id, doc_id, [("Alice loves hiking.", "experience")]
            )
            obs_id = await _insert_observation(conn, bank_id, "Alice is a hiker.", mem_ids)

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
                conn, bank_id, doc_id, [("Alice loves hiking.", "experience")]
            )
            obs_id = await _insert_observation(conn, bank_id, "Alice is a hiker.", mem_ids)

            # Verify memory starts consolidated
            assert await _get_consolidated_at(conn, mem_ids[0]) is not None

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.update_document(doc_id, bank_id, tags=["new-tag"], request_context=request_context)

        async with pool.acquire() as conn:
            consolidated_at = await _get_consolidated_at(conn, mem_ids[0])
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
                conn, bank_id, doc_id, [("Alice loves hiking.", "experience")]
            )
            await _insert_observation(conn, bank_id, "Alice is a hiker.", mem_ids)

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
            await _insert_document_with_memories(conn, bank_id, doc_id, [("Alice loves hiking.", "experience")])
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
                conn, bank_id, doc_id, [("Alice loves hiking.", "experience")]
            )
            # Unrelated memory from another document — co-sourced in the same observation
            other_mem = await _insert_memory(conn, bank_id, "Alice also rock-climbs.")
            obs_id = await _insert_observation(
                conn, bank_id, "Alice loves outdoor activities.", doc_mem_ids + [other_mem]
            )

            # Verify other_mem starts consolidated
            assert await _get_consolidated_at(conn, other_mem) is not None

        with patch.object(memory, "submit_async_consolidation", new=AsyncMock()):
            await memory.update_document(doc_id, bank_id, tags=["new-tag"], request_context=request_context)

        async with pool.acquire() as conn:
            obs_ids = await _get_observation_ids(conn, bank_id)
            assert str(obs_id) not in obs_ids, "Observation should have been invalidated"

            # other_mem (co-source from another document) must also be reset
            consolidated_at = await _get_consolidated_at(conn, other_mem)
            assert consolidated_at is None, "Co-source memory from other document should be reset"

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
            mem_ids = await _insert_document_with_memories(
                conn, bank_id, doc_id, [("Alice loves hiking.", "experience")]
            )
            # Unrelated memory not in the document
            unrelated = await _insert_memory(conn, bank_id, "Bob likes cycling.")
            unrelated_obs_id = await _insert_observation(conn, bank_id, "Bob is a cyclist.", [unrelated])

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
        from hindsight_api.engine.consolidation.consolidator import _create_observation_directly

        bank_id = f"test-race-create-filter-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            live = await _insert_memory(conn, bank_id, "Alice loves hiking.")
            dead = uuid.uuid4()  # never existed — stands in for a concurrently deleted source

            result = await _create_observation_directly(
                conn=conn,
                memory_engine=memory,
                bank_id=bank_id,
                source_memory_ids=[live, dead],
                observation_text="Alice enjoys hiking regularly.",
            )

            assert result["action"] == "created"
            stored = await conn.fetchval(
                "SELECT source_memory_ids FROM memory_units WHERE id = $1",
                uuid.UUID(result["observation_id"]),
            )
            stored_set = {str(s) for s in stored}
            assert str(live) in stored_set
            assert str(dead) not in stored_set, "Deleted source must not appear in stored observation"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_create_observation_skipped_when_all_sources_deleted(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        from hindsight_api.engine.consolidation.consolidator import _create_observation_directly

        bank_id = f"test-race-create-skip-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            result = await _create_observation_directly(
                conn=conn,
                memory_engine=memory,
                bank_id=bank_id,
                source_memory_ids=[uuid.uuid4(), uuid.uuid4()],
                observation_text="All sources gone.",
            )

            assert result["action"] == "skipped"
            assert result["reason"] == "sources_deleted"

            obs_ids = await _get_observation_ids(conn, bank_id)
            assert obs_ids == [], "No observation row should exist"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_update_observation_skipped_when_all_new_sources_deleted(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        from hindsight_api.engine.consolidation.consolidator import _execute_update_action
        from hindsight_api.engine.response_models import MemoryFact

        bank_id = f"test-race-update-skip-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            original_source = await _insert_memory(conn, bank_id, "Alice hikes.")
            obs_id = await _insert_observation(conn, bank_id, "Alice is a hiker.", [original_source])
            original_text = "Alice is a hiker."

            observation_model = MemoryFact(
                id=str(obs_id),
                text=original_text,
                fact_type="observation",
                source_fact_ids=[str(original_source)],
                tags=[],
            )

            await _execute_update_action(
                conn=conn,
                memory_engine=memory,
                bank_id=bank_id,
                source_memory_ids=[uuid.uuid4(), uuid.uuid4()],  # all dead
                observation_id=str(obs_id),
                new_text="This update must not land.",
                observations=[observation_model],
            )

            row = await conn.fetchrow("SELECT text, source_memory_ids FROM memory_units WHERE id = $1", obs_id)
            assert row["text"] == original_text, "Observation text must not change"
            stored_sources = {str(s) for s in row["source_memory_ids"]}
            assert stored_sources == {str(original_source)}, "Dead sources must not be appended"

        await memory.delete_bank(bank_id, request_context=request_context)

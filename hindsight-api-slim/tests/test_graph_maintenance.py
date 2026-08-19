"""Tests for async graph maintenance after delete.

These tests bypass the LLM-backed retain pipeline by inserting memory_units,
memory_links, entities, and unit_entities directly. That gives precise
control over the graph state so we can assert exact behaviour after a
delete + drain.

The fixture's task backend is ``SyncTaskBackend`` (see conftest), so
``submit_async_graph_maintenance`` runs the worker inline — no polling needed.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.graph_maintenance import (
    MAX_SEMANTIC_LINKS_PER_UNIT,
    MAX_TEMPORAL_LINKS_PER_UNIT,
    enqueue_entity_prune_candidates,
    enqueue_relink_victims,
    run_graph_maintenance_job,
)
from hindsight_api.engine.memory_engine import MemoryEngine

# Every test here seeds memory_units / memory_links / entities with raw INSERTs and
# asserts raw link-row counts, as the module docstring says — none of it round-trips
# through the store, so a backend that keeps those rows outside SQL sees an empty graph.
pytestmark = pytest.mark.memory_backend_incompatible


async def _ensure_bank(memory: MemoryEngine, bank_id: str, request_context: RequestContext) -> None:
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)


async def _insert_unit(
    conn,
    bank_id: str,
    text: str,
    event_date: datetime | None = None,
    fact_type: str = "experience",
    embedding: list[float] | None = None,
) -> uuid.UUID:
    """Insert a memory unit directly. Embedding defaults to NULL (fine for temporal tests); pass
    a 384-dim vector to make the unit a candidate for the semantic top-up pass."""
    mem_id = uuid.uuid4()
    emb = str(embedding) if embedding is not None else None
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, event_date, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5::vector, $6, NOW(), NOW())
        """,
        mem_id,
        bank_id,
        text,
        fact_type,
        emb,
        event_date or datetime.now(UTC),
    )
    return mem_id


async def _insert_link(
    conn,
    bank_id: str,
    from_id: uuid.UUID,
    to_id: uuid.UUID,
    link_type: str = "temporal",
    weight: float = 0.5,
) -> None:
    await conn.execute(
        """
        INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, bank_id)
        VALUES ($1, $2, $3, $4, $5)
        """,
        from_id,
        to_id,
        link_type,
        weight,
        bank_id,
    )


async def _insert_entity(conn, bank_id: str, name: str) -> uuid.UUID:
    """Insert an entity row directly. Returns its UUID."""
    entity_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO entities (id, bank_id, canonical_name, first_seen, last_seen, mention_count)
        VALUES ($1, $2, $3, NOW(), NOW(), 1)
        """,
        entity_id,
        bank_id,
        name,
    )
    return entity_id


async def _link_unit_entity(conn, unit_id: uuid.UUID, entity_id: uuid.UUID) -> None:
    await conn.execute(
        "INSERT INTO unit_entities (unit_id, entity_id) VALUES ($1, $2)",
        unit_id,
        entity_id,
    )


async def _insert_cooccurrence(conn, entity_a: uuid.UUID, entity_b: uuid.UUID, count: int = 1) -> None:
    # entity_cooccurrence_order_check enforces entity_id_1 < entity_id_2 (canonical
    # ordering avoids storing (A,B) and (B,A) as two rows). Sort before insert so
    # callers don't have to care about argument order. Python uuid.UUID compares
    # by .int, matching PostgreSQL's binary uuid ordering.
    first, second = sorted([entity_a, entity_b])
    await conn.execute(
        """
        INSERT INTO entity_cooccurrences (entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred)
        VALUES ($1, $2, $3, NOW())
        """,
        first,
        second,
        count,
    )


async def _insert_document(conn, bank_id: str, doc_id: str) -> None:
    await conn.execute(
        """
        INSERT INTO documents (id, bank_id, original_text, content_hash)
        VALUES ($1, $2, $3, $4)
        """,
        doc_id,
        bank_id,
        f"text-for-{doc_id}",
        doc_id,
    )


async def _attach_unit_to_doc(conn, unit_id: uuid.UUID, doc_id: str) -> None:
    await conn.execute("UPDATE memory_units SET document_id = $1 WHERE id = $2", doc_id, unit_id)


async def _queue_unit_ids(conn, bank_id: str) -> list[str]:
    rows = await conn.fetch(
        "SELECT unit_id FROM graph_maintenance_queue WHERE bank_id = $1 ORDER BY unit_id",
        bank_id,
    )
    return [str(r["unit_id"]) for r in rows]


async def _queue_entity_ids(conn, bank_id: str) -> list[str]:
    rows = await conn.fetch(
        "SELECT entity_id FROM entity_maintenance_queue WHERE bank_id = $1 ORDER BY entity_id",
        bank_id,
    )
    return [str(r["entity_id"]) for r in rows]


async def _seed_entity_candidates(conn, bank_id: str, entity_ids: list[uuid.UUID]) -> None:
    """Queue entities as prune candidates without going through a delete.

    ``enqueue_entity_prune_candidates`` reads its candidates out of
    ``unit_entities``, so it cannot name an entity that is *already* orphaned —
    the right contract for the delete paths, which run before the postings go,
    but awkward for a test that wants to start from an orphan. This writes the
    queue row directly instead.
    """
    await conn.executemany(
        "INSERT INTO entity_maintenance_queue (bank_id, entity_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        [(bank_id, eid) for eid in entity_ids],
    )


# ---------------------------------------------------------------------------
# enqueue_relink_victims
# ---------------------------------------------------------------------------


class TestEnqueueRelinkVictims:
    @pytest.mark.asyncio
    async def test_enqueues_units_with_outgoing_link_to_deleted(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        bank_id = f"test-gm-enq-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            doomed = await _insert_unit(conn, bank_id, "doomed")
            survivor = await _insert_unit(conn, bank_id, "survivor")
            # survivor → doomed (temporal). When doomed dies, survivor needs top-up.
            await _insert_link(conn, bank_id, survivor, doomed, "temporal")

            backend = await memory._get_backend()
            async with conn.transaction():
                count = await enqueue_relink_victims(conn, bank_id, [str(doomed)])

            assert count == 1
            assert await _queue_unit_ids(conn, bank_id) == [str(survivor)]

    @pytest.mark.asyncio
    async def test_excludes_deleted_units_themselves(self, memory: MemoryEngine, request_context: RequestContext):
        """A unit being deleted that linked TO another deleted unit must not enqueue itself."""
        bank_id = f"test-gm-self-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            a = await _insert_unit(conn, bank_id, "a")
            b = await _insert_unit(conn, bank_id, "b")
            await _insert_link(conn, bank_id, a, b, "temporal")
            await _insert_link(conn, bank_id, b, a, "temporal")

            backend = await memory._get_backend()
            async with conn.transaction():
                # Both a and b are being deleted — neither should be enqueued.
                count = await enqueue_relink_victims(conn, bank_id, [str(a), str(b)])

            assert count == 0
            assert await _queue_unit_ids(conn, bank_id) == []

    @pytest.mark.asyncio
    async def test_skips_entity_links(self, memory: MemoryEngine, request_context: RequestContext):
        """Entity links are being removed from the product — we don't enqueue for them."""
        bank_id = f"test-gm-ent-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            doomed = await _insert_unit(conn, bank_id, "doomed")
            survivor = await _insert_unit(conn, bank_id, "survivor")
            # Only an entity link — should NOT trigger enqueue.
            await _insert_link(conn, bank_id, survivor, doomed, "entity")

            backend = await memory._get_backend()
            async with conn.transaction():
                count = await enqueue_relink_victims(conn, bank_id, [str(doomed)])

            assert count == 0

    @pytest.mark.asyncio
    async def test_dedupes_via_on_conflict(self, memory: MemoryEngine, request_context: RequestContext):
        bank_id = f"test-gm-dup-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            doomed1 = await _insert_unit(conn, bank_id, "doomed1")
            doomed2 = await _insert_unit(conn, bank_id, "doomed2")
            survivor = await _insert_unit(conn, bank_id, "survivor")
            # Same survivor linked to two different doomed units across two
            # logical delete batches — should land in the queue only once.
            await _insert_link(conn, bank_id, survivor, doomed1, "temporal")
            await _insert_link(conn, bank_id, survivor, doomed2, "semantic")

            backend = await memory._get_backend()
            async with conn.transaction():
                await enqueue_relink_victims(conn, bank_id, [str(doomed1)])
                await enqueue_relink_victims(conn, bank_id, [str(doomed2)])

            assert await _queue_unit_ids(conn, bank_id) == [str(survivor)]

    @pytest.mark.asyncio
    async def test_include_affected_is_opt_in(self, memory: MemoryEngine, request_context: RequestContext):
        """Without ``include_affected_units`` an affected unit that survives is not enqueued —
        the default (a delete) only needs the victims that pointed AT it."""
        bank_id = f"test-gm-optin-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            edited = await _insert_unit(conn, bank_id, "edited")
            async with conn.transaction():
                count = await enqueue_relink_victims(conn, bank_id, [str(edited)])

            assert count == 0
            assert await _queue_unit_ids(conn, bank_id) == []

    @pytest.mark.asyncio
    async def test_include_affected_enqueues_self_without_victims(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Regression for #2889: an edit that strips a unit's own links but leaves it live must
        still enqueue the unit, even when nothing pointed at it — otherwise the edit is a silent
        no-op and the unit's outgoing adjacency is never rebuilt."""
        bank_id = f"test-gm-self-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            edited = await _insert_unit(conn, bank_id, "edited")
            async with conn.transaction():
                count = await enqueue_relink_victims(conn, bank_id, [str(edited)], include_affected_units=True)

            assert count == 1
            assert await _queue_unit_ids(conn, bank_id) == [str(edited)]

    @pytest.mark.asyncio
    async def test_include_affected_combines_self_and_victims_in_one_insert(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Regression for the ordered-lock deadlock (#2529/#2534): self and its victims go in as
        one enqueue so the queue's sorted insert ordering is preserved across the whole set — two
        separate inserts could take the per-row locks in opposing orders and deadlock."""
        bank_id = f"test-gm-combined-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            edited = await _insert_unit(conn, bank_id, "edited")
            survivor = await _insert_unit(conn, bank_id, "survivor")
            # survivor → edited, so survivor is a victim; edited is enqueued too via the opt-in.
            await _insert_link(conn, bank_id, survivor, edited, "semantic")
            async with conn.transaction():
                count = await enqueue_relink_victims(conn, bank_id, [str(edited)], include_affected_units=True)

            assert count == 2
            # Both present, and the queue is in the sorted order the deadlock-safe insert guarantees.
            assert await _queue_unit_ids(conn, bank_id) == sorted([str(edited), str(survivor)])

    @pytest.mark.asyncio
    async def test_semantic_topup_uses_configured_link_threshold(
        self, memory: MemoryEngine, request_context: RequestContext, monkeypatch
    ):
        """The relink pass must probe for semantic neighbours at the CONFIGURED similarity floor
        (``config.semantic_link_min_similarity``), not an implicit default — otherwise topped-up
        links diverge from the ones retain would have created."""
        from hindsight_api.config import get_config
        from hindsight_api.engine.memories.pg import graph as pg_graph

        captured: dict[str, float] = {}

        async def _capture(conn, bank_id, seed_ids, seed_embs, *, fact_types=None, threshold, **kwargs):
            captured["threshold"] = threshold
            return []

        monkeypatch.setattr(pg_graph, "compute_semantic_links_ann", _capture)

        bank_id = f"test-gm-thresh-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            # A world/experience unit with an embedding and no semantic links → a semantic top-up
            # candidate, so the relink pass reaches compute_semantic_links_ann.
            unit = await _insert_unit(conn, bank_id, "needs topup", embedding=[0.1] * 384)
            async with conn.transaction():
                await enqueue_relink_victims(conn, bank_id, [str(unit)], include_affected_units=True)

        await run_graph_maintenance_job(memory, bank_id, request_context)

        assert captured.get("threshold") == get_config().semantic_link_min_similarity


# ---------------------------------------------------------------------------
# delete_document hook
# ---------------------------------------------------------------------------


class TestDeleteDocumentEnqueue:
    @pytest.mark.asyncio
    async def test_delete_document_enqueues_cross_doc_victims(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        bank_id = f"test-gm-doc-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await _insert_document(conn, bank_id, "doc-A")
            await _insert_document(conn, bank_id, "doc-B")
            doomed = await _insert_unit(conn, bank_id, "in doc A")
            survivor = await _insert_unit(conn, bank_id, "in doc B")
            await _attach_unit_to_doc(conn, doomed, "doc-A")
            await _attach_unit_to_doc(conn, survivor, "doc-B")
            await _insert_link(conn, bank_id, survivor, doomed, "temporal")

        await memory.delete_document("doc-A", bank_id, request_context=request_context)

        async with pool.acquire() as conn:
            # The synchronous task backend means the worker already drained the
            # queue before delete_document returned — assert end-state, not the
            # intermediate enqueue. Queue should be empty.
            assert await _queue_unit_ids(conn, bank_id) == []

    @pytest.mark.asyncio
    async def test_delete_isolated_document_prunes_its_entities(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """#3196: an isolated document has no inbound links, so the delete enqueues
        zero relink victims. The job must still be submitted — its bank-wide orphan
        sweep is what reclaims the entities the deleted units were the only
        reference for."""
        bank_id = f"test-gm-solo-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await _insert_document(conn, bank_id, "doc-solo")
            unit = await _insert_unit(conn, bank_id, "the only unit in the bank")
            await _attach_unit_to_doc(conn, unit, "doc-solo")
            entity = await _insert_entity(conn, bank_id, "solo-entity")
            await _link_unit_entity(conn, unit, entity)

        await memory.delete_document("doc-solo", bank_id, request_context=request_context)

        async with pool.acquire() as conn:
            # Nothing was ever enqueued: this delete would have short-circuited
            # on `no_work` before the fix.
            assert await _queue_unit_ids(conn, bank_id) == []
            remaining = await conn.fetchval("SELECT COUNT(*) FROM entities WHERE bank_id = $1", bank_id)
            assert remaining == 0


class TestSubmitForceSweep:
    @pytest.mark.asyncio
    async def test_empty_queues_short_circuit_by_default(self, memory: MemoryEngine, request_context: RequestContext):
        """The default stays cheap for unconditional callers (every retain)."""
        bank_id = f"test-gm-nowork-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        result = await memory.submit_async_graph_maintenance(bank_id=bank_id, request_context=request_context)

        assert result == {"operation_id": None, "no_work": True}

    @pytest.mark.asyncio
    async def test_queued_entity_candidate_defeats_the_short_circuit(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """The pre-check reads BOTH queues. A delete that enqueued only entity
        candidates (an isolated document has no relink victims) must still get a
        job — checking only graph_maintenance_queue would short-circuit it away
        and leave the entities stranded."""
        bank_id = f"test-gm-entq-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            orphan = await _insert_entity(conn, bank_id, "unreferenced")
            await _seed_entity_candidates(conn, bank_id, [orphan])
            assert await _queue_unit_ids(conn, bank_id) == []

        result = await memory.submit_async_graph_maintenance(bank_id=bank_id, request_context=request_context)

        assert result.get("no_work") is not True
        assert result["operation_id"]

        async with pool.acquire() as conn:
            # SyncTaskBackend ran the job inline: the orphan is gone.
            assert await conn.fetchval("SELECT 1 FROM entities WHERE id = $1", orphan) is None

    @pytest.mark.asyncio
    async def test_force_sweep_submits_on_empty_queues(self, memory: MemoryEngine, request_context: RequestContext):
        """`force_sweep=True` still submits with both queues empty, for callers
        that would rather pay for an empty job than reason about the queues."""
        bank_id = f"test-gm-force-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        result = await memory.submit_async_graph_maintenance(
            bank_id=bank_id, request_context=request_context, force_sweep=True
        )

        assert result.get("no_work") is not True
        assert result["operation_id"]


# ---------------------------------------------------------------------------
# Relink pass (Pass 1)
# ---------------------------------------------------------------------------


class TestRelinkPass:
    @pytest.mark.asyncio
    async def test_drains_empty_queue_cleanly(self, memory: MemoryEngine, request_context: RequestContext):
        bank_id = f"test-gm-empty-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        result = await run_graph_maintenance_job(memory, bank_id, request_context)
        assert result == {
            "relink_units_processed": 0,
            "relink_links_added": 0,
            "entities_examined": 0,
            "orphan_entities_pruned": 0,
            "stale_cooccurrences_pruned": 0,
            "queues_drained": True,
        }

    @pytest.mark.asyncio
    async def test_skips_missing_unit_silently(self, memory: MemoryEngine, request_context: RequestContext):
        """Unit deleted between enqueue and drain: worker dequeues and no-ops."""
        bank_id = f"test-gm-miss-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            # Enqueue a unit_id that doesn't exist in memory_units.
            await conn.execute(
                "INSERT INTO graph_maintenance_queue (bank_id, unit_id) VALUES ($1, $2)",
                bank_id,
                uuid.uuid4(),
            )

        result = await run_graph_maintenance_job(memory, bank_id, request_context)
        assert result["relink_units_processed"] == 1
        assert result["relink_links_added"] == 0

        async with pool.acquire() as conn:
            assert await _queue_unit_ids(conn, bank_id) == []

    @pytest.mark.asyncio
    async def test_tops_up_temporal_when_under_cap(self, memory: MemoryEngine, request_context: RequestContext):
        """A victim under the temporal cap gets new outgoing links to neighbours
        that were never linked at retain time."""
        bank_id = f"test-gm-topup-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        # Build: one victim at t=0, 2 already-linked neighbours, and 5 unlinked
        # neighbours all within the 24h window. After top-up the victim should
        # have outgoing temporal links to all 7.
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            base = datetime.now(UTC).replace(microsecond=0)
            victim = await _insert_unit(conn, bank_id, "victim", event_date=base)

            already_linked = [
                await _insert_unit(conn, bank_id, f"linked-{i}", event_date=base + timedelta(minutes=i + 1))
                for i in range(2)
            ]
            for _ in range(5):
                await _insert_unit(conn, bank_id, "unlinked", event_date=base + timedelta(minutes=30))

            for nbr in already_linked:
                await _insert_link(conn, bank_id, victim, nbr, "temporal")

            await conn.execute(
                "INSERT INTO graph_maintenance_queue (bank_id, unit_id) VALUES ($1, $2)",
                bank_id,
                victim,
            )

        result = await run_graph_maintenance_job(memory, bank_id, request_context)
        assert result["relink_units_processed"] == 1
        # We probed for up to MAX_TEMPORAL_LINKS_PER_UNIT neighbours; bulk insert
        # is ON CONFLICT DO NOTHING, so the already-linked 2 are silently
        # skipped at insert time. The probe still returned them, so
        # relink_links_added counts what we attempted to insert, not what
        # actually landed. Verify the end-state via the DB instead.
        assert result["relink_links_added"] >= 5

        async with pool.acquire() as conn:
            outgoing = await conn.fetchval(
                """
                SELECT COUNT(*) FROM memory_links
                WHERE from_unit_id = $1 AND bank_id = $2 AND link_type = 'temporal'
                """,
                victim,
                bank_id,
            )
            # 2 originals + 5 new = 7 distinct outgoing temporal links.
            assert outgoing == 7

            assert await _queue_unit_ids(conn, bank_id) == []

    @pytest.mark.asyncio
    async def test_no_topup_when_victim_at_cap(self, memory: MemoryEngine, request_context: RequestContext):
        """If the victim already has cap links, probing is skipped."""
        bank_id = f"test-gm-atcap-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            base = datetime.now(UTC).replace(microsecond=0)
            victim = await _insert_unit(conn, bank_id, "victim", event_date=base)

            # Insert exactly cap temporal links from victim, plus extra unlinked
            # candidates. Probe should be skipped because victim is at cap.
            for i in range(MAX_TEMPORAL_LINKS_PER_UNIT):
                nbr = await _insert_unit(conn, bank_id, f"l-{i}", event_date=base + timedelta(minutes=i + 1))
                await _insert_link(conn, bank_id, victim, nbr, "temporal")

            # Plus extras that would be valid candidates if we DID probe.
            for i in range(3):
                await _insert_unit(conn, bank_id, f"x-{i}", event_date=base + timedelta(minutes=i + 100))

            await conn.execute(
                "INSERT INTO graph_maintenance_queue (bank_id, unit_id) VALUES ($1, $2)",
                bank_id,
                victim,
            )

        result = await run_graph_maintenance_job(memory, bank_id, request_context)
        assert result["relink_units_processed"] == 1

        async with pool.acquire() as conn:
            outgoing = await conn.fetchval(
                """
                SELECT COUNT(*) FROM memory_links
                WHERE from_unit_id = $1 AND link_type = 'temporal'
                """,
                victim,
            )
            assert outgoing == MAX_TEMPORAL_LINKS_PER_UNIT


# ---------------------------------------------------------------------------
# Orphan entity prune (Pass 2)
# ---------------------------------------------------------------------------


class TestOrphanEntityPrune:
    @pytest.mark.asyncio
    async def test_prunes_queued_entities_with_no_unit_references(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """A queued candidate with zero unit_entities rows is an orphan and gets
        deleted; a queued candidate that is still referenced is kept."""
        bank_id = f"test-gm-orphan-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            referenced = await _insert_entity(conn, bank_id, "referenced")
            orphan_a = await _insert_entity(conn, bank_id, "orphan_a")
            orphan_b = await _insert_entity(conn, bank_id, "orphan_b")

            unit = await _insert_unit(conn, bank_id, "with-entity")
            await _link_unit_entity(conn, unit, referenced)
            # All three are candidates: the drain decides which are actually
            # dead. Over-enqueueing is the safe direction.
            await _seed_entity_candidates(conn, bank_id, [referenced, orphan_a, orphan_b])

        result = await run_graph_maintenance_job(memory, bank_id, request_context)
        assert result["entities_examined"] == 3
        assert result["orphan_entities_pruned"] == 2

        async with pool.acquire() as conn:
            survivors = await conn.fetch("SELECT id FROM entities WHERE bank_id = $1 ORDER BY id", bank_id)
            assert {str(r["id"]) for r in survivors} == {str(referenced)}
            # The claim removed every queue row it processed.
            assert await _queue_entity_ids(conn, bank_id) == []

    @pytest.mark.asyncio
    async def test_ignores_unqueued_orphans(self, memory: MemoryEngine, request_context: RequestContext):
        """The pass is queue-driven, not a sweep: an orphan nothing enqueued is
        left alone. This is the whole point of #3222 — the cost of a run tracks
        what a delete touched, not the size of the bank."""
        bank_id = f"test-gm-unqueued-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            orphan = await _insert_entity(conn, bank_id, "never-queued")

        result = await run_graph_maintenance_job(memory, bank_id, request_context)
        assert result["entities_examined"] == 0
        assert result["orphan_entities_pruned"] == 0

        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT 1 FROM entities WHERE id = $1", orphan) == 1

    @pytest.mark.asyncio
    async def test_does_not_touch_other_banks(self, memory: MemoryEngine, request_context: RequestContext):
        """The prune is scoped by bank — a candidate queued under another bank's
        id must not be claimed, even though entity ids are globally unique."""
        bank_a = f"test-gm-scopea-{uuid.uuid4().hex[:8]}"
        bank_b = f"test-gm-scopeb-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_a, request_context)
        await _ensure_bank(memory, bank_b, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            orphan_in_a = await _insert_entity(conn, bank_a, "orphan_a")
            orphan_in_b = await _insert_entity(conn, bank_b, "orphan_b")
            await _seed_entity_candidates(conn, bank_a, [orphan_in_a])
            await _seed_entity_candidates(conn, bank_b, [orphan_in_b])

        await run_graph_maintenance_job(memory, bank_a, request_context)

        async with pool.acquire() as conn:
            # b's orphan must still exist — the drain was scoped to a.
            assert await conn.fetchval("SELECT 1 FROM entities WHERE id = $1", orphan_in_b) == 1
            assert await _queue_entity_ids(conn, bank_b) == [str(orphan_in_b)]
            # a's orphan is gone.
            assert await conn.fetchval("SELECT 1 FROM entities WHERE id = $1", orphan_in_a) is None


# ---------------------------------------------------------------------------
# Stale cooccurrence prune (Pass 3)
# ---------------------------------------------------------------------------


class TestStaleCooccurrencePrune:
    @pytest.mark.asyncio
    async def test_prunes_cooccurrence_with_no_shared_unit(self, memory: MemoryEngine, request_context: RequestContext):
        """Both entities still exist but no unit references both of them — the
        cooccurrence row is stale and should be pruned."""
        bank_id = f"test-gm-cocc-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            ent_a = await _insert_entity(conn, bank_id, "alice")
            ent_b = await _insert_entity(conn, bank_id, "bob")
            # Cooccurrence row records that A and B were observed together.
            await _insert_cooccurrence(conn, ent_a, ent_b, count=5)
            # Both entities still have references — but to DIFFERENT units, so
            # no current unit witnesses both A and B together.
            unit_a = await _insert_unit(conn, bank_id, "with_a")
            unit_b = await _insert_unit(conn, bank_id, "with_b")
            await _link_unit_entity(conn, unit_a, ent_a)
            await _link_unit_entity(conn, unit_b, ent_b)
            await _seed_entity_candidates(conn, bank_id, [ent_a])

        result = await run_graph_maintenance_job(memory, bank_id, request_context)
        assert result["stale_cooccurrences_pruned"] == 1
        # Both entities still exist — they weren't orphans.
        assert result["orphan_entities_pruned"] == 0

        async with pool.acquire() as conn:
            # Match canonical ordering enforced by entity_cooccurrence_order_check.
            first, second = sorted([ent_a, ent_b])
            remaining = await conn.fetchval(
                "SELECT COUNT(*) FROM entity_cooccurrences WHERE entity_id_1 = $1 AND entity_id_2 = $2",
                first,
                second,
            )
            assert remaining == 0

    @pytest.mark.asyncio
    async def test_keeps_cooccurrence_with_shared_unit(self, memory: MemoryEngine, request_context: RequestContext):
        """If at least one unit still references both entities, the cooccurrence
        row stays."""
        bank_id = f"test-gm-keep-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            ent_a = await _insert_entity(conn, bank_id, "alice")
            ent_b = await _insert_entity(conn, bank_id, "bob")
            await _insert_cooccurrence(conn, ent_a, ent_b, count=5)
            # A unit references both — cooccurrence is still grounded.
            unit = await _insert_unit(conn, bank_id, "alice-and-bob")
            await _link_unit_entity(conn, unit, ent_a)
            await _link_unit_entity(conn, unit, ent_b)
            await _seed_entity_candidates(conn, bank_id, [ent_a, ent_b])

        result = await run_graph_maintenance_job(memory, bank_id, request_context)
        assert result["stale_cooccurrences_pruned"] == 0

        async with pool.acquire() as conn:
            # Match canonical ordering enforced by entity_cooccurrence_order_check.
            first, second = sorted([ent_a, ent_b])
            still_there = await conn.fetchval(
                "SELECT cooccurrence_count FROM entity_cooccurrences WHERE entity_id_1 = $1 AND entity_id_2 = $2",
                first,
                second,
            )
            assert still_there == 5

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_prunes_only_the_stale_edge_around_a_hub_and_leaves_other_banks(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """The prune decides staleness against a SET of live pairs (#3367), not
        per-cooccurrence-row. Guard the two things that swap could get wrong: a
        hub's *still-grounded* edges survive while only its genuinely stale edge
        is pruned, and the ``live`` set never reaches across banks.

        The set is now seeded from the claimed candidates' units rather than the
        whole bank (#3222), so this also pins that the narrower seed still sees
        every unit that could ground one of the pairs being judged."""
        bank_id = f"test-gm-hub-{uuid.uuid4().hex[:8]}"
        other_bank = f"test-gm-oth-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)
        await _ensure_bank(memory, other_bank, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            hub = await _insert_entity(conn, bank_id, "hub")
            spoke_live = await _insert_entity(conn, bank_id, "spoke_live")
            spoke_stale = await _insert_entity(conn, bank_id, "spoke_stale")
            # Both edges recorded, but only hub+spoke_live still share a unit.
            await _insert_cooccurrence(conn, hub, spoke_live, count=3)
            await _insert_cooccurrence(conn, hub, spoke_stale, count=7)
            unit_together = await _insert_unit(conn, bank_id, "hub-and-live")
            await _link_unit_entity(conn, unit_together, hub)
            await _link_unit_entity(conn, unit_together, spoke_live)
            # spoke_stale still exists in its own unit (so it is not an orphan),
            # just no longer alongside the hub.
            unit_solo = await _insert_unit(conn, bank_id, "stale-solo")
            await _link_unit_entity(conn, unit_solo, spoke_stale)

            # A different bank with its own stale edge that pruning bank_id must
            # not touch — the scoped ``live`` build must not consider it grounded
            # or stale.
            oth_a = await _insert_entity(conn, other_bank, "oth_a")
            oth_b = await _insert_entity(conn, other_bank, "oth_b")
            await _insert_cooccurrence(conn, oth_a, oth_b, count=2)
            oth_unit_a = await _insert_unit(conn, other_bank, "oth-with-a")
            oth_unit_b = await _insert_unit(conn, other_bank, "oth-with-b")
            await _link_unit_entity(conn, oth_unit_a, oth_a)
            await _link_unit_entity(conn, oth_unit_b, oth_b)

            # Queue-driven now: the hub is the candidate a delete would have
            # enqueued. The other bank's entities are queued under their own
            # bank, which draining bank_id must not claim.
            await _seed_entity_candidates(conn, bank_id, [hub])
            await _seed_entity_candidates(conn, other_bank, [oth_a])

        result = await run_graph_maintenance_job(memory, bank_id, request_context)
        assert result["stale_cooccurrences_pruned"] == 1
        assert result["orphan_entities_pruned"] == 0

        async with pool.acquire() as conn:
            hub_live = sorted([hub, spoke_live])
            hub_stale = sorted([hub, spoke_stale])
            oth = sorted([oth_a, oth_b])
            live_kept = await conn.fetchval(
                "SELECT COUNT(*) FROM entity_cooccurrences WHERE entity_id_1 = $1 AND entity_id_2 = $2",
                hub_live[0],
                hub_live[1],
            )
            stale_gone = await conn.fetchval(
                "SELECT COUNT(*) FROM entity_cooccurrences WHERE entity_id_1 = $1 AND entity_id_2 = $2",
                hub_stale[0],
                hub_stale[1],
            )
            other_untouched = await conn.fetchval(
                "SELECT COUNT(*) FROM entity_cooccurrences WHERE entity_id_1 = $1 AND entity_id_2 = $2",
                oth[0],
                oth[1],
            )
            assert live_kept == 1
            assert stale_gone == 0
            assert other_untouched == 1

    @pytest.mark.asyncio
    async def test_prunes_when_only_the_second_endpoint_is_a_candidate(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Scoping matches a candidate on EITHER endpoint. The two arms are a
        UNION because an OR across entity_id_1/entity_id_2 can't be driven from
        either index (#3387's shape); dropping the entity_id_2 arm would leave
        exactly this row behind, and only for pairs whose surviving candidate
        happens to sort second."""
        bank_id = f"test-gm-cocc2-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            ent_a = await _insert_entity(conn, bank_id, "alice")
            ent_b = await _insert_entity(conn, bank_id, "bob")
            await _insert_cooccurrence(conn, ent_a, ent_b)
            unit_a = await _insert_unit(conn, bank_id, "with_a")
            unit_b = await _insert_unit(conn, bank_id, "with_b")
            await _link_unit_entity(conn, unit_a, ent_a)
            await _link_unit_entity(conn, unit_b, ent_b)
            # Queue only the endpoint stored as entity_id_2.
            _, second = sorted([ent_a, ent_b])
            await _seed_entity_candidates(conn, bank_id, [second])

        result = await run_graph_maintenance_job(memory, bank_id, request_context)
        assert result["stale_cooccurrences_pruned"] == 1


# ---------------------------------------------------------------------------
# Delta enqueue: what a delete captures before the cascade
# ---------------------------------------------------------------------------


class TestEnqueueEntityPruneCandidates:
    @pytest.mark.asyncio
    async def test_enqueues_entities_the_doomed_units_reference(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        bank_id = f"test-gm-cand-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            doomed = await _insert_unit(conn, bank_id, "doomed")
            untouched_unit = await _insert_unit(conn, bank_id, "untouched")
            shared = await _insert_entity(conn, bank_id, "shared")
            only_here = await _insert_entity(conn, bank_id, "only-here")
            elsewhere = await _insert_entity(conn, bank_id, "elsewhere")
            await _link_unit_entity(conn, doomed, shared)
            await _link_unit_entity(conn, doomed, only_here)
            await _link_unit_entity(conn, untouched_unit, shared)
            await _link_unit_entity(conn, untouched_unit, elsewhere)

            async with conn.transaction():
                count = await enqueue_entity_prune_candidates(conn, bank_id, [str(doomed)])

            # Both of the doomed unit's entities are candidates — including the
            # one another unit also references. The drain, not the enqueue,
            # decides which are actually dead.
            assert count == 2
            assert await _queue_entity_ids(conn, bank_id) == sorted([str(shared), str(only_here)])

    @pytest.mark.asyncio
    async def test_dedupes_across_overlapping_deletes(self, memory: MemoryEngine, request_context: RequestContext):
        """Two deletes naming the same entity leave one queue row: the composite
        primary key absorbs the duplicate."""
        bank_id = f"test-gm-canddup-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            unit_a = await _insert_unit(conn, bank_id, "a")
            unit_b = await _insert_unit(conn, bank_id, "b")
            entity = await _insert_entity(conn, bank_id, "shared")
            await _link_unit_entity(conn, unit_a, entity)
            await _link_unit_entity(conn, unit_b, entity)

            async with conn.transaction():
                await enqueue_entity_prune_candidates(conn, bank_id, [str(unit_a)])
                await enqueue_entity_prune_candidates(conn, bank_id, [str(unit_b)])

            assert await _queue_entity_ids(conn, bank_id) == [str(entity)]

    @pytest.mark.asyncio
    async def test_delete_memory_unit_prunes_the_entity_it_was_holding_up(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """End-to-end through the real delete path: the entity whose last posting
        the deleted unit held is gone, the one another unit still references stays."""
        bank_id = f"test-gm-del-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            doomed = await _insert_unit(conn, bank_id, "doomed")
            survivor = await _insert_unit(conn, bank_id, "survivor")
            sole = await _insert_entity(conn, bank_id, "sole-reference")
            shared = await _insert_entity(conn, bank_id, "shared")
            await _link_unit_entity(conn, doomed, sole)
            await _link_unit_entity(conn, doomed, shared)
            await _link_unit_entity(conn, survivor, shared)

        await memory.delete_memory_unit(str(doomed), request_context=request_context)

        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT 1 FROM entities WHERE id = $1", sole) is None
            assert await conn.fetchval("SELECT 1 FROM entities WHERE id = $1", shared) == 1
            # The drain consumed every candidate it enqueued.
            assert await _queue_entity_ids(conn, bank_id) == []


# ---------------------------------------------------------------------------
# Time budget
# ---------------------------------------------------------------------------


class TestEveryDeleteSiteEnqueues:
    """One test per site that removes units or replaces postings.

    A queue-driven prune is only as complete as its producers: a site that
    forgets to enqueue leaks orphan entities silently — no error, no failed
    operation, just garbage that accumulates. These assert the queue row exists
    rather than asserting on the prune, so a missing call fails here loudly
    instead of showing up as unexplained growth on someone's bank.
    """

    @pytest.mark.asyncio
    async def test_delta_retain_chunk_delete(self, memory: MemoryEngine, request_context: RequestContext):
        """``delete_chunks_by_ids`` — the cascade delta retain deletes facts through."""
        from hindsight_api.engine.retain import chunk_storage

        bank_id = f"test-gm-chunkdel-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        backend = await memory._get_backend()
        pool = await memory._get_pool()
        chunk_id = f"chunk-{uuid.uuid4().hex[:8]}"
        async with pool.acquire() as conn:
            await _insert_document(conn, bank_id, "doc-chunked")
            await conn.execute(
                """
                INSERT INTO chunks (chunk_id, document_id, bank_id, chunk_text, chunk_index, content_hash)
                VALUES ($1, $2, $3, 'body', 0, 'hash-0')
                """,
                chunk_id,
                "doc-chunked",
                bank_id,
            )
            unit = await _insert_unit(conn, bank_id, "fact in the chunk", fact_type="world")
            await conn.execute(
                "UPDATE memory_units SET document_id = $1, chunk_id = $2 WHERE id = $3",
                "doc-chunked",
                chunk_id,
                unit,
            )
            entity = await _insert_entity(conn, bank_id, "chunk-entity")
            await _link_unit_entity(conn, unit, entity)

        async with backend.acquire() as conn:
            async with conn.transaction():
                await chunk_storage.delete_chunks_by_ids(conn, [chunk_id], bank_id, ops=backend.ops)

        async with pool.acquire() as conn:
            assert await _queue_entity_ids(conn, bank_id) == [str(entity)]

    @pytest.mark.asyncio
    async def test_document_reingest(self, memory: MemoryEngine, request_context: RequestContext):
        """``handle_document_tracking`` — a full re-ingest replaces the old facts."""
        from hindsight_api.engine.retain.fact_storage import handle_document_tracking

        bank_id = f"test-gm-reingest-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        backend = await memory._get_backend()
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await _insert_document(conn, bank_id, "doc-reingest")
            unit = await _insert_unit(conn, bank_id, "old fact", fact_type="world")
            await _attach_unit_to_doc(conn, unit, "doc-reingest")
            entity = await _insert_entity(conn, bank_id, "reingest-entity")
            await _link_unit_entity(conn, unit, entity)

        async with backend.acquire() as conn:
            async with conn.transaction():
                await handle_document_tracking(
                    conn,
                    bank_id,
                    "doc-reingest",
                    "replacement text",
                    is_first_batch=True,
                    ops=backend.ops,
                )

        async with pool.acquire() as conn:
            assert str(entity) in await _queue_entity_ids(conn, bank_id)

    @pytest.mark.asyncio
    async def test_curation_invalidate(self, memory: MemoryEngine, request_context: RequestContext):
        """Invalidation moves the unit to the archive, taking its postings with it."""
        bank_id = f"test-gm-inval-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            unit = await _insert_unit(conn, bank_id, "to be invalidated", fact_type="world")
            entity = await _insert_entity(conn, bank_id, "invalidated-entity")
            await _link_unit_entity(conn, unit, entity)

        await memory.update_memory_unit(
            bank_id=bank_id,
            memory_id=str(unit),
            state="invalidated",
            reason="test",
            request_context=request_context,
        )

        async with pool.acquire() as conn:
            # SyncTaskBackend drained the queue inline, so assert the outcome:
            # the entity the archived unit was holding up is gone.
            assert await conn.fetchval("SELECT 1 FROM entities WHERE id = $1", entity) is None


class TestTimeBudget:
    @pytest.mark.asyncio
    async def test_expired_budget_leaves_the_queue_for_the_next_run(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """An exhausted budget is not a failure: nothing is claimed, the rows
        stay queued, and the job reports it rather than being cancelled
        mid-statement the way the bank-wide sweep was (#3222)."""
        from hindsight_api.engine.memories import get_memories

        bank_id = f"test-gm-budget-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            orphan = await _insert_entity(conn, bank_id, "orphan")
            await _seed_entity_candidates(conn, bank_id, [orphan])

        backend = await memory._get_backend()
        from hindsight_api.engine.schema import fq_table

        result = await get_memories().entity_prune_pass(
            backend=backend,
            fq_table=fq_table,
            bank_id=bank_id,
            deadline=time.monotonic() - 1,  # already expired
        )

        assert result.queue_exhausted is False
        assert result.entities_examined == 0

        async with pool.acquire() as conn:
            # Still queued, and the entity is untouched: the next run resumes.
            assert await _queue_entity_ids(conn, bank_id) == [str(orphan)]
            assert await conn.fetchval("SELECT 1 FROM entities WHERE id = $1", orphan) == 1

    @pytest.mark.asyncio
    async def test_budget_exhaustion_chains_a_follow_up_run(
        self, memory: MemoryEngine, request_context: RequestContext, monkeypatch
    ):
        """A backlog too big for one run has to schedule its own continuation.

        Otherwise the leftover queue rows sit until the next delete happens to
        trigger a job — which on a bank that has gone quiet is never. The chain
        is suppressed under a synchronous task backend (it would recurse rather
        than schedule), so this drives the real engine method with the guard
        pointed at a stand-in class.
        """
        from hindsight_api.engine import graph_maintenance as gm

        bank_id = f"test-gm-chain-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            orphan = await _insert_entity(conn, bank_id, "orphan")
            await _seed_entity_candidates(conn, bank_id, [orphan])

        # Budget already spent: neither pass claims anything, both report the
        # queue as not exhausted.
        monkeypatch.setattr(gm, "_JOB_TIME_BUDGET_SECONDS", -1.0)

        submitted: list[str] = []

        async def _record_submit(
            *, bank_id: str, request_context, dedupe_excludes_operation_id=None, force_sweep: bool = False
        ):
            submitted.append(bank_id)
            return {"operation_id": None, "no_work": True}

        monkeypatch.setattr(memory, "submit_async_graph_maintenance", _record_submit)

        class _NotSync:
            """Stands in for a queue-backed backend so the guard lets the chain through."""

        monkeypatch.setattr(memory, "_task_backend", _NotSync())

        result = await run_graph_maintenance_job(memory, bank_id, request_context)

        assert result["queues_drained"] is False
        assert submitted == [bank_id]


# ---------------------------------------------------------------------------
# Sanity check on cap values
# ---------------------------------------------------------------------------


def test_caps_match_retain_defaults():
    """If retain bumps its caps but graph_maintenance stays put, top-up will
    silently never reach the retain ceiling — the asserts here exist so a
    future cap change forces a paired update."""
    from hindsight_api.engine.retain.link_utils import MAX_TEMPORAL_LINKS_PER_UNIT as RETAIN_TEMPORAL

    assert MAX_TEMPORAL_LINKS_PER_UNIT == RETAIN_TEMPORAL
    assert MAX_SEMANTIC_LINKS_PER_UNIT == 50  # mirrors compute_semantic_links_ann's top_k default

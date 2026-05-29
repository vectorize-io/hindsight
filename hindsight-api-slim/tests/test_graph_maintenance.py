"""Tests for async graph maintenance after delete.

These tests bypass the LLM-backed retain pipeline by inserting memory_units,
memory_links, entities, and unit_entities directly. That gives precise
control over the graph state so we can assert exact behaviour after a
delete + drain.

The fixture's task backend is ``SyncTaskBackend`` (see conftest), so
``submit_async_graph_maintenance`` runs the worker inline — no polling needed.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.graph_maintenance import (
    MAX_SEMANTIC_LINKS_PER_UNIT,
    MAX_TEMPORAL_LINKS_PER_UNIT,
    enqueue_relink_victims,
    run_graph_maintenance_job,
)
from hindsight_api.engine.memory_engine import MemoryEngine


async def _ensure_bank(memory: MemoryEngine, bank_id: str, request_context: RequestContext) -> None:
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)


async def _insert_unit(
    conn,
    bank_id: str,
    text: str,
    event_date: datetime | None = None,
    fact_type: str = "experience",
) -> uuid.UUID:
    """Insert a memory unit directly. Skips embedding (NULL is fine for temporal tests)."""
    mem_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
        """,
        mem_id,
        bank_id,
        text,
        fact_type,
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
                count = await enqueue_relink_victims(conn, bank_id, [str(doomed)], ops=backend.ops)

            assert count == 1
            assert await _queue_unit_ids(conn, bank_id) == [str(survivor)]

    @pytest.mark.asyncio
    async def test_excludes_deleted_units_themselves(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
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
                count = await enqueue_relink_victims(conn, bank_id, [str(a), str(b)], ops=backend.ops)

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
                count = await enqueue_relink_victims(conn, bank_id, [str(doomed)], ops=backend.ops)

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
                await enqueue_relink_victims(conn, bank_id, [str(doomed1)], ops=backend.ops)
                await enqueue_relink_victims(conn, bank_id, [str(doomed2)], ops=backend.ops)

            assert await _queue_unit_ids(conn, bank_id) == [str(survivor)]


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


# ---------------------------------------------------------------------------
# Relink pass (Pass 1)
# ---------------------------------------------------------------------------


class TestRelinkPass:
    @pytest.mark.asyncio
    async def test_drains_empty_queue_cleanly(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        bank_id = f"test-gm-empty-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        result = await run_graph_maintenance_job(memory, bank_id, request_context)
        assert result == {
            "relink_units_processed": 0,
            "relink_links_added": 0,
            "orphan_entities_pruned": 0,
            "stale_cooccurrences_pruned": 0,
        }

    @pytest.mark.asyncio
    async def test_skips_missing_unit_silently(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
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
    async def test_tops_up_temporal_when_under_cap(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
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
    async def test_no_topup_when_victim_at_cap(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
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
    async def test_prunes_entities_with_no_unit_references(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """An entity with zero unit_entities rows is an orphan and should be
        deleted by the sweep."""
        bank_id = f"test-gm-orphan-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            referenced = await _insert_entity(conn, bank_id, "referenced")
            orphan_a = await _insert_entity(conn, bank_id, "orphan_a")
            orphan_b = await _insert_entity(conn, bank_id, "orphan_b")

            unit = await _insert_unit(conn, bank_id, "with-entity")
            await _link_unit_entity(conn, unit, referenced)

        result = await run_graph_maintenance_job(memory, bank_id, request_context)
        assert result["orphan_entities_pruned"] == 2

        async with pool.acquire() as conn:
            survivors = await conn.fetch(
                "SELECT id FROM entities WHERE bank_id = $1 ORDER BY id", bank_id
            )
            survivor_ids = {str(r["id"]) for r in survivors}
            assert survivor_ids == {str(referenced)}
            # Confirm orphans are gone.
            for orphan in (orphan_a, orphan_b):
                assert orphan not in survivor_ids

    @pytest.mark.asyncio
    async def test_does_not_touch_other_banks(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """The sweep is scoped by bank — orphan entities in OTHER banks
        must not be touched."""
        bank_a = f"test-gm-scopea-{uuid.uuid4().hex[:8]}"
        bank_b = f"test-gm-scopeb-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_a, request_context)
        await _ensure_bank(memory, bank_b, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            orphan_in_a = await _insert_entity(conn, bank_a, "orphan_a")
            orphan_in_b = await _insert_entity(conn, bank_b, "orphan_b")

        await run_graph_maintenance_job(memory, bank_a, request_context)

        async with pool.acquire() as conn:
            # b's orphan must still exist — the sweep was scoped to a.
            still_in_b = await conn.fetchval("SELECT 1 FROM entities WHERE id = $1", orphan_in_b)
            assert still_in_b == 1
            # a's orphan is gone.
            still_in_a = await conn.fetchval("SELECT 1 FROM entities WHERE id = $1", orphan_in_a)
            assert still_in_a is None


# ---------------------------------------------------------------------------
# Stale cooccurrence prune (Pass 3)
# ---------------------------------------------------------------------------


class TestStaleCooccurrencePrune:
    @pytest.mark.asyncio
    async def test_prunes_cooccurrence_with_no_shared_unit(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
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
    async def test_keeps_cooccurrence_with_shared_unit(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
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

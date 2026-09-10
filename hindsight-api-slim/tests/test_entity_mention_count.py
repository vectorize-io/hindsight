"""``entities.mention_count`` comes back down when the mentions go (#4291).

The counter is written once per mention at retain time and used for prominence:
it orders the entity list, sizes the graph nodes, and is returned to callers. It
used to be increment-only, so replacing or deleting a document left every entity
it named inflated — and the drift was proportional to how often the bank's
documents were rewritten, not to anything about the entities.

The decrement lives at the one choke point every unlink path already goes
through, :func:`enqueue_entity_prune_candidates`, which is why the tests below
drive it there and at the two API-level paths that reach it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.db.ops_oracle import OracleOps
from hindsight_api.engine.graph_maintenance import enqueue_entity_prune_candidates
from hindsight_api.engine.memory_engine import MemoryEngine

# Every test seeds `entities` / `unit_entities` with raw INSERTs and asserts on raw
# `mention_count` values, so a store that keeps its entity registry outside SQL has
# nothing here to look at.
pytestmark = pytest.mark.memory_backend_incompatible


async def _ensure_bank(memory: MemoryEngine, bank_id: str, request_context: RequestContext) -> None:
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)


async def _insert_entity(conn, bank_id: str, name: str, mention_count: int = 1) -> uuid.UUID:
    entity_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO entities (id, bank_id, canonical_name, first_seen, last_seen, mention_count)
        VALUES ($1, $2, $3, NOW(), NOW(), $4)
        """,
        entity_id,
        bank_id,
        name,
        mention_count,
    )
    return entity_id


async def _insert_unit(conn, bank_id: str, text: str) -> uuid.UUID:
    unit_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, created_at, updated_at)
        VALUES ($1, $2, $3, 'experience', $4, NOW(), NOW())
        """,
        unit_id,
        bank_id,
        text,
        datetime.now(UTC),
    )
    return unit_id


async def _link(conn, unit_id: uuid.UUID, entity_id: uuid.UUID) -> None:
    await conn.execute("INSERT INTO unit_entities (unit_id, entity_id) VALUES ($1, $2)", unit_id, entity_id)


async def _mention_count(conn, entity_id: uuid.UUID) -> int:
    return await conn.fetchval("SELECT mention_count FROM entities WHERE id = $1", entity_id)


async def _posting_count(conn, entity_id: uuid.UUID) -> int:
    return await conn.fetchval("SELECT COUNT(*) FROM unit_entities WHERE entity_id = $1", entity_id)


# ---------------------------------------------------------------------------
# SQL shape (no database)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oracle_release_reads_the_postings_once_and_queues_before_it_debits():
    """Oracle adapter: one read of `unit_entities` feeds both writes, the queue
    MERGE runs before the entity UPDATE (the order the drain takes those locks,
    so a delete cannot cycle against a worker), and each entity is debited by
    its own posting count, bound in sorted id order."""
    ops = OracleOps()
    conn = AsyncMock()
    id_a = "00000000-0000-0000-0000-00000000000a"
    id_b = "00000000-0000-0000-0000-00000000000b"
    # Deliberately returned out of id order.
    conn.fetch.return_value = [
        {"entity_id": id_b, "n": 1},
        {"entity_id": id_a, "n": 3},
    ]

    enqueued = await ops.release_entity_postings(
        conn, "entity_maintenance_queue", "entities", "unit_entities", "bank-1", ["u1", "u2"]
    )

    assert enqueued == 2
    assert conn.fetch.await_count == 1, "both halves must come out of one scan of the postings"

    (queue_sql, queue_rows), (debit_sql, debit_rows) = (call.args for call in conn.executemany.await_args_list)
    assert "MERGE INTO entity_maintenance_queue" in queue_sql
    assert queue_rows == [("bank-1", id_a), ("bank-1", id_b)]
    assert "GREATEST(mention_count - $3, 0)" in debit_sql
    assert debit_rows == [(id_a, "bank-1", 3), (id_b, "bank-1", 1)]


@pytest.mark.asyncio
async def test_oracle_release_of_units_with_no_postings_writes_nothing():
    """Units that never named an entity cost no write round-trip."""
    ops = OracleOps()
    conn = AsyncMock()
    conn.fetch.return_value = []

    released = await ops.release_entity_postings(
        conn, "entity_maintenance_queue", "entities", "unit_entities", "bank-1", ["u1"]
    )

    assert released == 0
    conn.executemany.assert_not_awaited()


@pytest.mark.asyncio
async def test_oracle_restore_credits_only_the_entities_that_still_exist():
    """The archive snapshot names entities the orphan prune may since have
    swept; the surviving set drives both the posting insert and the credit."""
    ops = OracleOps()
    conn = AsyncMock()
    id_a = "00000000-0000-0000-0000-00000000000a"
    id_b = "00000000-0000-0000-0000-00000000000b"
    gone = "00000000-0000-0000-0000-00000000000c"
    conn.fetch.return_value = [{"id": id_b}, {"id": id_a}]

    posted = await ops.restore_entity_postings(
        conn, "unit_entities", "entities", "bank-1", "unit-1", [id_a, id_b, gone]
    )

    assert posted == 2
    (post_sql, post_rows), (credit_sql, credit_rows) = (call.args for call in conn.executemany.await_args_list)
    assert "INSERT INTO unit_entities" in post_sql
    assert post_rows == [("unit-1", id_a), ("unit-1", id_b)]
    assert "mention_count + 1" in credit_sql
    assert credit_rows == [(id_a, "bank-1"), (id_b, "bank-1")]


@pytest.mark.asyncio
async def test_oracle_restore_of_nothing_writes_nothing():
    ops = OracleOps()
    conn = AsyncMock()

    assert await ops.restore_entity_postings(conn, "unit_entities", "entities", "bank-1", "unit-1", []) == 0
    conn.executemany.assert_not_awaited()


# ---------------------------------------------------------------------------
# Against the database
# ---------------------------------------------------------------------------


class TestReleaseAtTheChokePoint:
    @pytest.mark.asyncio
    async def test_subtracts_the_postings_each_entity_actually_loses(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Two units name Alice, one names Bob. Dropping one unit gives Alice one
        mention back and leaves Bob alone."""
        bank_id = f"test-mc-release-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            alice = await _insert_entity(conn, bank_id, "Alice", mention_count=2)
            bob = await _insert_entity(conn, bank_id, "Bob", mention_count=1)
            unit_a = await _insert_unit(conn, bank_id, "Alice moved to Berlin.")
            unit_b = await _insert_unit(conn, bank_id, "Alice plays cello with Bob.")
            await _link(conn, unit_a, alice)
            await _link(conn, unit_b, alice)
            await _link(conn, unit_b, bob)

            async with conn.transaction():
                await enqueue_entity_prune_candidates(conn, bank_id, [str(unit_a)])
                await conn.execute("DELETE FROM memory_units WHERE id = $1", unit_a)

            assert await _mention_count(conn, alice) == 1
            assert await _mention_count(conn, alice) == await _posting_count(conn, alice)
            assert await _mention_count(conn, bob) == 1
            # The debit and the prune-candidate queueing are one statement, so
            # the queue is part of this path's contract, not a separate concern.
            queued = await conn.fetch("SELECT entity_id FROM entity_maintenance_queue WHERE bank_id = $1", bank_id)
            assert {str(r["entity_id"]) for r in queued} == {str(alice)}

    @pytest.mark.asyncio
    async def test_one_unit_naming_an_entity_twice_gives_back_both_postings(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """The decrement counts postings, not units: an entity that two of the
        deleted units named loses two."""
        bank_id = f"test-mc-multi-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            alice = await _insert_entity(conn, bank_id, "Alice", mention_count=3)
            unit_a = await _insert_unit(conn, bank_id, "one")
            unit_b = await _insert_unit(conn, bank_id, "two")
            unit_c = await _insert_unit(conn, bank_id, "three")
            for unit in (unit_a, unit_b, unit_c):
                await _link(conn, unit, alice)

            async with conn.transaction():
                await enqueue_entity_prune_candidates(conn, bank_id, [str(unit_a), str(unit_b)])
                await conn.execute("DELETE FROM memory_units WHERE id = ANY($1::uuid[])", [unit_a, unit_b])

            assert await _mention_count(conn, alice) == 1

    @pytest.mark.asyncio
    async def test_floors_at_zero(self, memory: MemoryEngine, request_context: RequestContext):
        """A count that is already lower than the postings it is losing — a bank
        that drifted before this fix, or two spellings in one fact that resolved
        to one entity — clamps at zero rather than going negative."""
        bank_id = f"test-mc-floor-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            alice = await _insert_entity(conn, bank_id, "Alice", mention_count=1)
            unit_a = await _insert_unit(conn, bank_id, "one")
            unit_b = await _insert_unit(conn, bank_id, "two")
            await _link(conn, unit_a, alice)
            await _link(conn, unit_b, alice)

            async with conn.transaction():
                await enqueue_entity_prune_candidates(conn, bank_id, [str(unit_a), str(unit_b)])
                await conn.execute("DELETE FROM memory_units WHERE id = ANY($1::uuid[])", [unit_a, unit_b])

            assert await _mention_count(conn, alice) == 0

    @pytest.mark.asyncio
    async def test_leaves_other_banks_alone(self, memory: MemoryEngine, request_context: RequestContext):
        """`unit_entities` has no bank column; the release is scoped through
        `entities.bank_id`, so a same-named entity next door is untouched."""
        bank_id = f"test-mc-scope-{uuid.uuid4().hex[:8]}"
        other_bank = f"{bank_id}-other"
        await _ensure_bank(memory, bank_id, request_context)
        await _ensure_bank(memory, other_bank, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            alice = await _insert_entity(conn, bank_id, "Alice", mention_count=1)
            alice_elsewhere = await _insert_entity(conn, other_bank, "Alice", mention_count=1)
            unit = await _insert_unit(conn, bank_id, "Alice moved to Berlin.")
            await _link(conn, unit, alice)

            async with conn.transaction():
                await enqueue_entity_prune_candidates(conn, bank_id, [str(unit)])
                await conn.execute("DELETE FROM memory_units WHERE id = $1", unit)

            assert await _mention_count(conn, alice) == 0
            assert await _mention_count(conn, alice_elsewhere) == 1


class TestCurationPaths:
    @pytest.mark.asyncio
    async def test_invalidate_then_revert_restores_the_mention(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """Invalidation takes the posting, so it takes the mention with it; the
        revert puts both back. Getting only one of the two halves right is what
        turns a stale-high counter into a stale-low one."""
        bank_id = f"test-mc-revert-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            alice = await _insert_entity(conn, bank_id, "Alice", mention_count=2)
            keeper = await _insert_unit(conn, bank_id, "Alice plays cello.")
            doomed = await _insert_unit(conn, bank_id, "Alice moved to Berlin.")
            await _link(conn, keeper, alice)
            await _link(conn, doomed, alice)

        await memory.update_memory_unit(
            bank_id, str(doomed), state="invalidated", reason="test", request_context=request_context
        )
        async with pool.acquire() as conn:
            assert await _mention_count(conn, alice) == 1

        await memory.update_memory_unit(bank_id, str(doomed), state="valid", request_context=request_context)
        async with pool.acquire() as conn:
            assert await _mention_count(conn, alice) == 2
            assert await _mention_count(conn, alice) == await _posting_count(conn, alice)

    @pytest.mark.asyncio
    async def test_revert_skips_an_entity_swept_while_the_memory_was_archived(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """The restore only re-posts entities that still exist, so it may only
        give some of the mentions back — it must count what it actually posted,
        not what the archive snapshot named. Alice keeps a second fact so she
        survives the archive; Berlin does not, and the orphan prune takes her."""
        bank_id = f"test-mc-swept-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            alice = await _insert_entity(conn, bank_id, "Alice", mention_count=2)
            berlin = await _insert_entity(conn, bank_id, "Berlin", mention_count=1)
            keeper = await _insert_unit(conn, bank_id, "Alice plays cello.")
            unit = await _insert_unit(conn, bank_id, "Alice moved to Berlin.")
            await _link(conn, keeper, alice)
            await _link(conn, unit, alice)
            await _link(conn, unit, berlin)

        await memory.update_memory_unit(
            bank_id, str(unit), state="invalidated", reason="test", request_context=request_context
        )
        async with pool.acquire() as conn:
            assert await _mention_count(conn, alice) == 1
            # Berlin lost her only posting, so the drain the invalidation
            # triggered swept her — the case the restore has to cope with.
            assert await conn.fetchval("SELECT COUNT(*) FROM entities WHERE id = $1", berlin) == 0

        await memory.update_memory_unit(bank_id, str(unit), state="valid", request_context=request_context)
        async with pool.acquire() as conn:
            # Only Alice's mention comes back: Berlin has no row to credit.
            assert await _mention_count(conn, alice) == 2
            assert await _mention_count(conn, alice) == await _posting_count(conn, alice)

    @pytest.mark.asyncio
    async def test_deleting_a_document_gives_back_only_its_own_mentions(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """The path from the issue: two documents name Alice, one goes, Alice's
        count follows the facts that survive rather than staying where it was."""
        bank_id = f"test-mc-doc-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            alice = await _insert_entity(conn, bank_id, "Alice", mention_count=2)
            for doc_id in ("doc-1", "doc-2"):
                await conn.execute(
                    """
                    INSERT INTO documents (id, bank_id, original_text, content_hash)
                    VALUES ($1, $2, $3, $4)
                    """,
                    doc_id,
                    bank_id,
                    f"text of {doc_id}",
                    f"hash-{doc_id}",
                )
                unit_id = uuid.uuid4()
                await conn.execute(
                    """
                    INSERT INTO memory_units
                        (id, bank_id, document_id, text, fact_type, event_date, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, 'experience', NOW(), NOW(), NOW())
                    """,
                    unit_id,
                    bank_id,
                    doc_id,
                    f"Alice appears in {doc_id}.",
                )
                await _link(conn, unit_id, alice)

        await memory.delete_document("doc-1", bank_id, request_context=request_context)

        async with pool.acquire() as conn:
            assert await _mention_count(conn, alice) == 1
            assert await _mention_count(conn, alice) == await _posting_count(conn, alice)


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_replacing_a_document_leaves_every_count_matching_its_postings(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        """The issue's own reproduction, through the real retain pipeline.

        Two documents name Alice, then one is replaced. The assertion is the
        invariant rather than a number for Alice: whatever the extractor made of
        the text, no entity in the bank may claim more mentions than it has
        surviving facts. Before the fix Alice came out of this at 3 with 2
        postings, and the gap grew with every rewrite of the document.
        """
        bank_id = f"test-mc-e2e-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        await memory.retain_async(
            bank_id=bank_id, content="Alice moved to Berlin.", document_id="d1", request_context=request_context
        )
        await memory.retain_async(
            bank_id=bank_id, content="Alice plays cello.", document_id="d2", request_context=request_context
        )
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Alice moved to Lisbon.", "document_id": "d1", "update_mode": "replace"}],
            request_context=request_context,
        )

        listed = await memory.list_entities(bank_id, request_context=request_context)
        assert listed["items"], "the retain pipeline should have produced entities to check"

        # The number to compare the listing against is how many facts still mention
        # each entity, which no read endpoint exposes — `list_entities` returns the
        # denormalised counter itself, so checking it against itself proves nothing.
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            postings = {
                str(r["entity_id"]): r["n"]
                for r in await conn.fetch(
                    """
                    SELECT ue.entity_id, COUNT(*) AS n
                    FROM unit_entities ue
                    JOIN memory_units mu ON mu.id = ue.unit_id
                    WHERE mu.bank_id = $1
                    GROUP BY ue.entity_id
                    """,
                    bank_id,
                )
            }
        inflated = [
            (e["canonical_name"], e["mention_count"], postings.get(e["id"], 0))
            for e in listed["items"]
            if e["mention_count"] != postings.get(e["id"], 0)
        ]
        assert not inflated, f"mention_count drifted from the surviving postings: {inflated}"

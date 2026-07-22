"""Regression coverage for the retain Phase-1 / orphan-prune FK race (#2662).

Between Phase-1 entity resolution (a separate, already-committed connection) and
the Phase-2 ``unit_entities`` insert, ``prune_orphan_entities`` can delete a
just-resolved parent that has no ``unit_entities`` row yet. The Phase-2 FK insert
then fails and the whole batch is dropped as non-retryable — silent memory loss.

The fix reasserts the resolved parents on the Phase-2 connection immediately
before linking: existing rows are locked so the pruner blocks, and rows already
pruned are re-created with their stored canonical name.
"""

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncpg
import pytest

from hindsight_api.engine.db import create_database_backend
from hindsight_api.engine.db.ops_oracle import OracleOps
from hindsight_api.engine.db.ops_postgresql import PostgreSQLOps
from hindsight_api.engine.db.postgresql import PostgresConnection
from hindsight_api.engine.entity_resolver import EntityResolver
from hindsight_api.engine.memory_engine import fq_table
from hindsight_api.engine.retain.orchestrator import (
    _insert_facts_and_links,
    _pre_resolve_phase1,
)
from hindsight_api.engine.retain.types import EntityRef, ProcessedFact, ResolvedEntity, RetainContent
from hindsight_api.pg0 import resolve_database_url

_ID_A = "00000000-0000-0000-0000-000000000001"
_ID_B = "00000000-0000-0000-0000-000000000002"


@pytest.mark.asyncio
async def test_oracle_reassert_locks_in_stable_order_then_reinserts():
    """Oracle adapter: lock every parent (FOR UPDATE) in sorted id order, then
    idempotently re-insert the same rows in that order."""
    resolver = EntityResolver(pool=SimpleNamespace(ops=OracleOps()))
    conn = AsyncMock()
    # Deliberately out of order and duplicated on the way in.
    resolved_entities = [
        ResolvedEntity(entity_id=_ID_B, canonical_name="Bob"),
        ResolvedEntity(entity_id=_ID_A, canonical_name="Alice"),
        ResolvedEntity(entity_id=_ID_B, canonical_name="Bob"),
    ]

    await resolver.reassert_entities_batch("bank-1", resolved_entities, conn)

    lock_ids = [call.args[1] for call in conn.fetchrow.await_args_list]
    assert lock_ids == [_ID_A, _ID_B], "parents must be locked once each, in stable id order"
    assert all("FOR UPDATE" in call.args[0] for call in conn.fetchrow.await_args_list)

    insert_sql, rows = conn.executemany.await_args.args
    assert "ON CONFLICT DO NOTHING" in insert_sql
    assert rows == [(_ID_A, "bank-1", "Alice"), (_ID_B, "bank-1", "Bob")]


@pytest.mark.asyncio
async def test_reassert_of_empty_batch_is_a_noop():
    """No resolved entities → no SQL issued (avoids a pointless round-trip)."""
    resolver = EntityResolver(pool=SimpleNamespace(ops=OracleOps()))
    conn = AsyncMock()

    await resolver.reassert_entities_batch("bank-1", [], conn)

    conn.fetchrow.assert_not_awaited()
    conn.executemany.assert_not_awaited()


@pytest.mark.asyncio
async def test_reassert_locks_existing_parent_until_child_insert(pg0_db_url):
    """A parent that still exists at reassert time is locked FOR KEY SHARE, so a
    concurrent prune DELETE blocks until the child link commits."""
    bank_id = f"test-reassert-lock-{uuid.uuid4().hex[:8]}"
    setup = await asyncpg.connect(pg0_db_url)
    phase2_conn = await asyncpg.connect(pg0_db_url)
    prune_conn = await asyncpg.connect(pg0_db_url)
    ops = PostgreSQLOps()

    try:
        entity_id = await setup.fetchval(
            "INSERT INTO entities (bank_id, canonical_name) VALUES ($1, 'Alice Smith') RETURNING id",
            bank_id,
        )

        phase2_tx = phase2_conn.transaction()
        await phase2_tx.start()
        unit_id = await phase2_conn.fetchval(
            """
            INSERT INTO memory_units (bank_id, text, event_date, fact_type)
            VALUES ($1, 'Alice Smith joined the project.', now(), 'world')
            RETURNING id
            """,
            bank_id,
        )
        wrapped_phase2 = PostgresConnection(phase2_conn)
        await ops.bulk_reassert_entities(wrapped_phase2, "entities", bank_id, [str(entity_id)], ["Alice Smith"])

        # The pruner cannot delete the locked parent — it blocks and times out.
        await prune_conn.execute("SET statement_timeout = '500ms'")
        with pytest.raises(asyncpg.QueryCanceledError):
            await prune_conn.execute("DELETE FROM entities WHERE id = $1", entity_id)

        await ops.bulk_insert_unit_entities(wrapped_phase2, "unit_entities", [str(unit_id)], [str(entity_id)])
        await phase2_tx.commit()

        # Positive control: once committed, the link exists and the parent is present.
        assert (
            await setup.fetchval(
                "SELECT count(*) FROM unit_entities WHERE unit_id = $1 AND entity_id = $2",
                unit_id,
                entity_id,
            )
            == 1
        )
    finally:
        await setup.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
        await setup.execute("DELETE FROM entities WHERE bank_id = $1", bank_id)
        await setup.close()
        await phase2_conn.close()
        await prune_conn.close()


@pytest.mark.asyncio
async def test_phase2_reasserts_entity_pruned_after_resolution(pg0_db_url):
    """End-to-end: a prune between the two retain phases must not become data loss.

    The input is a fuzzy alias ("Alice Smit"), so Phase 2 must carry the stored
    canonical name ("Alice Smith") — it is unrecoverable once the row is pruned.
    """
    resolved_url = await resolve_database_url(pg0_db_url)
    backend = create_database_backend("postgresql")
    await backend.initialize(resolved_url, min_size=1, max_size=3, command_timeout=30)

    bank_id = f"test-retain-prune-race-{uuid.uuid4().hex[:8]}"
    event_date = datetime(2026, 7, 13, 12, tzinfo=UTC)
    resolver = EntityResolver(pool=backend, entity_lookup="full")
    config = SimpleNamespace(entity_labels=None)
    contents = [RetainContent(content="Alice Smit joined the project.")]
    processed_facts = [
        ProcessedFact(
            fact_text="Alice Smit joined the project.",
            fact_type="world",
            embedding=[0.0] * 384,
            occurred_start=event_date,
            occurred_end=None,
            mentioned_at=event_date,
            context="",
            metadata={},
            entities=[EntityRef(name="Alice Smit")],
        )
    ]

    try:
        async with backend.acquire() as conn:
            original_entity_id = await conn.fetchval(
                f"""
                INSERT INTO {fq_table("entities")}
                    (bank_id, canonical_name, first_seen, last_seen, mention_count)
                VALUES ($1, 'Alice Smith', $2, $2, 1)
                RETURNING id
                """,
                bank_id,
                event_date,
            )

        phase1 = await _pre_resolve_phase1(
            backend, resolver, bank_id, contents, processed_facts, config, [], skip_semantic_ann=True
        )
        assert phase1.entities.resolved_entities[0].entity_id == str(original_entity_id)
        assert phase1.entities.resolved_entities[0].canonical_name == "Alice Smith"

        # Prune the resolved parent in the window before Phase 2 links it.
        async with backend.acquire() as prune_conn:
            async with prune_conn.transaction():
                pruned = await backend.ops.prune_orphan_entities(
                    prune_conn, fq_table("entities"), fq_table("unit_entities"), bank_id
                )
        assert pruned == 1

        async with backend.acquire() as phase2_conn:
            async with phase2_conn.transaction():
                unit_id_groups = await _insert_facts_and_links(
                    phase2_conn,
                    resolver,
                    bank_id,
                    contents,
                    [],
                    processed_facts,
                    config,
                    [],
                    resolved_entities=phase1.entities.resolved_entities,
                    entity_to_unit=phase1.entities.entity_to_unit,
                    unit_to_entity_ids=phase1.entities.unit_to_entity_ids,
                    semantic_ann_links=[],
                    skip_semantic_links=True,
                    ops=backend.ops,
                )

        assert len(unit_id_groups[0]) == 1
        async with backend.acquire() as verify_conn:
            restored = await verify_conn.fetchrow(
                f"SELECT id, canonical_name FROM {fq_table('entities')} WHERE id = $1",
                original_entity_id,
            )
            linked_entity_id = await verify_conn.fetchval(
                f"""
                SELECT ue.entity_id
                FROM {fq_table("unit_entities")} ue
                JOIN {fq_table("memory_units")} mu ON mu.id = ue.unit_id
                WHERE mu.bank_id = $1
                """,
                bank_id,
            )

        assert restored is not None, "the pruned parent must be re-created"
        assert restored["canonical_name"] == "Alice Smith", "reassert must restore the stored canonical name"
        assert linked_entity_id == original_entity_id, "the unit must link back to the original entity id"
    finally:
        async with backend.acquire() as conn:
            await conn.execute(f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1", bank_id)
            await conn.execute(f"DELETE FROM {fq_table('entities')} WHERE bank_id = $1", bank_id)
        await backend.shutdown()

"""Deleting a bank leaves no row of it in any table.

Every table that names a bank is read from the catalog, so a table added later is
checked without anyone remembering to list it here; the test refuses to pass until
the new table is seeded too (an empty table proves nothing about the delete).
"""

import json
import uuid

import asyncpg
import pytest

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.models import RequestContext

# Seeds the memory tables with raw SQL, so it only holds for the default Postgres store.
pytestmark = pytest.mark.memory_backend_incompatible

# Tables that keep a deleted bank's rows on purpose, with why.
KEPT_AFTER_DELETE = {
    # The audit trail records what was done to the bank, its deletion included, so it
    # outlives the bank and is trimmed by HINDSIGHT_API_AUDIT_LOG_RETENTION_DAYS instead.
    "audit_log",
}


async def _seed_bank(conn: asyncpg.Connection, bank_id: str) -> list[uuid.UUID]:
    """One row in every bank-scoped table. Returns the unit and entity ids, for the junction tables."""
    doc_id = f"doc-{uuid.uuid4().hex[:8]}"
    mm_id = f"mm-{uuid.uuid4().hex[:8]}"
    chunk_id = f"{bank_id}-chunk-{uuid.uuid4().hex[:8]}"
    await conn.execute("INSERT INTO banks (bank_id) VALUES ($1)", bank_id)
    await conn.execute("INSERT INTO bank_aliases (alias, bank_id) VALUES ($1, $2)", f"alias-{bank_id}", bank_id)
    await conn.execute("INSERT INTO documents (id, bank_id, original_text) VALUES ($1, $2, 'text')", doc_id, bank_id)
    await conn.execute(
        "INSERT INTO chunks (chunk_id, document_id, bank_id, chunk_index, chunk_text) VALUES ($1, $2, $3, 0, 'text')",
        chunk_id,
        doc_id,
        bank_id,
    )
    units = [
        await conn.fetchval(
            """INSERT INTO memory_units (bank_id, document_id, text, fact_type, event_date)
               VALUES ($1, $2, $3, 'world', NOW()) RETURNING id""",
            bank_id,
            doc_id,
            text,
        )
        for text in ("Alice likes tea", "Bob likes coffee")
    ]
    entities = [
        await conn.fetchval(
            "INSERT INTO entities (canonical_name, bank_id) VALUES ($1, $2) RETURNING id", name, bank_id
        )
        for name in ("Alice", "Bob")
    ]
    unit_a, unit_b = units
    entity_a, entity_b = entities
    await conn.execute("INSERT INTO unit_entities (unit_id, entity_id) VALUES ($1, $2)", unit_a, entity_a)
    await conn.execute(
        "INSERT INTO entity_cooccurrences (entity_id_1, entity_id_2) VALUES ($1, $2)",
        min(entity_a, entity_b),
        max(entity_a, entity_b),
    )
    await conn.execute(
        "INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, bank_id) VALUES ($1, $2, 'semantic', $3)",
        unit_a,
        unit_b,
        bank_id,
    )
    await conn.execute(
        "INSERT INTO observation_history (observation_id, bank_id, content) VALUES ($1, $2, '{}'::jsonb)",
        unit_a,
        bank_id,
    )
    await conn.execute("INSERT INTO invalidated_memory_units (bank_id, text) VALUES ($1, 'Alice liked tea')", bank_id)
    await conn.execute(
        """INSERT INTO attachments
               (bank_id, attachment_hash, short_id, media_type, byte_size, storage_key, document_id)
           VALUES ($1, $2, $3, 'text/plain', 4, $4, $5)""",
        bank_id,
        uuid.uuid4().hex,
        uuid.uuid4().hex[:8],
        f"tenants/test/{bank_id}/attachments/a.txt",
        doc_id,
    )
    await conn.execute(
        """INSERT INTO mental_models (id, bank_id, subtype, name, source_query, content)
           VALUES ($1, $2, 'pinned', 'Model', 'q?', 'c')""",
        mm_id,
        bank_id,
    )
    await conn.execute(
        "INSERT INTO mental_model_history (mental_model_id, bank_id, content) VALUES ($1, $2, '{}'::jsonb)",
        mm_id,
        bank_id,
    )
    await conn.execute(
        "INSERT INTO knowledge_pages (id, bank_id, kind, name) VALUES ($1, $2, 'folder', 'Folder')",
        f"kp-{uuid.uuid4().hex[:8]}",
        bank_id,
    )
    await conn.execute("INSERT INTO directives (bank_id, name, content) VALUES ($1, 'tone', 'be brief')", bank_id)
    await conn.execute("INSERT INTO webhooks (bank_id, url) VALUES ($1, 'https://example.com/hook')", bank_id)
    await conn.execute(
        "INSERT INTO async_operations (bank_id, operation_type, status) VALUES ($1, 'consolidation', 'completed')",
        bank_id,
    )
    await conn.execute(
        "INSERT INTO audit_log (bank_id, action, transport, started_at) VALUES ($1, 'retain', 'http', NOW())",
        bank_id,
    )
    await conn.execute(
        "INSERT INTO llm_requests (bank_id, status, started_at, input) VALUES ($1, 'success', NOW(), $2::jsonb)",
        bank_id,
        json.dumps({"messages": [{"role": "user", "content": "Alice likes tea"}]}),
    )
    await conn.execute(
        "INSERT INTO bank_stats_cache (bank_id, payload) VALUES ($1, $2::jsonb)",
        bank_id,
        json.dumps({"total_nodes": 2}),
    )
    await conn.execute("INSERT INTO graph_maintenance_queue (bank_id, unit_id) VALUES ($1, $2)", bank_id, unit_a)
    await conn.execute("INSERT INTO entity_maintenance_queue (bank_id, entity_id) VALUES ($1, $2)", bank_id, entity_a)
    return [unit_a, unit_b, entity_a, entity_b]


async def _counts(conn: asyncpg.Connection, bank_id: str, ids: list[uuid.UUID]) -> dict[str, int]:
    # Raw SQL on purpose: the property under test is that no table keeps a row of the
    # bank, which no engine read API can enumerate.
    tables = await conn.fetch(
        """SELECT c.table_name FROM information_schema.columns c
           JOIN information_schema.tables t USING (table_schema, table_name)
           WHERE c.table_schema = current_schema() AND c.column_name = 'bank_id' AND t.table_type = 'BASE TABLE'"""
    )
    counts = {
        r["table_name"]: await conn.fetchval(f"SELECT count(*) FROM {r['table_name']} WHERE bank_id = $1", bank_id)
        for r in tables
    }
    # The junction tables carry no bank_id: reach them through the bank's own ids.
    counts["unit_entities"] = await conn.fetchval(
        "SELECT count(*) FROM unit_entities WHERE unit_id = ANY($1) OR entity_id = ANY($1)", ids
    )
    counts["entity_cooccurrences"] = await conn.fetchval(
        "SELECT count(*) FROM entity_cooccurrences WHERE entity_id_1 = ANY($1) OR entity_id_2 = ANY($1)", ids
    )
    return counts


@pytest.mark.asyncio
async def test_deleting_a_bank_leaves_no_row_in_any_table(memory: MemoryEngine, request_context: RequestContext):
    bank_id = f"test-delete-all-{uuid.uuid4().hex[:8]}"
    bystander = f"test-delete-bystander-{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        ids = await _seed_bank(conn, bank_id)
        bystander_ids = await _seed_bank(conn, bystander)
        before = await _counts(conn, bank_id, ids)
        bystander_before = await _counts(conn, bystander, bystander_ids)

    unseeded = sorted(t for t, n in before.items() if not n)
    assert not unseeded, f"seed these tables in _seed_bank, or the delete is not checked for them: {unseeded}"

    await memory.delete_bank(bank_id, request_context=request_context)

    async with pool.acquire() as conn:
        left = {t: n for t, n in (await _counts(conn, bank_id, ids)).items() if n and t not in KEPT_AFTER_DELETE}
        assert not left, f"rows of the deleted bank left behind: {left}"
        assert await _counts(conn, bystander, bystander_ids) == bystander_before

    await memory.delete_bank(bystander, request_context=request_context)

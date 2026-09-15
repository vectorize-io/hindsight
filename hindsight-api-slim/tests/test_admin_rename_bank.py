"""hindsight-admin rename-bank: move a bank to a new id in place.

Runs against an isolated, freshly migrated schema, so the catalog assertions see
exactly what the migrations produce and the index DDL never touches the shared
public.memory_units.
"""

import uuid

import asyncpg
import pytest
import pytest_asyncio

from hindsight_api.admin.cli import _RIGID_BANK_ID_FKS_SQL, RenameBankError, _rename_bank, _run_rename_bank
from hindsight_api.engine.retain.bank_utils import _BANK_INDEX_FACT_TYPES, _vector_index_clause
from hindsight_api.engine.vector_index_health import plan_bank_vector_indexes, reconcile_bank_vector_indexes
from hindsight_api.migrations import run_migrations


@pytest_asyncio.fixture
async def rename_schema(pg0_db_url):
    schema = f"rename_test_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
    finally:
        await conn.close()
    run_migrations(pg0_db_url, schema=schema)
    conn = await asyncpg.connect(pg0_db_url)
    try:
        yield pg0_db_url, schema, conn
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


async def _seed_bank(conn: asyncpg.Connection, schema: str, bank_id: str) -> None:
    """A bank with rows behind each composite (id, bank_id) FK."""
    doc_id = f"doc-{uuid.uuid4().hex[:8]}"
    mm_id = f"mm-{uuid.uuid4().hex[:8]}"
    await conn.execute(f"INSERT INTO {schema}.banks (bank_id) VALUES ($1)", bank_id)
    await conn.execute(
        f"INSERT INTO {schema}.documents (id, bank_id, original_text) VALUES ($1, $2, 'text')", doc_id, bank_id
    )
    await conn.execute(
        f"""INSERT INTO {schema}.memory_units (bank_id, document_id, text, fact_type, event_date)
            VALUES ($1, $2, 'Alice likes tea', 'world', NOW())""",
        bank_id,
        doc_id,
    )
    await conn.execute(
        f"""INSERT INTO {schema}.mental_models (id, bank_id, subtype, name, source_query, content)
            VALUES ($1, $2, 'pinned', 'Model', 'q?', 'c')""",
        mm_id,
        bank_id,
    )
    await conn.execute(
        f"""INSERT INTO {schema}.mental_model_history (mental_model_id, bank_id, content)
            VALUES ($1, $2, '{{}}'::jsonb)""",
        mm_id,
        bank_id,
    )
    await conn.execute(
        f"INSERT INTO {schema}.directives (bank_id, name, content) VALUES ($1, 'tone', 'be brief')", bank_id
    )


async def _counts(conn: asyncpg.Connection, schema: str, bank_id: str) -> dict[str, int]:
    # Raw SQL on purpose: the property under test is that no table in the schema
    # keeps a row under the old id, which no engine read API can enumerate.
    tables = await conn.fetch(
        """SELECT c.table_name FROM information_schema.columns c
           JOIN information_schema.tables t USING (table_schema, table_name)
           WHERE c.table_schema = $1 AND c.column_name = 'bank_id' AND t.table_type = 'BASE TABLE'""",
        schema,
    )
    return {
        r["table_name"]: await conn.fetchval(
            f"SELECT count(*) FROM {schema}.{r['table_name']} WHERE bank_id = $1", bank_id
        )
        for r in tables
    }


@pytest.mark.asyncio
async def test_rename_restores_fk_deferrability(rename_schema):
    """The FKs are made DEFERRABLE only for the rename; the schema ends as the migrations declared it."""
    _, schema, conn = rename_schema
    await _seed_bank(conn, schema, "old")
    rigid_before = sorted(tuple(r) for r in await conn.fetch(_RIGID_BANK_ID_FKS_SQL, schema))
    assert rigid_before, "expected NOT DEFERRABLE bank_id FKs in a freshly migrated schema"

    await _rename_bank(conn, schema, "old", "new", dry_run=False)
    with pytest.raises(RenameBankError):
        await _rename_bank(conn, schema, "old", "newer", dry_run=False)

    assert sorted(tuple(r) for r in await conn.fetch(_RIGID_BANK_ID_FKS_SQL, schema)) == rigid_before


@pytest.mark.asyncio
async def test_rename_moves_every_row_and_dry_run_moves_none(rename_schema):
    _, schema, conn = rename_schema
    await _seed_bank(conn, schema, "old")
    await _seed_bank(conn, schema, "bystander")
    before = await _counts(conn, schema, "old")
    bystander = await _counts(conn, schema, "bystander")
    expected = {t: n for t, n in before.items() if n}

    assert await _rename_bank(conn, schema, "old", "new", dry_run=True) == expected
    assert await _counts(conn, schema, "old") == before

    assert await _rename_bank(conn, schema, "old", "new", dry_run=False) == expected
    assert await _counts(conn, schema, "new") == before
    assert not any((await _counts(conn, schema, "old")).values())
    assert await _counts(conn, schema, "bystander") == bystander


@pytest.mark.asyncio
async def test_rename_refuses_without_changing_anything(rename_schema):
    _, schema, conn = rename_schema
    await _seed_bank(conn, schema, "old")
    await _seed_bank(conn, schema, "taken")
    await conn.execute(
        f"""INSERT INTO {schema}.async_operations (operation_id, bank_id, operation_type, status)
            VALUES ($1, 'old', 'consolidation', 'completed')""",
        uuid.uuid4(),
    )
    before = await _counts(conn, schema, "old")

    with pytest.raises(RenameBankError, match="does not exist"):
        await _rename_bank(conn, schema, "missing", "new", dry_run=False)
    with pytest.raises(RenameBankError, match="already exists"):
        await _rename_bank(conn, schema, "old", "taken", dry_run=False)

    await conn.execute(f"UPDATE {schema}.async_operations SET status = 'pending' WHERE bank_id = 'old'")
    with pytest.raises(RenameBankError, match="pending or processing"):
        await _rename_bank(conn, schema, "old", "new", dry_run=False)

    assert await _counts(conn, schema, "old") == before


@pytest.mark.asyncio
async def test_rename_rebuilds_vector_indexes_for_the_new_id(rename_schema):
    """The per-bank indexes are partial on the bank_id literal: left alone they cover nothing."""
    db_url, schema, conn = rename_schema
    index_clause = _vector_index_clause()
    if index_clause is None:
        pytest.skip("configured vector backend has no per-bank indexes")
    await _seed_bank(conn, schema, "old")
    built = await reconcile_bank_vector_indexes(conn, schema, "old", index_clause)
    assert built.created == len(_BANK_INDEX_FACT_TYPES)

    await _run_rename_bank(db_url, schema, "old", "new", dry_run=False)

    plan = await plan_bank_vector_indexes(conn, schema, "new")
    assert plan.is_empty and plan.already_present == len(_BANK_INDEX_FACT_TYPES)
    defs = await conn.fetch(
        "SELECT indexdef FROM pg_indexes WHERE schemaname = $1 AND indexname LIKE 'idx_mu_emb_%'", schema
    )
    assert defs and all("'new'::text" in r["indexdef"] for r in defs)

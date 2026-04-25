"""
Tests for admin reindex-embeddings functionality.

Closes vectorize-io/hindsight#743.

These tests use an isolated schema to avoid interfering with other tests.
"""

import os
import uuid

import asyncpg
import pytest
import pytest_asyncio

from hindsight_api.admin.reindex import (
    VectorColumnInfo,
    _count_pending,
    _discover_vector_columns,
    _reembed_table,
    _reindex_embeddings,
)
from hindsight_api.migrations import run_migrations


pytestmark = pytest.mark.xdist_group(name="reindex_embeddings")


@pytest_asyncio.fixture(scope="function")
async def reindex_test_schema(pg0_db_url, embeddings):
    """Create an isolated schema for reindex tests with a unique name."""
    await embeddings.initialize()

    schema_name = f"reindex_test_{uuid.uuid4().hex[:8]}"

    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(f"CREATE SCHEMA {schema_name}")
    finally:
        await conn.close()

    run_migrations(pg0_db_url, schema=schema_name)

    yield pg0_db_url, schema_name, embeddings

    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_discover_vector_columns_finds_known_tables(reindex_test_schema):
    """Schema introspection should find memory_units.embedding and mental_models.embedding."""
    db_url, schema_name, _ = reindex_test_schema
    conn = await asyncpg.connect(db_url)
    try:
        cols = await _discover_vector_columns(conn, schema_name)
    finally:
        await conn.close()

    table_cols = {(c.table, c.column) for c in cols}
    assert ("memory_units", "embedding") in table_cols
    assert ("mental_models", "embedding") in table_cols
    # text source columns mapped correctly
    by_table = {c.table: c for c in cols}
    assert by_table["memory_units"].text_column == "text"
    assert by_table["mental_models"].text_column == "content"


@pytest.mark.asyncio
async def test_discover_vector_columns_parses_dimension(reindex_test_schema):
    """Discovered columns should report the correct dimension from the schema."""
    db_url, schema_name, embeddings = reindex_test_schema
    conn = await asyncpg.connect(db_url)
    try:
        cols = await _discover_vector_columns(conn, schema_name)
    finally:
        await conn.close()

    for c in cols:
        # Test schema was created with the embeddings fixture's dimension
        assert c.dimension == embeddings.dimension


@pytest.mark.asyncio
async def test_count_pending_returns_zero_for_empty_table(reindex_test_schema):
    """Empty table should return 0 pending rows."""
    db_url, schema_name, _ = reindex_test_schema
    conn = await asyncpg.connect(db_url)
    try:
        cols = await _discover_vector_columns(conn, schema_name)
        for col in cols:
            n = await _count_pending(conn, col)
            assert n == 0
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_count_pending_counts_null_embeddings(reindex_test_schema):
    """Inserting rows with NULL embedding should be counted."""
    db_url, schema_name, _ = reindex_test_schema
    conn = await asyncpg.connect(db_url)
    bank_id = f"test-{uuid.uuid4().hex[:8]}"
    try:
        await conn.execute(
            f'INSERT INTO {schema_name}.banks (bank_id) VALUES ($1) ON CONFLICT DO NOTHING',
            bank_id,
        )
        # Insert 3 memory units without embeddings
        for i in range(3):
            await conn.execute(
                f"""
                INSERT INTO {schema_name}.memory_units
                  (id, bank_id, text, fact_type)
                VALUES ($1, $2, $3, 'world')
                """,
                uuid.uuid4(),
                bank_id,
                f"test memory {i}",
            )

        cols = await _discover_vector_columns(conn, schema_name)
        memory_col = next(c for c in cols if c.table == "memory_units")
        n = await _count_pending(conn, memory_col)
        assert n == 3
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_reembed_table_populates_null_embeddings(reindex_test_schema):
    """Running re-embed on rows with NULL embeddings should fill them in."""
    db_url, schema_name, embeddings = reindex_test_schema
    conn = await asyncpg.connect(db_url)
    bank_id = f"test-{uuid.uuid4().hex[:8]}"
    try:
        await conn.execute(
            f'INSERT INTO {schema_name}.banks (bank_id) VALUES ($1) ON CONFLICT DO NOTHING',
            bank_id,
        )
        ids = []
        for i in range(5):
            mid = uuid.uuid4()
            ids.append(mid)
            await conn.execute(
                f"""
                INSERT INTO {schema_name}.memory_units
                  (id, bank_id, text, fact_type)
                VALUES ($1, $2, $3, 'world')
                """,
                mid,
                bank_id,
                f"test memory {i}",
            )

        cols = await _discover_vector_columns(conn, schema_name)
        memory_col = next(c for c in cols if c.table == "memory_units")
        memory_col.fq_table = f'"{schema_name}"."memory_units"'

        processed, skipped = await _reembed_table(
            conn, embeddings, memory_col, batch_size=8, bank_id=bank_id
        )
        assert processed == 5
        assert skipped == 0

        # Verify all 5 rows now have non-null embeddings
        n_with_embedding = await conn.fetchval(
            f"""
            SELECT COUNT(*) FROM {schema_name}.memory_units
            WHERE embedding IS NOT NULL AND bank_id = $1
            """,
            bank_id,
        )
        assert n_with_embedding == 5
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_reembed_is_idempotent(reindex_test_schema):
    """Running re-embed twice should be a no-op the second time."""
    db_url, schema_name, embeddings = reindex_test_schema
    conn = await asyncpg.connect(db_url)
    bank_id = f"test-{uuid.uuid4().hex[:8]}"
    try:
        await conn.execute(
            f'INSERT INTO {schema_name}.banks (bank_id) VALUES ($1) ON CONFLICT DO NOTHING',
            bank_id,
        )
        for i in range(3):
            await conn.execute(
                f"""
                INSERT INTO {schema_name}.memory_units
                  (id, bank_id, text, fact_type)
                VALUES ($1, $2, $3, 'world')
                """,
                uuid.uuid4(),
                bank_id,
                f"idempotency test {i}",
            )

        cols = await _discover_vector_columns(conn, schema_name)
        memory_col = next(c for c in cols if c.table == "memory_units")
        memory_col.fq_table = f'"{schema_name}"."memory_units"'

        # First pass: re-embed all 3
        p1, _ = await _reembed_table(conn, embeddings, memory_col, 8, bank_id)
        assert p1 == 3

        # Second pass: should be 0 since none are NULL anymore
        p2, _ = await _reembed_table(conn, embeddings, memory_col, 8, bank_id)
        assert p2 == 0
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_bank_filter_only_processes_specified_bank(reindex_test_schema):
    """--bank flag should only re-embed memories from the specified bank."""
    db_url, schema_name, embeddings = reindex_test_schema
    conn = await asyncpg.connect(db_url)
    bank_a = f"bank-a-{uuid.uuid4().hex[:8]}"
    bank_b = f"bank-b-{uuid.uuid4().hex[:8]}"
    try:
        for b in (bank_a, bank_b):
            await conn.execute(
                f'INSERT INTO {schema_name}.banks (bank_id) VALUES ($1) ON CONFLICT DO NOTHING',
                b,
            )
            for i in range(2):
                await conn.execute(
                    f"""
                    INSERT INTO {schema_name}.memory_units
                      (id, bank_id, text, fact_type)
                    VALUES ($1, $2, $3, 'world')
                    """,
                    uuid.uuid4(),
                    b,
                    f"{b} memory {i}",
                )

        cols = await _discover_vector_columns(conn, schema_name)
        memory_col = next(c for c in cols if c.table == "memory_units")
        memory_col.fq_table = f'"{schema_name}"."memory_units"'

        # Re-embed only bank_a
        p, _ = await _reembed_table(conn, embeddings, memory_col, 8, bank_a)
        assert p == 2

        # Bank A should have embeddings, Bank B should not
        a_with = await conn.fetchval(
            f"SELECT COUNT(*) FROM {schema_name}.memory_units WHERE bank_id = $1 AND embedding IS NOT NULL",
            bank_a,
        )
        b_with = await conn.fetchval(
            f"SELECT COUNT(*) FROM {schema_name}.memory_units WHERE bank_id = $1 AND embedding IS NOT NULL",
            bank_b,
        )
        assert a_with == 2
        assert b_with == 0
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_skips_rows_with_empty_text(reindex_test_schema):
    """Rows with NULL or empty text should be skipped (count == 0 for them)."""
    db_url, schema_name, _ = reindex_test_schema
    conn = await asyncpg.connect(db_url)
    bank_id = f"test-{uuid.uuid4().hex[:8]}"
    try:
        await conn.execute(
            f'INSERT INTO {schema_name}.banks (bank_id) VALUES ($1) ON CONFLICT DO NOTHING',
            bank_id,
        )
        # 1 row with text, 1 with empty, 1 with NULL — only the first should count
        await conn.execute(
            f"INSERT INTO {schema_name}.memory_units (id, bank_id, text, fact_type) "
            f"VALUES ($1, $2, $3, 'world')",
            uuid.uuid4(), bank_id, "actual content",
        )
        await conn.execute(
            f"INSERT INTO {schema_name}.memory_units (id, bank_id, text, fact_type) "
            f"VALUES ($1, $2, $3, 'world')",
            uuid.uuid4(), bank_id, "",
        )
        # NULL text — depends on schema NOT NULL constraint; skip if it would fail
        try:
            await conn.execute(
                f"INSERT INTO {schema_name}.memory_units (id, bank_id, fact_type) "
                f"VALUES ($1, $2, 'world')",
                uuid.uuid4(), bank_id,
            )
        except Exception:
            pass  # NOT NULL constraint, fine

        cols = await _discover_vector_columns(conn, schema_name)
        memory_col = next(c for c in cols if c.table == "memory_units")
        memory_col.fq_table = f'"{schema_name}"."memory_units"'

        n = await _count_pending(conn, memory_col)
        # Only the row with non-empty text should be counted
        assert n == 1
    finally:
        await conn.close()

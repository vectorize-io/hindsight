"""
Tests for admin backup and restore functionality.

These tests use an isolated schema to avoid interfering with other tests.
The backup/restore operations truncate tables, which would cause deadlocks
and race conditions if run against the shared public schema.
"""

import tempfile
import uuid
import zipfile
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio

from hindsight_api.admin.cli import _backup, _restore, BACKUP_TABLES
from hindsight_api.migrations import run_migrations


# Run these tests sequentially since they do full DB backup/restore
pytestmark = pytest.mark.xdist_group(name="backup_restore")


@pytest_asyncio.fixture(scope="function")
async def backup_test_schema(pg0_db_url, embeddings):
    """Create an isolated schema for backup/restore tests.

    Uses a unique schema name per test invocation to avoid conflicts with
    parallel test runs or leftover state from interrupted runs.

    Returns a tuple of (db_url, schema_name, fq_helper, embeddings).
    """
    # Initialize embeddings if not already done
    await embeddings.initialize()

    # Use unique schema name to avoid conflicts
    schema_name = f"backup_test_{uuid.uuid4().hex[:8]}"

    def _fq(table: str) -> str:
        """Get fully-qualified table name in test schema."""
        return f"{schema_name}.{table}"

    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(f"CREATE SCHEMA {schema_name}")
    finally:
        await conn.close()

    # Run migrations on the isolated schema
    run_migrations(pg0_db_url, schema=schema_name)

    yield pg0_db_url, schema_name, _fq, embeddings

    # Cleanup after test
    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_backup_restore_roundtrip(backup_test_schema):
    """Test that backup and restore preserves all data correctly."""
    db_url, schema_name, _fq, embeddings = backup_test_schema
    bank_id = f"test-backup-{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_url)

    try:
        # Create a bank
        await conn.execute(
            f"INSERT INTO {_fq('banks')} (bank_id) VALUES ($1) ON CONFLICT DO NOTHING",
            bank_id,
        )

        # Create some test memory units with embeddings
        # Convert embedding list to pgvector format string
        embedding_list = embeddings.encode(["Test content about Alice"])[0]
        embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"
        for text in [
            "Alice is a software engineer who loves Python.",
            "Bob works with Alice on the backend team.",
            "The team uses PostgreSQL for their database.",
        ]:
            await conn.execute(
                f"""INSERT INTO {_fq('memory_units')}
                    (bank_id, text, fact_type, embedding, event_date)
                    VALUES ($1, $2, 'world', $3::vector, NOW())""",
                bank_id,
                text,
                embedding_str,
            )

        # Get counts before backup
        counts_before = {}
        for table in BACKUP_TABLES:
            counts_before[table] = await conn.fetchval(f"SELECT COUNT(*) FROM {_fq(table)}")

        # Verify we have data
        assert counts_before["banks"] > 0
        assert counts_before["memory_units"] > 0

    finally:
        await conn.close()

    # Backup to a temp file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        backup_path = Path(f.name)

    try:
        manifest = await _backup(db_url, backup_path, schema=schema_name)

        # Verify backup file exists and is valid
        assert backup_path.exists()
        assert backup_path.stat().st_size > 0

        # Verify manifest
        assert manifest["version"] == "1"
        assert "created_at" in manifest
        for table in BACKUP_TABLES:
            assert table in manifest["tables"]
            assert manifest["tables"][table]["rows"] == counts_before[table]

        # Verify zip contents
        with zipfile.ZipFile(backup_path, "r") as zf:
            assert "manifest.json" in zf.namelist()
            for table in BACKUP_TABLES:
                assert f"{table}.bin" in zf.namelist()

        # Clear all data
        conn = await asyncpg.connect(db_url)
        try:
            for table in reversed(BACKUP_TABLES):
                await conn.execute(f"TRUNCATE TABLE {_fq(table)} CASCADE")

            # Verify data is gone
            for table in BACKUP_TABLES:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {_fq(table)}")
                assert count == 0, f"Table {table} should be empty after truncate"
        finally:
            await conn.close()

        # Restore from backup
        await _restore(db_url, backup_path, schema=schema_name)

        # Verify counts match original
        conn = await asyncpg.connect(db_url)
        try:
            for table in BACKUP_TABLES:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {_fq(table)}")
                assert count == counts_before[table], f"Table {table} count mismatch after restore"

            # Verify data content is preserved
            texts = await conn.fetch(
                f"SELECT text FROM {_fq('memory_units')} WHERE bank_id = $1",
                bank_id,
            )
            text_content = " ".join(r["text"] for r in texts)
            assert "Alice" in text_content or "software" in text_content
        finally:
            await conn.close()

    finally:
        # Cleanup
        if backup_path.exists():
            backup_path.unlink()


@pytest.mark.asyncio
async def test_backup_restore_preserves_all_column_types(backup_test_schema):
    """Test that all column types are preserved: vectors, UUIDs, timestamps, JSONB."""
    db_url, schema_name, _fq, embeddings = backup_test_schema
    bank_id = f"test-types-{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_url)

    try:
        # Create a bank
        await conn.execute(
            f"INSERT INTO {_fq('banks')} (bank_id) VALUES ($1) ON CONFLICT DO NOTHING",
            bank_id,
        )

        # Create a memory unit with all column types
        # Convert embedding list to pgvector format string
        embedding_list = embeddings.encode(["John Smith engineer"])[0]
        embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"
        await conn.execute(
            f"""INSERT INTO {_fq('memory_units')}
                (bank_id, text, fact_type, embedding, event_date, metadata)
                VALUES ($1, $2, 'world', $3::vector, NOW(), $4)""",
            bank_id,
            "John Smith is a senior engineer at Acme Corp since 2020.",
            embedding_str,
            '{"key": "value"}',
        )

        # Create an entity
        await conn.execute(
            f"""INSERT INTO {_fq('entities')}
                (bank_id, canonical_name, metadata)
                VALUES ($1, $2, $3)""",
            bank_id,
            "John Smith",
            '{"role": "engineer"}',
        )

        # Get original data
        original_unit = await conn.fetchrow(
            f"""SELECT id, embedding, event_date, created_at, metadata, text
               FROM {_fq('memory_units')} WHERE bank_id = $1 LIMIT 1""",
            bank_id,
        )
        original_entity = await conn.fetchrow(
            f"""SELECT id, first_seen, last_seen, metadata, canonical_name
               FROM {_fq('entities')} WHERE bank_id = $1 LIMIT 1""",
            bank_id,
        )
        original_bank = await conn.fetchrow(
            f"SELECT bank_id, created_at, updated_at FROM {_fq('banks')} WHERE bank_id = $1",
            bank_id,
        )
    finally:
        await conn.close()

    assert original_unit is not None, "Should have created memory units"
    assert original_unit["embedding"] is not None, "Should have embedding"
    assert original_unit["id"] is not None, "Should have UUID"
    assert original_entity is not None, "Should have created entities"

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        backup_path = Path(f.name)

    try:
        await _backup(db_url, backup_path, schema=schema_name)

        # Clear all data
        conn = await asyncpg.connect(db_url)
        try:
            for table in reversed(BACKUP_TABLES):
                await conn.execute(f"TRUNCATE TABLE {_fq(table)} CASCADE")
        finally:
            await conn.close()

        await _restore(db_url, backup_path, schema=schema_name)

        # Verify all column types are preserved exactly
        conn = await asyncpg.connect(db_url)
        try:
            restored_unit = await conn.fetchrow(
                f"""SELECT id, embedding, event_date, created_at, metadata, text
                   FROM {_fq('memory_units')} WHERE bank_id = $1 LIMIT 1""",
                bank_id,
            )
            restored_entity = await conn.fetchrow(
                f"""SELECT id, first_seen, last_seen, metadata, canonical_name
                   FROM {_fq('entities')} WHERE bank_id = $1 LIMIT 1""",
                bank_id,
            )
            restored_bank = await conn.fetchrow(
                f"SELECT bank_id, created_at, updated_at FROM {_fq('banks')} WHERE bank_id = $1",
                bank_id,
            )
        finally:
            await conn.close()

        # Verify memory_units
        assert restored_unit is not None, "Should have restored memory unit"
        assert restored_unit["id"] == original_unit["id"], "UUID should match exactly"
        assert restored_unit["text"] == original_unit["text"], "Text should match"
        assert list(restored_unit["embedding"]) == list(original_unit["embedding"]), "Vector embedding should match exactly"
        assert restored_unit["event_date"] == original_unit["event_date"], "Timestamp should match exactly"
        assert restored_unit["created_at"] == original_unit["created_at"], "Created timestamp should match"
        assert restored_unit["metadata"] == original_unit["metadata"], "JSONB metadata should match"

        # Verify entities
        assert restored_entity is not None, "Should have restored entity"
        assert restored_entity["id"] == original_entity["id"], "Entity UUID should match"
        assert restored_entity["canonical_name"] == original_entity["canonical_name"], "Entity name should match"
        assert restored_entity["first_seen"] == original_entity["first_seen"], "Entity first_seen should match"
        assert restored_entity["last_seen"] == original_entity["last_seen"], "Entity last_seen should match"
        assert restored_entity["metadata"] == original_entity["metadata"], "Entity metadata should match"

        # Verify banks
        assert restored_bank is not None, "Should have restored bank"
        assert restored_bank["bank_id"] == original_bank["bank_id"], "Bank ID should match"
        assert restored_bank["created_at"] == original_bank["created_at"], "Bank created_at should match"

    finally:
        if backup_path.exists():
            backup_path.unlink()

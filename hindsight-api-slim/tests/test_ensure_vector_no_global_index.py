"""The global memory_units vector index on per-bank backends: never created, migrated away.

For per-bank backends (pgvector / pgvectorscale / vchord) every vector search is
bank + fact_type scoped and served by the per-(bank, fact_type) partial indexes
created at bank-creation time. The global `idx_memory_units_embedding` is never
chosen by the planner (migration d5e6f7a8b9c0 drops it for exactly this reason).

Contract under test:
- the post-migration reconcile (`ensure_vector_extension`) must not recreate the
  index on a fresh schema;
- the reconcile must be hands-off when a legacy schema still carries the index
  (no runtime DROP INDEX — that DDL belongs in the migration path);
- migration f2a6d8c4b1e9 removes the leftover index.
"""

import asyncio
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, create_engine, text

import hindsight_api
from hindsight_api._vector_index import uses_per_bank_vector_indexes
from hindsight_api.config import HindsightConfig
from hindsight_api.migrations import ensure_vector_extension, run_migrations, to_libpq_url


@pytest.fixture(scope="module")
def vec_db_url():
    """A dedicated pg0 instance so the test owns its schema/index state."""
    from hindsight_api.pg0 import EmbeddedPostgres

    pg0 = EmbeddedPostgres(name="hindsight-vecidx-test", port=5570)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(pg0.ensure_running())
    finally:
        loop.close()


def _skip_unless_per_bank_backend() -> str:
    vec = HindsightConfig.from_env().vector_extension
    if not uses_per_bank_vector_indexes(vec):
        pytest.skip(f"backend {vec!r} uses a global vector index by design (no per-bank indexes)")
    return vec


def _global_index_count(conn: Connection, schema: str) -> int:
    return conn.execute(
        text(
            "SELECT COUNT(*) FROM pg_indexes "
            "WHERE schemaname = :schema AND tablename = 'memory_units' "
            "AND indexname = 'idx_memory_units_embedding'"
        ),
        {"schema": schema},
    ).scalar()


def _reset_schema(db_url: str, schema: str) -> None:
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            conn.commit()
    finally:
        engine.dispose()


# The repair migration under test: drops the stale global memory_units index.
_REPAIR_REVISION = "f2a6d8c4b1e9"


def _alembic_config(db_url: str, schema: str) -> Config:
    """Programmatic alembic config matching what run_migrations builds."""
    cfg = Config()
    cfg.set_main_option("script_location", str(Path(hindsight_api.__file__).parent / "alembic"))
    cfg.set_main_option("sqlalchemy.url", to_libpq_url(db_url))
    cfg.set_main_option("path_separator", "os")
    cfg.set_main_option("target_schema", schema)
    return cfg


@pytest.mark.xdist_group("vecidx_pg0")
def test_per_bank_backend_does_not_create_global_memory_units_index(vec_db_url):
    vec = _skip_unless_per_bank_backend()

    schema = "vecidx_fresh"
    _reset_schema(vec_db_url, schema)

    run_migrations(vec_db_url, schema=schema)
    # Fresh, empty schema (no banks yet) → the reconcile must be a no-op for the
    # global index, not recreate it.
    ensure_vector_extension(vec_db_url, vector_extension=vec, schema=schema)

    engine = create_engine(vec_db_url)
    try:
        with engine.connect() as conn:
            global_index_count = _global_index_count(conn, schema)
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            conn.commit()
    finally:
        engine.dispose()

    assert global_index_count == 0


@pytest.mark.xdist_group("vecidx_pg0")
def test_reconcile_leaves_stale_global_index_untouched(vec_db_url):
    """The reconcile must not perform runtime index DDL: a leftover global index
    stays until the migration removes it (no surprise ACCESS EXCLUSIVE locks at
    startup/provisioning time)."""
    vec = _skip_unless_per_bank_backend()

    schema = "vecidx_stale_global"
    _reset_schema(vec_db_url, schema)
    run_migrations(vec_db_url, schema=schema)

    engine = create_engine(vec_db_url)
    try:
        # Simulate the legacy state: an older reconcile recreated the global
        # index after migration d5e6f7a8b9c0 dropped it.
        with engine.connect() as conn:
            conn.execute(
                text(
                    f'CREATE INDEX idx_memory_units_embedding ON "{schema}".memory_units '
                    "USING hnsw (embedding vector_cosine_ops)"
                )
            )
            conn.commit()
            assert _global_index_count(conn, schema) == 1

        ensure_vector_extension(vec_db_url, vector_extension=vec, schema=schema)

        with engine.connect() as conn:
            count_after = _global_index_count(conn, schema)
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            conn.commit()
    finally:
        engine.dispose()

    assert count_after == 1


@pytest.mark.xdist_group("vecidx_pg0")
def test_migration_drops_stale_global_index(vec_db_url):
    """Upgrading from the pre-repair revision removes a leftover global index."""
    _skip_unless_per_bank_backend()

    schema = "vecidx_migration_repair"
    _reset_schema(vec_db_url, schema)
    run_migrations(vec_db_url, schema=schema)

    cfg = _alembic_config(vec_db_url, schema)
    # Step back below the repair migration, then plant the legacy index the
    # way an older reconcile would have left it.
    #
    # The parent is resolved from the revision map rather than written as
    # "f2a6d8c4b1e9@-1": that is alembic's branch@relative syntax, which counts
    # back from the *head* of the branch containing the revision, not from the
    # revision itself. It meant "the parent" only while the repair migration was
    # head, so the first migration added on top of it (b3e8d1c6f4a9) made this
    # downgrade stop ON the repair migration instead of below it — the stale index
    # was then planted after the drop had already run, the upgrade never re-ran it,
    # and the test failed for reasons unrelated to the behaviour under test.
    parent = ScriptDirectory.from_config(cfg).get_revision(_REPAIR_REVISION).down_revision
    assert isinstance(parent, str), f"expected a single parent for {_REPAIR_REVISION}, got {parent!r}"
    command.downgrade(cfg, parent)

    engine = create_engine(vec_db_url)
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    f'CREATE INDEX idx_memory_units_embedding ON "{schema}".memory_units '
                    "USING hnsw (embedding vector_cosine_ops)"
                )
            )
            conn.commit()
            assert _global_index_count(conn, schema) == 1

        command.upgrade(cfg, "heads")

        with engine.connect() as conn:
            count_after = _global_index_count(conn, schema)
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            conn.commit()
    finally:
        engine.dispose()

    assert count_after == 0

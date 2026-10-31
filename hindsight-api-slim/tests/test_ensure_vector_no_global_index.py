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
- migration f2a6d8c4b1e9 removes the leftover index;
- the startup embedding-dimension resize does not build one either, and migration
  e8c1a4f7b2d9 removes any global vector index it left behind, whatever its name (#5707).
"""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, create_engine, text

import hindsight_api
from hindsight_api._vector_index import uses_per_bank_vector_indexes
from hindsight_api.config import HindsightConfig
from hindsight_api.migrations import (
    ensure_embedding_dimension,
    ensure_vector_extension,
    run_migrations,
    to_libpq_url,
)


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


def _non_partial_vector_index_names(conn: Connection, schema: str) -> list[str]:
    """Every global (non-partial) vector index on memory_units, whatever its name."""
    rows = conn.execute(
        text(
            "SELECT i.relname FROM pg_index x "
            "JOIN pg_class t ON t.oid = x.indrelid "
            "JOIN pg_namespace n ON n.oid = t.relnamespace "
            "JOIN pg_class i ON i.oid = x.indexrelid "
            "JOIN pg_am am ON am.oid = i.relam "
            "WHERE n.nspname = :schema AND t.relname = 'memory_units' "
            "AND x.indpred IS NULL AND am.amname IN ('hnsw', 'vchordrq', 'diskann', 'scann')"
        ),
        {"schema": schema},
    ).fetchall()
    return [row[0] for row in rows]


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
def test_per_bank_backend_does_not_create_global_memory_units_index(fresh_db_url):
    vec = _skip_unless_per_bank_backend()

    schema = "vecidx_fresh"
    _reset_schema(fresh_db_url, schema)

    run_migrations(fresh_db_url, schema=schema)
    # Fresh, empty schema (no banks yet) → the reconcile must be a no-op for the
    # global index, not recreate it.
    ensure_vector_extension(fresh_db_url, vector_extension=vec, schema=schema)

    engine = create_engine(fresh_db_url)
    try:
        with engine.connect() as conn:
            global_index_count = _global_index_count(conn, schema)
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            conn.commit()
    finally:
        engine.dispose()

    assert global_index_count == 0


@pytest.mark.xdist_group("vecidx_pg0")
def test_reconcile_leaves_stale_global_index_untouched(fresh_db_url):
    """The reconcile must not perform runtime index DDL: a leftover global index
    stays until the migration removes it (no surprise ACCESS EXCLUSIVE locks at
    startup/provisioning time)."""
    vec = _skip_unless_per_bank_backend()

    schema = "vecidx_stale_global"
    _reset_schema(fresh_db_url, schema)
    run_migrations(fresh_db_url, schema=schema)

    engine = create_engine(fresh_db_url)
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

        ensure_vector_extension(fresh_db_url, vector_extension=vec, schema=schema)

        with engine.connect() as conn:
            count_after = _global_index_count(conn, schema)
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            conn.commit()
    finally:
        engine.dispose()

    assert count_after == 1


@pytest.mark.xdist_group("vecidx_pg0")
@pytest.mark.memory_backend_incompatible
def test_migration_drops_stale_global_index(fresh_db_url):
    """Upgrading from the pre-repair revision removes a leftover global index."""
    _skip_unless_per_bank_backend()

    schema = "vecidx_migration_repair"
    _reset_schema(fresh_db_url, schema)
    run_migrations(fresh_db_url, schema=schema)

    cfg = _alembic_config(fresh_db_url, schema)
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

    engine = create_engine(fresh_db_url)
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


@pytest.mark.xdist_group("vecidx_pg0")
def test_embedding_dimension_resize_does_not_create_global_index(fresh_db_url):
    """#5707: a fresh schema whose model dimension differs from the column default
    goes through the resize path, which must not build a global memory_units index."""
    vec = _skip_unless_per_bank_backend()

    schema = "vecidx_resize"
    _reset_schema(fresh_db_url, schema)
    run_migrations(fresh_db_url, schema=schema)
    ensure_embedding_dimension(fresh_db_url, 1024, schema=schema, vector_extension=vec)
    ensure_vector_extension(fresh_db_url, vector_extension=vec, schema=schema)

    engine = create_engine(fresh_db_url)
    try:
        with engine.connect() as conn:
            leftover = _non_partial_vector_index_names(conn, schema)
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            conn.commit()
    finally:
        engine.dispose()

    assert leftover == []


@pytest.mark.xdist_group("vecidx_pg0")
@pytest.mark.memory_backend_incompatible
def test_migration_drops_global_index_by_any_name(fresh_db_url):
    """#5707: the resize path left ``idx_memory_units_embedding_hnsw`` behind, which
    f2a6d8c4b1e9 never matched. e8c1a4f7b2d9 drops every global vector index found in
    the catalog, and keeps the per-bank partial ones."""
    _skip_unless_per_bank_backend()

    schema = "vecidx_migration_any_name"
    _reset_schema(fresh_db_url, schema)
    run_migrations(fresh_db_url, schema=schema)

    cfg = _alembic_config(fresh_db_url, schema)
    parent = ScriptDirectory.from_config(cfg).get_revision("e8c1a4f7b2d9").down_revision
    assert isinstance(parent, str)
    command.downgrade(cfg, parent)

    engine = create_engine(fresh_db_url)
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    f'CREATE INDEX idx_memory_units_embedding_hnsw ON "{schema}".memory_units '
                    "USING hnsw (embedding vector_cosine_ops)"
                )
            )
            conn.execute(
                text(
                    f'CREATE INDEX idx_mu_emb_partial ON "{schema}".memory_units '
                    "USING hnsw (embedding vector_cosine_ops) WHERE fact_type = 'world'"
                )
            )
            conn.commit()

        command.upgrade(cfg, "heads")

        with engine.connect() as conn:
            leftover = _non_partial_vector_index_names(conn, schema)
            partial_kept = conn.execute(
                text("SELECT COUNT(*) FROM pg_indexes WHERE schemaname = :s AND indexname = 'idx_mu_emb_partial'"),
                {"s": schema},
            ).scalar()
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            conn.commit()
    finally:
        engine.dispose()

    assert leftover == []
    assert partial_kept == 1

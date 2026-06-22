"""ensure_vector_extension must not create the (unused) global memory_units index.

For per-bank backends (pgvector / pgvectorscale / vchord) every vector search is
bank + fact_type scoped and served by the per-(bank, fact_type) partial indexes
created at bank-creation time. The global `idx_memory_units_embedding` is never
chosen by the planner (migration d5e6f7a8b9c0 drops it for exactly this reason),
so the post-migration reconcile must not recreate it on a fresh schema.
"""

import asyncio

import pytest
from sqlalchemy import create_engine, text

from hindsight_api._vector_index import uses_per_bank_vector_indexes
from hindsight_api.config import HindsightConfig
from hindsight_api.migrations import ensure_vector_extension, run_migrations


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


def test_per_bank_backend_does_not_create_global_memory_units_index(vec_db_url):
    config = HindsightConfig.from_env()
    vec = config.vector_extension
    if not uses_per_bank_vector_indexes(vec):
        pytest.skip(f"backend {vec!r} uses a global vector index by design (no per-bank indexes)")

    schema = "vecidx_fresh"

    engine = create_engine(vec_db_url)
    try:
        with engine.connect() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            conn.commit()
    finally:
        engine.dispose()

    run_migrations(vec_db_url, schema=schema)
    # Fresh, empty schema (no banks yet) → the reconcile must be a no-op for the
    # global index, not recreate it.
    ensure_vector_extension(vec_db_url, vector_extension=vec, schema=schema)

    engine = create_engine(vec_db_url)
    try:
        with engine.connect() as conn:
            global_index_count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM pg_indexes "
                    "WHERE schemaname = :schema AND tablename = 'memory_units' "
                    "AND indexname = 'idx_memory_units_embedding'"
                ),
                {"schema": schema},
            ).scalar()
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            conn.commit()
    finally:
        engine.dispose()

    assert global_index_count == 0

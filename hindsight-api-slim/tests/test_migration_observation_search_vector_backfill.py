"""Regression for the consolidator search_vector gap (PR #2425).

Observations written by the consolidator under the ``native`` text-search
backend landed with a NULL ``search_vector`` and were invisible to BM25. The
writer is fixed to populate the tsvector; migration
``c3f7a1b9d2e4`` backfills the historical NULL observations.

This test seeds an observation with a NULL ``search_vector`` at the revision
just before the backfill, runs the migration to head, and asserts the row is
populated with a valid tsvector — and that already-populated rows are left
untouched. Uses a dedicated pg0 instance (mirrors test_migration_backsweep) so
it controls exactly which migrations have run and never stamps the shared test
instance.
"""

import asyncio
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

_SCRIPT_LOCATION = str(Path(__file__).parent.parent / "hindsight_api" / "alembic")

# Revision immediately before the backfill migration.
_PRE_BACKFILL_REVISION = "f4d1c2b3a5e6"
_BACKFILL_REVISION = "c3f7a1b9d2e4"


def _alembic_cfg(db_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", _SCRIPT_LOCATION)
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("prepend_sys_path", ".")
    cfg.set_main_option("path_separator", "os")
    return cfg


@pytest.fixture(scope="module")
def pre_backfill_db_url():
    """pg0 instance brought to the revision just before the backfill so the
    migration's UPDATE runs against seeded NULL-search_vector observations."""
    from hindsight_api.pg0 import EmbeddedPostgres

    # port=None lets pg0 auto-assign a free port. A hardcoded port is not
    # xdist-safe: under `-n` the fixed port collides with a concurrent or
    # left-over postgres ("could not bind 127.0.0.1: Address already in use"),
    # which the retry loop then reports as "Failed to start embedded PostgreSQL
    # after 5 attempts". The connection URL from ensure_running() carries the
    # assigned port, so nothing downstream needs to know it.
    pg0 = EmbeddedPostgres(name="hindsight-obs-sv-backfill-test", port=None)
    loop = asyncio.new_event_loop()
    try:
        url = loop.run_until_complete(pg0.ensure_running())
    finally:
        loop.close()

    # pg0 data dirs persist across runs, so normalise: go to head, then down to
    # just before the backfill.
    command.upgrade(_alembic_cfg(url), "heads")
    command.downgrade(_alembic_cfg(url), _PRE_BACKFILL_REVISION)
    return url


def test_backfill_populates_null_observation_search_vector(pre_backfill_db_url):
    db_url = pre_backfill_db_url
    bank_id = f"obs-sv-{uuid.uuid4().hex[:12]}"

    null_obs_id = uuid.uuid4()
    populated_obs_id = uuid.uuid4()
    world_id = uuid.uuid4()

    engine = create_engine(db_url)
    with engine.connect() as conn:
        # Sanity: under the default native backend search_vector is a regular
        # tsvector column; otherwise this test wouldn't exercise the gate.
        udt = conn.execute(
            text(
                """
                SELECT udt_name FROM information_schema.columns
                WHERE table_name = 'memory_units' AND column_name = 'search_vector'
                """
            )
        ).scalar()
        assert udt == "tsvector", f"expected native tsvector backend, got {udt!r}"

        conn.execute(text("INSERT INTO banks (bank_id) VALUES (:b)"), {"b": bank_id})

        # The bug shape: an observation with NULL search_vector.
        conn.execute(
            text(
                """
                INSERT INTO memory_units (id, bank_id, text, fact_type, search_vector)
                VALUES (:id, :b, 'Django uses middleware for request processing', 'observation', NULL)
                """
            ),
            {"id": null_obs_id, "b": bank_id},
        )
        # An observation already populated — must be left byte-for-byte intact.
        conn.execute(
            text(
                """
                INSERT INTO memory_units (id, bank_id, text, fact_type, search_vector)
                VALUES (:id, :b, 'Already indexed observation', 'observation',
                        to_tsvector('english', 'Already indexed observation'))
                """
            ),
            {"id": populated_obs_id, "b": bank_id},
        )
        # A non-observation row with NULL search_vector — must NOT be touched
        # (the migration is scoped to fact_type = 'observation').
        conn.execute(
            text(
                """
                INSERT INTO memory_units (id, bank_id, text, fact_type, search_vector)
                VALUES (:id, :b, 'A world fact', 'world', NULL)
                """
            ),
            {"id": world_id, "b": bank_id},
        )
        conn.commit()

    # Run the backfill.
    command.upgrade(_alembic_cfg(db_url), _BACKFILL_REVISION)

    with engine.connect() as conn:
        null_obs_sv, null_obs_match = conn.execute(
            text(
                """
                SELECT search_vector IS NOT NULL,
                       search_vector @@ plainto_tsquery('english', 'middleware')
                FROM memory_units WHERE id = :id
                """
            ),
            {"id": null_obs_id},
        ).fetchone()
        assert null_obs_sv, "backfill must populate the NULL observation's search_vector"
        assert null_obs_match, "backfilled tsvector must be BM25-searchable on its own text"

        populated_match = conn.execute(
            text("SELECT search_vector @@ plainto_tsquery('english', 'indexed') FROM memory_units WHERE id = :id"),
            {"id": populated_obs_id},
        ).scalar()
        assert populated_match, "pre-populated observation must remain searchable"

        world_null = conn.execute(
            text("SELECT search_vector IS NULL FROM memory_units WHERE id = :id"),
            {"id": world_id},
        ).scalar()
        assert world_null, "non-observation rows must be left untouched by the backfill"

    # Idempotency: re-running touches nothing and stays at head.
    command.upgrade(_alembic_cfg(db_url), "heads")
    with engine.connect() as conn:
        still_populated = conn.execute(
            text("SELECT search_vector IS NOT NULL FROM memory_units WHERE id = :id"),
            {"id": null_obs_id},
        ).scalar()
        assert still_populated

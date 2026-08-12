"""Tests for migration c4f7a91b2d38 (entity_maintenance_queue + its seed).

The graph-maintenance entity prune is queue-driven (#3222): it only looks at
entities something enqueued. That leaves one class of garbage nothing can
enqueue — entities already stranded *before* the queue existed, whose postings
are long gone and which no future delete will ever name. The migration's seed
insert is the only thing that reclaims those, so it is worth a test of its own:
without it the upgrade silently strands every orphan a bank had accumulated
while its bank-wide sweep was failing, which is the exact population #3222 is
about.

Uses a dedicated pg0 instance so the test controls which migrations have run.
"""

import asyncio
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

_SCRIPT_LOCATION = str(Path(__file__).parent.parent / "hindsight_api" / "alembic")

# The revision immediately before the one under test.
_PRE_REVISION = "d9c1a7b4e2f6"
_REVISION = "c4f7a91b2d38"


def _alembic_cfg(db_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", _SCRIPT_LOCATION)
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("prepend_sys_path", ".")
    cfg.set_main_option("path_separator", "os")
    return cfg


def _upgrade(db_url: str, revision: str) -> None:
    command.upgrade(_alembic_cfg(db_url), revision)


def _reset_public_schema(db_url: str) -> None:
    engine = create_engine(db_url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as conn:
            conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
    finally:
        engine.dispose()


@pytest.fixture(scope="module")
def pre_queue_db_url() -> str:
    """A dedicated database migrated to the revision just before the queue."""
    from hindsight_api.pg0 import EmbeddedPostgres

    pg0 = EmbeddedPostgres(name="hindsight-entity-queue-test", port=5563)
    loop = asyncio.new_event_loop()
    try:
        url = loop.run_until_complete(pg0.ensure_running())
    finally:
        loop.close()

    _reset_public_schema(url)
    _upgrade(url, _PRE_REVISION)
    return url


def test_seed_enqueues_every_pre_existing_entity(pre_queue_db_url: str) -> None:
    """Every entity that predates the queue becomes a candidate exactly once.

    Both kinds have to be seeded, not just the visibly-orphaned one: the drain
    is what decides which are dead, and it cannot decide about a row it never
    sees. Seeding only the orphans would also mean re-running the bank-wide
    NOT EXISTS the migration exists to get rid of.
    """
    db_url = pre_queue_db_url
    engine = create_engine(db_url)
    bank_id = f"bank_{uuid.uuid4().hex[:8]}"

    try:
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO banks (bank_id) VALUES (:b)"), {"b": bank_id})
            unit_id = conn.execute(
                text(
                    "INSERT INTO memory_units (bank_id, text, fact_type, event_date) "
                    "VALUES (:b, 'a fact', 'world', now()) RETURNING id"
                ),
                {"b": bank_id},
            ).scalar_one()
            referenced = conn.execute(
                text("INSERT INTO entities (bank_id, canonical_name) VALUES (:b, 'referenced') RETURNING id"),
                {"b": bank_id},
            ).scalar_one()
            orphan = conn.execute(
                text("INSERT INTO entities (bank_id, canonical_name) VALUES (:b, 'stranded') RETURNING id"),
                {"b": bank_id},
            ).scalar_one()
            conn.execute(
                text("INSERT INTO unit_entities (unit_id, entity_id) VALUES (:u, :e)"),
                {"u": unit_id, "e": referenced},
            )

        _upgrade(db_url, _REVISION)

        with engine.connect() as conn:
            queued = {
                str(row[0])
                for row in conn.execute(
                    text("SELECT entity_id FROM entity_maintenance_queue WHERE bank_id = :b"), {"b": bank_id}
                )
            }
        assert queued == {str(referenced), str(orphan)}
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM banks WHERE bank_id = :b"), {"b": bank_id})
        engine.dispose()

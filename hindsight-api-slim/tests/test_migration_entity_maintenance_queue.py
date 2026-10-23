"""Tests for migration c4f7a91b2d38 (entity_maintenance_queue).

Two properties the prune depends on and which are easy to lose in a later edit:
the composite primary key (it is what collapses overlapping deletes into one
row, and what the #3034 locking upsert conflicts on), and that the upgrade does
NOT backfill existing entities — a backfill would write a row per entity inside
a migration that runs at API startup and charge a prune check for each.

Uses a database of its own (the ``fresh_db_url`` fixture) so the test controls which migrations have run.
"""

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


@pytest.fixture(scope="module")
def pre_queue_db_url(fresh_db_url) -> str:
    """A dedicated database migrated to the revision just before the queue."""
    url = fresh_db_url
    _upgrade(url, _PRE_REVISION)
    return url


def test_upgrade_creates_an_empty_deduping_queue(pre_queue_db_url: str) -> None:
    """The upgrade adds the queue but enqueues nothing, and the key dedupes.

    A backfill here is tempting — it would reclaim what a bank stranded while its
    sweep was failing — but it costs one row per entity written during startup
    migration plus a prune check each, to collect rows that cost the bank
    nothing. New deletes fill the queue; historical strays stay.
    """
    db_url = pre_queue_db_url
    engine = create_engine(db_url)
    bank_id = f"bank_{uuid.uuid4().hex[:8]}"

    try:
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO banks (bank_id) VALUES (:b)"), {"b": bank_id})
            for name in ("referenced", "stranded"):
                conn.execute(
                    text("INSERT INTO entities (bank_id, canonical_name) VALUES (:b, :n)"),
                    {"b": bank_id, "n": name},
                )

        _upgrade(db_url, _REVISION)

        with engine.connect() as conn:
            queued = conn.execute(
                text("SELECT count(*) FROM entity_maintenance_queue WHERE bank_id = :b"), {"b": bank_id}
            ).scalar_one()
            assert queued == 0, "the upgrade must not backfill existing entities"

            entity_id = conn.execute(
                text("SELECT id FROM entities WHERE bank_id = :b LIMIT 1"), {"b": bank_id}
            ).scalar_one()

        # The composite key is what makes overlapping deletes collapse to one row.
        with engine.begin() as conn:
            for _ in range(2):
                conn.execute(
                    text(
                        "INSERT INTO entity_maintenance_queue (bank_id, entity_id) VALUES (:b, :e) "
                        "ON CONFLICT (bank_id, entity_id) DO UPDATE "
                        "SET enqueued_at = entity_maintenance_queue.enqueued_at"
                    ),
                    {"b": bank_id, "e": entity_id},
                )
        with engine.connect() as conn:
            assert (
                conn.execute(
                    text("SELECT count(*) FROM entity_maintenance_queue WHERE bank_id = :b"), {"b": bank_id}
                ).scalar_one()
                == 1
            )
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM entity_maintenance_queue WHERE bank_id = :b"), {"b": bank_id})
            conn.execute(text("DELETE FROM banks WHERE bank_id = :b"), {"b": bank_id})
        engine.dispose()

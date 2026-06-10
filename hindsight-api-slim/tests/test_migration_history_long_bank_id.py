"""Regression for issue #2106: the split-history migration must not truncate
``bank_id`` on PostgreSQL.

``a7b8c9d0e1f2`` originally declared ``observation_history.bank_id`` as
``VARCHAR(64)`` while its backfill source ``memory_units.bank_id`` is ``TEXT``.
A bank_id longer than 64 chars aborted the backfill with
``StringDataRightTruncation``, rolled back the migration, and bricked startup.

This test seeds a 78-char bank_id (the shape reported in the issue) at the
revision just before the migration, then runs the migration up to head and
asserts the row is backfilled intact and both history tables expose a ``TEXT``
``bank_id``. Uses a dedicated pg0 instance (mirrors test_migration_backsweep)
so we control exactly which migrations have run.
"""

import asyncio
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

_SCRIPT_LOCATION = str(Path(__file__).parent.parent / "hindsight_api" / "alembic")

# Revision immediately before the split-history migration; at this point
# memory_units.history still exists and the history tables do not.
_PRE_SPLIT_REVISION = "d3e4f5a6b7c8"
_SPLIT_REVISION = "a7b8c9d0e1f2"


def _alembic_cfg(db_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", _SCRIPT_LOCATION)
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("prepend_sys_path", ".")
    cfg.set_main_option("path_separator", "os")
    return cfg


@pytest.fixture(scope="module")
def pre_split_db_url():
    """pg0 instance with schema at the revision just before the split-history
    migration so the migration's backfill actually runs against seeded data."""
    from hindsight_api.pg0 import EmbeddedPostgres

    pg0 = EmbeddedPostgres(name="hindsight-long-bankid-test", port=5567)
    loop = asyncio.new_event_loop()
    try:
        url = loop.run_until_complete(pg0.ensure_running())
    finally:
        loop.close()

    # pg0 data dirs persist across runs, so the DB may already be past the
    # split. Bring everything to head, then downgrade to before the split so
    # memory_units.history is present and the history tables are gone.
    command.upgrade(_alembic_cfg(url), "heads")
    command.downgrade(_alembic_cfg(url), _PRE_SPLIT_REVISION)
    return url


def _col_type(conn, table: str) -> str:
    return conn.execute(
        text("SELECT data_type FROM information_schema.columns WHERE table_name = :t AND column_name = 'bank_id'"),
        {"t": table},
    ).scalar()


def test_split_history_backfills_long_bank_id(pre_split_db_url):
    db_url = pre_split_db_url
    # bank_id matching the shape reported in the issue:
    # <scope>::ou_<32hex>::ou_<32hex> — well over the old 64-char cap. A unique
    # suffix keeps the test idempotent against pg0 data dirs that persist across
    # runs (otherwise a re-run collides on the banks PK).
    long_bank = f"tenantA::ou_{uuid.uuid4().hex}::ou_{uuid.uuid4().hex}"
    assert len(long_bank) > 64

    obs_id = uuid.uuid4()
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("INSERT INTO banks (bank_id) VALUES (:b)"), {"b": long_bank})
        conn.execute(
            text(
                """
                INSERT INTO memory_units (id, bank_id, text, fact_type, history)
                VALUES (:id, :b, 'obs text', 'observation',
                        '[{"changed_at":"2026-01-01T00:00:00Z","previous_text":"a"}]'::jsonb)
                """
            ),
            {"id": obs_id, "b": long_bank},
        )
        conn.commit()

    # The migration that used to abort with StringDataRightTruncation.
    command.upgrade(_alembic_cfg(db_url), _SPLIT_REVISION)

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT bank_id FROM observation_history WHERE observation_id = :id"),
            {"id": obs_id},
        ).fetchone()
        assert row is not None, "history entry for the long-bank_id observation was not backfilled"
        assert row[0] == long_bank, "bank_id was truncated during backfill"
        assert _col_type(conn, "observation_history") == "text"
        assert _col_type(conn, "mental_model_history") == "text"

    # Forward-repair migration to head is a clean no-op on already-TEXT columns.
    command.upgrade(_alembic_cfg(db_url), "heads")
    with engine.connect() as conn:
        assert _col_type(conn, "observation_history") == "text"
        assert _col_type(conn, "mental_model_history") == "text"

    engine.dispose()

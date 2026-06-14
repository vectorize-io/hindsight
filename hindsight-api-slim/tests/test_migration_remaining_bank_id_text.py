"""Regression for the issue #2106 follow-up: the live ``bank_id`` columns must
not truncate on PostgreSQL.

``c3e5a7b9d1f4`` widened the *history* tables to ``TEXT``, but ``directives``,
``mental_models`` and ``mental_model_versions`` kept their original
``VARCHAR(64)`` ``bank_id`` while ``banks.bank_id`` is ``TEXT``. A bank_id longer
than 64 chars (the 78-char shape reported in #2106) can create the bank but then
500s with ``StringDataRightTruncation`` on the next write to those tables.

This test migrates a dedicated pg0 instance to head and asserts all three
columns are ``TEXT``, then writes a >64-char bank_id through every widened table
-- including ``mental_model_versions``, whose composite foreign key
``(mental_model_id, bank_id) -> mental_models(id, bank_id)`` exercises the
widened columns on both sides. Uses a dedicated pg0 instance (mirrors
test_migration_history_long_bank_id) so the migrated schema is well defined.
"""

import asyncio
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

_SCRIPT_LOCATION = str(Path(__file__).parent.parent / "hindsight_api" / "alembic")

_WIDEN_TABLES = ("directives", "mental_models", "mental_model_versions")


def _alembic_cfg(db_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", _SCRIPT_LOCATION)
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("prepend_sys_path", ".")
    cfg.set_main_option("path_separator", "os")
    return cfg


def _col_type(conn, table: str) -> str:
    return conn.execute(
        text(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = :t AND column_name = 'bank_id'"
        ),
        {"t": table},
    ).scalar()


@pytest.fixture(scope="module")
def head_db_url():
    """pg0 instance migrated to head (includes the widen migration)."""
    from hindsight_api.pg0 import EmbeddedPostgres

    pg0 = EmbeddedPostgres(name="hindsight-remaining-bankid-test", port=5568)
    loop = asyncio.new_event_loop()
    try:
        url = loop.run_until_complete(pg0.ensure_running())
    finally:
        loop.close()

    command.upgrade(_alembic_cfg(url), "heads")
    return url


def test_remaining_bank_id_columns_are_text(head_db_url):
    engine = create_engine(head_db_url)
    try:
        with engine.connect() as conn:
            for table in _WIDEN_TABLES:
                assert _col_type(conn, table) == "text", (
                    f"{table}.bank_id must be TEXT to match banks.bank_id"
                )
    finally:
        engine.dispose()


def test_long_bank_id_round_trips_through_widened_tables(head_db_url):
    # bank_id matching the shape reported in the issue, well over the old 64-char
    # cap. Unique suffixes keep the test idempotent against pg0 data dirs that
    # persist across runs (otherwise a re-run collides on the banks PK).
    long_bank = f"tenantA::ou_{uuid.uuid4().hex}::ou_{uuid.uuid4().hex}"
    assert len(long_bank) > 64
    mm_id = f"mm-{uuid.uuid4().hex}"  # mental_models.id is VARCHAR(64)

    engine = create_engine(head_db_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO banks (bank_id) VALUES (:b)"), {"b": long_bank})
            conn.execute(
                text(
                    "INSERT INTO directives (bank_id, name, content) "
                    "VALUES (:b, :n, :c)"
                ),
                {"b": long_bank, "n": "long-bank directive", "c": "rule body"},
            )
            conn.execute(
                text(
                    "INSERT INTO mental_models (id, bank_id, subtype, name, description) "
                    "VALUES (:mid, :b, 'structural', :n, :d)"
                ),
                {"mid": mm_id, "b": long_bank, "n": "long-bank model", "d": "desc"},
            )
            # Composite FK (mental_model_id, bank_id) -> mental_models(id, bank_id):
            # only inserts cleanly once BOTH sides' bank_id columns are TEXT.
            conn.execute(
                text(
                    "INSERT INTO mental_model_versions (mental_model_id, bank_id, version) "
                    "VALUES (:mid, :b, 1)"
                ),
                {"mid": mm_id, "b": long_bank},
            )
            conn.commit()

            for table in _WIDEN_TABLES:
                got = conn.execute(
                    text(f"SELECT bank_id FROM {table} WHERE bank_id = :b LIMIT 1"),
                    {"b": long_bank},
                ).scalar()
                assert got == long_bank, f"bank_id was truncated on the widened {table} table"
    finally:
        engine.dispose()

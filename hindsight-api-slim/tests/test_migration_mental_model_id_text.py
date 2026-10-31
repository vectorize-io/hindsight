"""Regression for issue #5788: columns referencing ``mental_models.id`` must
hold ids over 64 characters on PostgreSQL.

``mental_model_history.mental_model_id`` and ``knowledge_pages.mental_model_id``
were ``VARCHAR(64)`` while ``mental_models.id`` is ``TEXT``, so refreshing a
mental model with a long id failed on every history-row insert.

The test narrows both columns back to ``VARCHAR(64)`` (the state of a deployment
that ran the old DDL), runs the repair migration, then writes rows with an
87-char id. Uses a database of its own so we control which migrations ran.
"""

import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

_SCRIPT_LOCATION = str(Path(__file__).parent.parent / "hindsight_api" / "alembic")
_PRE_REPAIR_REVISION = "e8c1a4f7b2d9"
_COLUMNS = (("mental_model_history", "mental_model_id"), ("knowledge_pages", "mental_model_id"))


def _alembic_cfg(db_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", _SCRIPT_LOCATION)
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("prepend_sys_path", ".")
    cfg.set_main_option("path_separator", "os")
    return cfg


@pytest.fixture(scope="module")
def head_db_url(fresh_db_url):
    url = fresh_db_url
    command.upgrade(_alembic_cfg(url), "heads")
    return url


def _col_type(conn, table: str, column: str) -> str:
    return conn.execute(
        text("SELECT data_type FROM information_schema.columns WHERE table_name = :t AND column_name = :c"),
        {"t": table, "c": column},
    ).scalar()


def test_repair_widens_mental_model_id_refs(head_db_url):
    engine = create_engine(head_db_url)
    # Recreate the narrow columns an already-migrated deployment still has.
    command.downgrade(_alembic_cfg(head_db_url), _PRE_REPAIR_REVISION)
    with engine.connect() as conn:
        for table, column in _COLUMNS:
            conn.execute(text(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE VARCHAR(64)"))
        conn.commit()

    command.upgrade(_alembic_cfg(head_db_url), "heads")

    bank_id = f"bank-{uuid.uuid4().hex}"
    mm_id = f"project-conventions-slack-inc-2026-10-19-lag-on-filecoin-raw-receipt-{uuid.uuid4().hex}"
    assert len(mm_id) > 64
    with engine.connect() as conn:
        for table, column in _COLUMNS:
            assert _col_type(conn, table, column) == "text"
        conn.execute(text("INSERT INTO banks (bank_id) VALUES (:b)"), {"b": bank_id})
        conn.execute(
            text(
                "INSERT INTO mental_models (id, bank_id, subtype, name, source_query, content) "
                "VALUES (:id, :b, 'pinned', 'n', 'q', 'c')"
            ),
            {"id": mm_id, "b": bank_id},
        )
        conn.execute(
            text("INSERT INTO mental_model_history (mental_model_id, bank_id, content) VALUES (:id, :b, '{}')"),
            {"id": mm_id, "b": bank_id},
        )
        conn.execute(
            text(
                "INSERT INTO knowledge_pages (id, bank_id, kind, name, mental_model_id) "
                "VALUES (:pid, :b, 'page', 'n', :id)"
            ),
            {"pid": uuid.uuid4().hex[:32], "b": bank_id, "id": mm_id},
        )
        conn.commit()
    engine.dispose()

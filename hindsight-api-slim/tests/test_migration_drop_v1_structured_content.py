"""Regression for the schema-v1 ``structured_content`` drop (``d1e2f3a4b5c6``).

``mental_models.structured_content`` used to hold a typed block tree parsed out
of the document's markdown. That parse flattened every construct the block union
could not express, permanently (#3361), so a v1 blob is a strictly worse copy of
the row's own ``content``. The migration therefore deletes v1 blobs rather than
converting them: the next refresh re-imports the structure from ``content``,
losslessly.

What this pins:
- a v1 (or otherwise untagged) blob is cleared;
- a v2 blob is left exactly as it was — the migration must not touch documents
  already on the current schema;
- ``content`` is never modified, on any row. It is what users read, and clearing
  the structure must not disturb it.

Uses a dedicated pg0 instance (mirrors test_migration_drop_access_count) so it
controls exactly which migrations have run and never stamps the shared test
instance.
"""

import asyncio
import json
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

pytestmark = pytest.mark.xdist_group("migration-drop-v1-structured-content-pg0")

_SCRIPT_LOCATION = str(Path(__file__).parent.parent / "hindsight_api" / "alembic")

_REVISION = "d1e2f3a4b5c6"
# Revision immediately before the drop.
_PRE_REVISION = "c4e8a1b7d2f6"

_BANK_ID = "migration-v1-structured"

_V1_BLOB = {
    "version": 1,
    "sections": [
        {
            "id": "ops",
            "heading": "Ops",
            "level": 2,
            # What v1 stored for a table: one welded paragraph.
            "blocks": [{"type": "paragraph", "text": "| a | b | |---|---| | 1 | 2 |"}],
        }
    ],
}
_V2_BLOB = {
    "version": 2,
    "sections": [
        {
            "id": "ops",
            "heading": "Ops",
            "level": 2,
            "blocks": [{"id": "b1234abcd", "text": "| a | b |\n| --- | --- |\n| 1 | 2 |"}],
        }
    ],
}
_UNTAGGED_BLOB = {"sections": []}

_CONTENT = "## Ops\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"


def _alembic_cfg(db_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", _SCRIPT_LOCATION)
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("prepend_sys_path", ".")
    cfg.set_main_option("path_separator", "os")
    return cfg


@pytest.fixture(scope="module")
def migrated_db_url():
    """pg0 instance seeded at the pre-drop revision, then migrated through it."""
    from hindsight_api.pg0 import EmbeddedPostgres

    # port=None lets pg0 auto-assign a free port; a hardcoded port is not xdist-safe.
    pg0 = EmbeddedPostgres(name="hindsight-drop-v1-structured-test", port=None)
    loop = asyncio.new_event_loop()
    try:
        url = loop.run_until_complete(pg0.ensure_running())
    finally:
        loop.close()

    cfg = _alembic_cfg(url)
    # Bring the schema up first, then seed, then step the *stamp* back over this
    # revision and re-apply it. A pg0 instance survives between runs, and alembic
    # will not replay a migration on a DB already stamped past it — seeding into
    # a stamped DB would leave the rows untouched and the test asserting nothing.
    # This migration's downgrade is a no-op, so stepping back costs nothing.
    command.upgrade(cfg, _REVISION)

    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO banks (bank_id) VALUES (:b) ON CONFLICT DO NOTHING"),
                {"b": _BANK_ID},
            )
            # Idempotent seed: the instance persists across runs.
            conn.execute(text("DELETE FROM mental_models WHERE bank_id = :b"), {"b": _BANK_ID})
            for mm_id, blob in (
                ("mm-v1", _V1_BLOB),
                ("mm-v2", _V2_BLOB),
                ("mm-untagged", _UNTAGGED_BLOB),
                ("mm-null", None),
            ):
                conn.execute(
                    text(
                        "INSERT INTO mental_models "
                        "(id, bank_id, subtype, name, description, source_query, content, structured_content) "
                        "VALUES (:id, :bank, 'pinned', :name, '', 'what are the ops?', :content, "
                        "CAST(:sc AS jsonb))"
                    ),
                    {
                        "id": mm_id,
                        "bank": _BANK_ID,
                        "name": mm_id,
                        "content": _CONTENT,
                        "sc": json.dumps(blob) if blob is not None else None,
                    },
                )
    finally:
        engine.dispose()

    command.downgrade(cfg, _PRE_REVISION)
    command.upgrade(cfg, _REVISION)
    return url


def _row(conn, mm_id: str):
    return conn.execute(
        text("SELECT content, structured_content FROM mental_models WHERE id = :id AND bank_id = :bank"),
        {"id": mm_id, "bank": _BANK_ID},
    ).one()


def test_v1_structure_is_cleared(migrated_db_url):
    engine = create_engine(migrated_db_url)
    try:
        with engine.connect() as conn:
            assert _row(conn, "mm-v1").structured_content is None
    finally:
        engine.dispose()


def test_untagged_structure_is_cleared(migrated_db_url):
    """A blob with no ``version`` key is not v2 either."""
    engine = create_engine(migrated_db_url)
    try:
        with engine.connect() as conn:
            assert _row(conn, "mm-untagged").structured_content is None
    finally:
        engine.dispose()


def test_v2_structure_is_untouched(migrated_db_url):
    engine = create_engine(migrated_db_url)
    try:
        with engine.connect() as conn:
            assert _row(conn, "mm-v2").structured_content == _V2_BLOB
    finally:
        engine.dispose()


def test_content_is_never_modified(migrated_db_url):
    """The markdown is the document; clearing a structure must not disturb it."""
    engine = create_engine(migrated_db_url)
    try:
        with engine.connect() as conn:
            for mm_id in ("mm-v1", "mm-v2", "mm-untagged", "mm-null"):
                assert _row(conn, mm_id).content == _CONTENT, mm_id
    finally:
        engine.dispose()

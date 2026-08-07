"""Regression for the ``memory_links`` entity-schema drop (``c1e7a9d3f5b2``).

Entity edges stopped being materialized in ``memory_links`` once retain moved
memory-to-entity associations to ``unit_entities`` and the read paths (the /graph
endpoint and recall) began deriving entity edges from that table. Migration
``e9b2c7d1f3a4`` deleted the stored entity rows; this migration removes the
now-dead entity *schema* — the ``entity_id`` column and FK, the entity index,
``link_type = 'entity'`` from the CHECK, and the ``entity_id`` term in the
function-based unique index (which collapses to ``(from_unit_id, to_unit_id,
link_type)``).

Uses a dedicated pg0 instance (mirrors test_migration_drop_access_count) so it
controls exactly which migrations have run and never stamps the shared test
instance.
"""

import asyncio
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

# One module-scoped pg0 instance shared across tests; pin the module to a single
# xdist worker so concurrent workers don't race to provision the same instance
# or re-migrate a DB another worker is reading.
pytestmark = pytest.mark.xdist_group("migration-drop-memory-links-entity-pg0")

_SCRIPT_LOCATION = str(Path(__file__).parent.parent / "hindsight_api" / "alembic")

_DROP_REVISION = "c1e7a9d3f5b2"
# Revision immediately before the drop.
_PRE_DROP_REVISION = "e4a7c1b9d2f6"


def _alembic_cfg(db_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", _SCRIPT_LOCATION)
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("prepend_sys_path", ".")
    cfg.set_main_option("path_separator", "os")
    return cfg


def _columns(conn, table: str) -> set[str]:
    return {
        r[0]
        for r in conn.execute(
            text("SELECT column_name FROM information_schema.columns WHERE table_name = :t"),
            {"t": table},
        )
    }


def _index_exists(conn, name: str) -> bool:
    return bool(conn.execute(text("SELECT 1 FROM pg_indexes WHERE indexname = :n"), {"n": name}).scalar())


def _index_def(conn, name: str) -> str:
    return conn.execute(text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"), {"n": name}).scalar() or ""


def _link_type_check_def(conn) -> str:
    return (
        conn.execute(
            text("SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'memory_links_link_type_check'")
        ).scalar()
        or ""
    )


@pytest.fixture(scope="module")
def head_db_url():
    """pg0 instance migrated to head (includes the entity-schema drop)."""
    from hindsight_api.pg0 import EmbeddedPostgres

    # port=None lets pg0 auto-assign a free port; a hardcoded port is not xdist-safe.
    pg0 = EmbeddedPostgres(name="hindsight-drop-ml-entity-test", port=None)
    loop = asyncio.new_event_loop()
    try:
        url = loop.run_until_complete(pg0.ensure_running())
    finally:
        loop.close()

    command.upgrade(_alembic_cfg(url), "heads")
    return url


def test_entity_schema_is_gone(head_db_url):
    engine = create_engine(head_db_url)
    try:
        with engine.connect() as conn:
            assert "entity_id" not in _columns(conn, "memory_links"), (
                "memory_links.entity_id still exists at head — entity edges are derived from unit_entities"
            )
            assert not _index_exists(conn, "idx_memory_links_entity"), "the entity index should be gone with the column"
    finally:
        engine.dispose()


def test_unique_index_is_three_columns(head_db_url):
    engine = create_engine(head_db_url)
    try:
        with engine.connect() as conn:
            definition = _index_def(conn, "idx_memory_links_unique").lower()
            assert definition, "idx_memory_links_unique is missing"
            assert "unique" in definition
            for col in ("from_unit_id", "to_unit_id", "link_type"):
                assert col in definition, f"{col} missing from idx_memory_links_unique"
            # The old expression key coalesced a nullable entity_id; both must be gone.
            assert "entity_id" not in definition
            assert "coalesce" not in definition
    finally:
        engine.dispose()


def test_check_rejects_entity_keeps_causal(head_db_url):
    engine = create_engine(head_db_url)
    try:
        with engine.connect() as conn:
            definition = _link_type_check_def(conn).lower()
            assert definition, "memory_links_link_type_check is missing"
            assert "'entity'" not in definition, "CHECK should no longer permit link_type = 'entity'"
            for keep in ("'temporal'", "'semantic'", "'caused_by'", "'causes'", "'enables'", "'prevents'"):
                assert keep in definition, f"CHECK dropped a still-supported link_type {keep}"
    finally:
        engine.dispose()


def test_downgrade_restores_entity_schema(head_db_url):
    """Downgrade restores the former schema shape (column, index, CHECK), and
    re-upgrading removes it again — so the migration is not a one-way door."""
    cfg = _alembic_cfg(head_db_url)
    engine = create_engine(head_db_url)
    try:
        command.downgrade(cfg, _PRE_DROP_REVISION)
        with engine.connect() as conn:
            assert "entity_id" in _columns(conn, "memory_links"), "downgrade did not restore memory_links.entity_id"
            assert _index_exists(conn, "idx_memory_links_entity")
            assert "'entity'" in _link_type_check_def(conn).lower()
            assert "entity_id" in _index_def(conn, "idx_memory_links_unique").lower()

        command.upgrade(cfg, _DROP_REVISION)
        with engine.connect() as conn:
            assert "entity_id" not in _columns(conn, "memory_links")
            assert not _index_exists(conn, "idx_memory_links_entity")
            assert "'entity'" not in _link_type_check_def(conn).lower()
            assert "entity_id" not in _index_def(conn, "idx_memory_links_unique").lower()
    finally:
        # Leave the module fixture's instance at head for any later user.
        command.upgrade(cfg, "heads")
        engine.dispose()

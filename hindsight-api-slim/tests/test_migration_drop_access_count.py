"""Regression for the ``access_count`` drop (``e4a7c1b9d2f6``).

``memory_units.access_count`` was created by the initial schema together with an
``access_count DESC`` index, but nothing ever wrote it and nothing ever read it —
it was 0 on every row of every install, and PostgreSQL maintained a btree over it
for nothing. The migration drops it from the live table *and* from the curation
archive.

Dropping it from both tables is not cosmetic symmetry: curation moves a row with
an ``INSERT INTO invalidated_memory_units … SELECT … FROM memory_units`` whose
column list is read from the catalog at runtime
(``memories/pg/writes.py::_memory_unit_columns``), minus ``embedding`` and
``search_vector``. If the drop hit only one of the two tables the round-trip
would break on a column mismatch, so this test pins the lockstep invariant as
well as the drop itself.

Uses a dedicated pg0 instance (mirrors test_migration_observation_search_vector_backfill)
so it controls exactly which migrations have run and never stamps the shared test
instance.
"""

import asyncio
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

# All three tests share one module-scoped pg0 instance. CI runs with
# `--dist loadgroup`, which, absent an xdist_group, scatters them across workers
# that then each instantiate the module fixture and race to provision the SAME
# named instance. Pinning the module to a single worker serialises that — and
# keeps the downgrade test from re-migrating a DB another worker is reading.
pytestmark = pytest.mark.xdist_group("migration-drop-access-count-pg0")

_SCRIPT_LOCATION = str(Path(__file__).parent.parent / "hindsight_api" / "alembic")

_DROP_REVISION = "e4a7c1b9d2f6"
# Revision immediately before the drop.
_PRE_DROP_REVISION = "a9b8c7d6e5f4"

_TABLES = ("memory_units", "invalidated_memory_units")
# The archive is cold storage and carries neither recall-surface column; both are
# recomputed on revert (d4f6a8c2e1b3, e7c3a9f1b2d5).
_ARCHIVE_OMITTED = {"embedding", "search_vector"}


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
    return bool(
        conn.execute(
            text("SELECT 1 FROM pg_indexes WHERE indexname = :n"),
            {"n": name},
        ).scalar()
    )


@pytest.fixture(scope="module")
def head_db_url():
    """pg0 instance migrated to head (includes the drop migration)."""
    from hindsight_api.pg0 import EmbeddedPostgres

    # port=None lets pg0 auto-assign a free port; a hardcoded port is not xdist-safe.
    pg0 = EmbeddedPostgres(name="hindsight-drop-access-count-test", port=None)
    loop = asyncio.new_event_loop()
    try:
        url = loop.run_until_complete(pg0.ensure_running())
    finally:
        loop.close()

    command.upgrade(_alembic_cfg(url), "heads")
    return url


def test_access_count_is_gone_from_both_tables(head_db_url):
    engine = create_engine(head_db_url)
    try:
        with engine.connect() as conn:
            for table in _TABLES:
                assert "access_count" not in _columns(conn, table), (
                    f"{table}.access_count still exists at head — nothing writes or reads it"
                )
            assert not _index_exists(conn, "idx_memory_units_access_count"), (
                "dropping the column should have dropped its index with it"
            )
    finally:
        engine.dispose()


def test_archive_stays_in_lockstep_with_memory_units(head_db_url):
    """The curation INSERT…SELECT builds its column list from the live table, so
    the archive must still carry every ``memory_units`` column but the two
    recall-surface ones."""
    engine = create_engine(head_db_url)
    try:
        with engine.connect() as conn:
            live = _columns(conn, "memory_units")
            archive = _columns(conn, "invalidated_memory_units")
            missing = (live - _ARCHIVE_OMITTED) - archive
            assert not missing, f"invalidated_memory_units is missing {sorted(missing)} — curation would break"
    finally:
        engine.dispose()


def test_downgrade_restores_the_column_and_index(head_db_url):
    """The downgrade re-adds an (empty) column and its index, and re-upgrading
    drops them again — so the migration is not a one-way door on a live DB."""
    cfg = _alembic_cfg(head_db_url)
    engine = create_engine(head_db_url)
    try:
        command.downgrade(cfg, _PRE_DROP_REVISION)
        with engine.connect() as conn:
            for table in _TABLES:
                assert "access_count" in _columns(conn, table), f"downgrade did not restore {table}.access_count"
            assert _index_exists(conn, "idx_memory_units_access_count")

        command.upgrade(cfg, _DROP_REVISION)
        with engine.connect() as conn:
            for table in _TABLES:
                assert "access_count" not in _columns(conn, table)
            assert not _index_exists(conn, "idx_memory_units_access_count")
    finally:
        # Leave the module fixture's instance at head for any later user.
        command.upgrade(cfg, "heads")
        engine.dispose()

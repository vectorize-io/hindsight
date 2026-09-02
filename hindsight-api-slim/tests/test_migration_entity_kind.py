"""Regression for the ``entity_kind`` column + partial trigram index (``b3e8d1c6f4a9``).

Label entities resolve by exact match only, so their rows in the shared trigram
index were pure fuzzy-probe overhead (#3208). The migration materialises the
runtime label classification into ``entities.entity_kind``, backfills existing
rows per bank from the bank's ``entity_labels`` config, and rebuilds the
trigram index as a partial index excluding label rows.

The backfill is the risky part — it re-runs the resolver's ``is_label_entity``
classification inside the migration — so this test migrates a dedicated pg0 to
the revision *before* the change, seeds a bank with an ``entity_labels`` config
plus pre-existing label/regular entity rows, and verifies the upgrade
classifies exactly the label rows.

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

# Single worker for the module: the tests share one module-scoped pg0 instance
# and the downgrade test re-migrates a DB the others read.
pytestmark = pytest.mark.xdist_group("migration-entity-kind-pg0")

_SCRIPT_LOCATION = str(Path(__file__).parent.parent / "hindsight_api" / "alembic")

_REVISION = "b3e8d1c6f4a9"
# Revision immediately before the entity_kind change.
_PRE_REVISION = "f2a6d8c4b1e9"

_OLD_INDEX = "entities_canonical_name_lower_trgm_idx"
_NEW_INDEX = "entities_canonical_name_lower_trgm_nonlabel_idx"

_BANK_WITH_LABELS = "entity-kind-bank-labels"
_BANK_WITHOUT_LABELS = "entity-kind-bank-plain"

# One enum group and one free-text group, exercising both classification rules
# the backfill must reproduce (exact lookup-set hit vs. key-prefix match).
_ENTITY_LABELS_CONFIG = {
    "entity_labels": [
        {"key": "engagement", "type": "value", "values": [{"value": "active"}]},
        {"key": "topic", "type": "text"},
    ]
}

# canonical_name -> expected entity_kind after the backfill.
_SEEDED_LABEL_BANK_ROWS = {
    "engagement:active": "label",
    "topic:quadratic equations": "label",
    "Alice Chen": "regular",
    # Looks label-ish but matches no configured group: must stay regular.
    "grade:B+": "regular",
}


def _alembic_cfg(db_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", _SCRIPT_LOCATION)
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("prepend_sys_path", ".")
    cfg.set_main_option("path_separator", "os")
    return cfg


def _index_def(conn, name: str) -> str | None:
    return conn.execute(
        text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"),
        {"n": name},
    ).scalar()


@pytest.fixture(scope="module")
def head_db_url():
    """pg0 migrated to just before the change, seeded, then upgraded to head."""
    from hindsight_api.pg0 import EmbeddedPostgres

    # port=None lets pg0 auto-assign a free port; a hardcoded port is not xdist-safe.
    pg0 = EmbeddedPostgres(name="hindsight-entity-kind-test", port=None)
    loop = asyncio.new_event_loop()
    try:
        url = loop.run_until_complete(pg0.ensure_running())
    finally:
        loop.close()

    cfg = _alembic_cfg(url)

    # pg0 instances persist between test runs, so a previous session leaves the
    # DB at head with the seed rows already present (and already backfilled).
    # Reset to an empty schema so the pre-revision seeding below is genuinely
    # pre-migration on every run; the migration chain recreates extensions.
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))

    command.upgrade(cfg, _PRE_REVISION)

    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO banks (bank_id, config) VALUES (:b, CAST(:c AS jsonb)), (:b2, '{}'::jsonb)"),
                {"b": _BANK_WITH_LABELS, "c": json.dumps(_ENTITY_LABELS_CONFIG), "b2": _BANK_WITHOUT_LABELS},
            )
            for name in _SEEDED_LABEL_BANK_ROWS:
                conn.execute(
                    text("INSERT INTO entities (bank_id, canonical_name) VALUES (:b, :n)"),
                    {"b": _BANK_WITH_LABELS, "n": name},
                )
            # Same names in a bank with NO label config: none may be reclassified.
            for name in _SEEDED_LABEL_BANK_ROWS:
                conn.execute(
                    text("INSERT INTO entities (bank_id, canonical_name) VALUES (:b, :n)"),
                    {"b": _BANK_WITHOUT_LABELS, "n": name},
                )
    finally:
        engine.dispose()

    command.upgrade(cfg, "heads")
    return url


def test_backfill_classifies_only_configured_label_rows(head_db_url):
    engine = create_engine(head_db_url)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT canonical_name, entity_kind FROM entities WHERE bank_id = :b"),
                {"b": _BANK_WITH_LABELS},
            ).fetchall()
            assert dict(rows) == _SEEDED_LABEL_BANK_ROWS

            plain = (
                conn.execute(
                    text("SELECT DISTINCT entity_kind FROM entities WHERE bank_id = :b"),
                    {"b": _BANK_WITHOUT_LABELS},
                )
                .scalars()
                .all()
            )
            assert plain == ["regular"], "a bank without entity_labels config must not be reclassified"
    finally:
        engine.dispose()


def test_trigram_index_is_partial_and_replaces_old_one(head_db_url):
    engine = create_engine(head_db_url)
    try:
        with engine.connect() as conn:
            new_def = _index_def(conn, _NEW_INDEX)
            assert new_def is not None, "partial trigram index missing at head"
            # pg_indexes normalises `!=` to `<>`.
            assert "WHERE" in new_def and "entity_kind" in new_def and "'label'" in new_def, new_def
            assert "gin_trgm_ops" in new_def, new_def
            assert _index_def(conn, _OLD_INDEX) is None, "old full trigram index should be dropped"
    finally:
        engine.dispose()


def test_downgrade_restores_full_index_and_drops_column(head_db_url):
    cfg = _alembic_cfg(head_db_url)
    engine = create_engine(head_db_url)
    try:
        command.downgrade(cfg, _PRE_REVISION)
        with engine.connect() as conn:
            cols = {
                r[0]
                for r in conn.execute(
                    text("SELECT column_name FROM information_schema.columns WHERE table_name = 'entities'")
                )
            }
            assert "entity_kind" not in cols
            assert _index_def(conn, _OLD_INDEX) is not None, "downgrade must restore the full trigram index"
            assert _index_def(conn, _NEW_INDEX) is None, "dropping the column must drop the partial index"

        command.upgrade(cfg, _REVISION)
        with engine.connect() as conn:
            assert _index_def(conn, _NEW_INDEX) is not None
            assert _index_def(conn, _OLD_INDEX) is None
    finally:
        # Leave the module fixture's instance at head for any later user.
        command.upgrade(cfg, "heads")
        engine.dispose()

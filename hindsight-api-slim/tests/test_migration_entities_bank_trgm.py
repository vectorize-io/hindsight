"""Regression for the bank-scoped entities trigram index (``7c2e5a9d1f40``).

The fuzzy candidate probe filters on ``bank_id`` plus pg_trgm's ``%``, but the
old index covered ``LOWER(canonical_name)`` alone, so every probe matched across
all banks before narrowing to one (12-15 s per 20 names on an 11.3M-entity
table, independent of bank size). The migration replaces it with a
``btree_gin`` composite that starts with ``bank_id``.

Checks the index shape, that the resolver's own candidate query is served by it,
and that downgrade restores the old index. Uses a dedicated pg0 instance
(mirrors test_migration_entity_kind) so it never stamps the shared one.
"""

import asyncio
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

pytestmark = pytest.mark.xdist_group("migration-entities-bank-trgm-pg0")

_SCRIPT_LOCATION = str(Path(__file__).parent.parent / "hindsight_api" / "alembic")

_REVISION = "7c2e5a9d1f40"
_PRE_REVISION = "d4f8b1c6e903"

_OLD_INDEX = "entities_canonical_name_lower_trgm_nonlabel_idx"
_NEW_INDEX = "entities_bank_lower_name_trgm_nonlabel_idx"

# The resolver's fuzzy probe (entity_resolver._resolve_entities_batch_trigram),
# with the table unqualified. Kept textually aligned: the partial predicate must
# match for the planner to use the index.
_CANDIDATE_QUERY = """
    SELECT c.id FROM unnest(CAST(:names AS text[])) AS q(query_text)
    CROSS JOIN LATERAL (
        SELECT e.id FROM entities e
        WHERE e.bank_id = :bank
          AND e.entity_kind != 'label'
          AND LOWER(e.canonical_name) % LOWER(q.query_text)
        ORDER BY similarity(LOWER(e.canonical_name), LOWER(q.query_text)) DESC, e.id
        LIMIT 20
    ) c
"""


def _alembic_cfg(db_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", _SCRIPT_LOCATION)
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("prepend_sys_path", ".")
    cfg.set_main_option("path_separator", "os")
    return cfg


def _index_def(conn, name: str) -> str | None:
    return conn.execute(text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"), {"n": name}).scalar()


@pytest.fixture(scope="module")
def head_db_url():
    """pg0 migrated to head from an empty schema."""
    from hindsight_api.pg0 import EmbeddedPostgres

    pg0 = EmbeddedPostgres(name="hindsight-entities-bank-trgm-test", port=None)
    loop = asyncio.new_event_loop()
    try:
        url = loop.run_until_complete(pg0.ensure_running())
        # pg0 instances persist between runs; start from an empty schema each time.
        engine = create_engine(url)
        with engine.begin() as conn:
            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
        engine.dispose()

        command.upgrade(_alembic_cfg(url), "heads")
        yield url
    finally:
        # Stop the postmaster even when setup fails, so it cannot squat its port
        # for an unrelated test later.
        loop.run_until_complete(pg0.stop())
        loop.close()


def test_index_is_bank_scoped_partial_and_replaces_old_one(head_db_url):
    engine = create_engine(head_db_url)
    try:
        with engine.connect() as conn:
            new_def = _index_def(conn, _NEW_INDEX)
            assert new_def is not None, "bank-scoped trigram index missing at head"
            assert "gin (bank_id, lower(canonical_name) gin_trgm_ops)" in new_def, new_def
            # pg_indexes normalises `!=` to `<>`.
            assert "WHERE" in new_def and "entity_kind" in new_def and "'label'" in new_def, new_def
            assert _index_def(conn, _OLD_INDEX) is None, "bank-agnostic trigram index should be dropped"
            schema = conn.execute(
                text(
                    "SELECT n.nspname FROM pg_extension e JOIN pg_namespace n ON n.oid = e.extnamespace "
                    "WHERE e.extname = 'btree_gin'"
                )
            ).scalar()
            assert schema == "public"
    finally:
        engine.dispose()


def test_resolver_candidate_query_can_use_the_index(head_db_url):
    """The partial predicate must stay textually aligned with the resolver's query.

    Which index wins is data-dependent: at production scale (thousands of
    banks) the planner picks this one, while in a two-bank test table the plain
    ``bank_id`` B-tree is cheaper. So the competing ``bank_id`` indexes are
    dropped inside a transaction that is rolled back, leaving only the question
    that can silently regress: whether this index can serve the query at all.
    """
    engine = create_engine(head_db_url)
    conn = engine.connect()
    trans = conn.begin()
    try:
        conn.execute(text("INSERT INTO banks (bank_id) VALUES ('trgm-a'), ('trgm-b') ON CONFLICT DO NOTHING"))
        conn.execute(
            text(
                "INSERT INTO entities (bank_id, canonical_name) "
                "SELECT CASE WHEN g % 2 = 0 THEN 'trgm-a' ELSE 'trgm-b' END, 'entity name ' || g "
                "FROM generate_series(1, 2000) g ON CONFLICT DO NOTHING"
            )
        )
        for name in conn.execute(
            text(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'entities' "
                "AND indexdef LIKE '%USING btree (bank_id%'"
            )
        ).scalars():
            conn.execute(text(f'DROP INDEX "{name}"'))
        conn.execute(text("ANALYZE entities"))
        conn.execute(text("SET LOCAL enable_seqscan = off"))
        params = {"names": ["entity name 42"], "bank": "trgm-a"}
        plan = "\n".join(r[0] for r in conn.execute(text(f"EXPLAIN {_CANDIDATE_QUERY}"), params))
        assert _NEW_INDEX in plan, plan
        assert conn.execute(text(_CANDIDATE_QUERY), params).fetchall(), "the probe must still find the entity"
    finally:
        trans.rollback()
        conn.close()
        engine.dispose()


def test_downgrade_restores_bank_agnostic_index(head_db_url):
    cfg = _alembic_cfg(head_db_url)
    engine = create_engine(head_db_url)
    try:
        command.downgrade(cfg, _PRE_REVISION)
        with engine.connect() as conn:
            assert _index_def(conn, _OLD_INDEX) is not None, "downgrade must restore the old trigram index"
            assert _index_def(conn, _NEW_INDEX) is None
        command.upgrade(cfg, _REVISION)
        with engine.connect() as conn:
            assert _index_def(conn, _NEW_INDEX) is not None
            assert _index_def(conn, _OLD_INDEX) is None
    finally:
        command.upgrade(cfg, "heads")
        engine.dispose()


def test_upgrade_keeps_old_index_when_btree_gin_is_unavailable(head_db_url, monkeypatch):
    """Managed Postgres without btree_gin keeps the bank-agnostic index instead of failing the migration."""
    import hindsight_api._pg_extensions as pg_extensions

    real_create_extension = pg_extensions.create_extension

    def _unavailable(conn, name, *, cascade=False):
        if name == "btree_gin":
            raise RuntimeError('extension "btree_gin" is not available')
        return real_create_extension(conn, name, cascade=cascade)

    cfg = _alembic_cfg(head_db_url)
    engine = create_engine(head_db_url)
    try:
        command.downgrade(cfg, _PRE_REVISION)
        # Alembic re-imports the revision module on each command, so the migration
        # binds the patched helper.
        monkeypatch.setattr(pg_extensions, "create_extension", _unavailable)
        command.upgrade(cfg, _REVISION)
        with engine.connect() as conn:
            assert _index_def(conn, _OLD_INDEX) is not None, "the old index must stay when btree_gin is missing"
            assert _index_def(conn, _NEW_INDEX) is None
            assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == _REVISION
    finally:
        monkeypatch.undo()
        engine.dispose()
        # Rebuild the scoped index for any later test in this module.
        command.downgrade(cfg, _PRE_REVISION)
        command.upgrade(cfg, "heads")

"""A brand-new schema's migrations must not wait on transactions elsewhere in the database.

The migrations build and drop indexes ``CONCURRENTLY``, which waits for every transaction open
in the database when it starts. On a schema being created from nothing that wait protects
nothing — its tables are empty — but one long transaction in any other schema stalled each of
those statements, so provisioning a schema took minutes instead of seconds.
"""

import threading
import uuid

import asyncpg
import pytest

from hindsight_api.alembic._new_schema_indexes import plain_index_ddl
from hindsight_api.migrations import run_migrations

SCHEMA = "tenant_abc"


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        (
            'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx ON "tenant_abc".memory_units (bank_id)',
            'CREATE INDEX IF NOT EXISTS idx ON "tenant_abc".memory_units (bank_id)',
        ),
        (
            'CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx ON "tenant_abc".t (a)',
            'CREATE UNIQUE INDEX IF NOT EXISTS idx ON "tenant_abc".t (a)',
        ),
        (
            'DROP INDEX CONCURRENTLY IF EXISTS "tenant_abc".idx',
            'DROP INDEX IF EXISTS "tenant_abc".idx',
        ),
        (
            '\n        create index concurrently if not exists idx\n        ON "tenant_abc".entities USING GIN (x)',
            '\n        create index if not exists idx\n        ON "tenant_abc".entities USING GIN (x)',
        ),
    ],
)
def test_index_ddl_on_the_new_schema_drops_concurrently(statement, expected):
    assert plain_index_ddl(statement, SCHEMA) == expected


@pytest.mark.parametrize(
    "statement",
    [
        # Another schema, or the shared one: may be populated and live, so it stays online.
        'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx ON "tenant_other".memory_units (bank_id)',
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx ON public.memory_units (bank_id)",
        # Unqualified: which table it lands on depends on search_path, so it is left alone.
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx ON memory_units (bank_id)",
        # Not index DDL at all, even though it names the schema and the word.
        """SELECT 'CREATE INDEX CONCURRENTLY' FROM "tenant_abc".t""",
        'REINDEX INDEX CONCURRENTLY "tenant_abc".idx',
    ],
)
def test_everything_else_is_left_alone(statement):
    assert plain_index_ddl(statement, SCHEMA) == statement


@pytest.mark.asyncio
async def test_a_new_schema_migrates_while_another_transaction_stays_open(pg0_db_url):
    """With a snapshot held open elsewhere — the shape of a long UPDATE in another tenant — a new
    schema still migrates to head. Before, its first ``CREATE INDEX CONCURRENTLY`` waited for that
    transaction to end, so the migration never finished while it stayed open."""
    schema = f"newschema_idx_{uuid.uuid4().hex[:8]}"
    holder = await asyncpg.connect(pg0_db_url)
    finished = threading.Event()
    errors: list[BaseException] = []

    def _migrate():
        try:
            run_migrations(pg0_db_url, schema=schema)
        except BaseException as e:  # surfaced below, not swallowed
            errors.append(e)
        finally:
            finished.set()

    try:
        # REPEATABLE READ pins one snapshot for the whole transaction, as a long statement does.
        await holder.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
        await holder.fetchval("SELECT txid_current()")

        worker = threading.Thread(target=_migrate, daemon=True)
        worker.start()
        # Generous: a full migration to head on pg0 takes seconds. Waiting on the open
        # transaction would never end on its own.
        assert finished.wait(timeout=180), "migrating a new schema waited on an unrelated open transaction"
        assert not errors, errors[0]
    finally:
        await holder.execute("ROLLBACK")
        finished.wait(timeout=180)
        await holder.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await holder.close()


@pytest.mark.asyncio
async def test_a_new_schema_gets_the_same_indexes_either_way(pg0_db_url, monkeypatch):
    """Building them plainly must not change WHAT is built: a schema migrated with the rewrite
    ends with exactly the index definitions of one migrated the old way, none left INVALID."""
    import hindsight_api.alembic._new_schema_indexes as new_schema_indexes

    plain = f"newschema_idx_{uuid.uuid4().hex[:8]}"
    online = f"newschema_idx_{uuid.uuid4().hex[:8]}"
    run_migrations(pg0_db_url, schema=plain)
    with monkeypatch.context() as m:
        m.setattr(new_schema_indexes, "is_new_schema", lambda connection, schema: False)
        run_migrations(pg0_db_url, schema=online)

    conn = await asyncpg.connect(pg0_db_url)
    try:

        async def _definitions(schema: str) -> dict[str, str]:
            rows = await conn.fetch("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = $1", schema)
            return {r["indexname"]: r["indexdef"].replace(schema, "<schema>") for r in rows}

        plain_defs, online_defs = await _definitions(plain), await _definitions(online)
        assert plain_defs, "the migrated schema has no indexes at all"
        assert plain_defs == online_defs

        invalid = await conn.fetch(
            """
            SELECT c.relname FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = $1 AND NOT i.indisvalid
            """,
            plain,
        )
        assert not invalid, [r["relname"] for r in invalid]
    finally:
        for schema in (plain, online):
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()

"""Extensions must end up in `public` when migrations run against a tenant schema.

Regression cover for #4118: migrations set `search_path` to the tenant schema
first, so `CREATE EXTENSION pg_trgm` installed pg_trgm there. The runtime uses
the default search_path, could not resolve `%`, and every retain failed forever
with nothing surfaced to the caller.
"""

import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from hindsight_api.db_url import to_libpq_url
from hindsight_api.migrations import run_migrations


@pytest.fixture
def tenant_schema(pg0_db_url):
    """Migrate a throwaway tenant schema, then drop it."""
    schema = f"tenant_ext_{uuid.uuid4().hex[:8]}"
    run_migrations(pg0_db_url, schema=schema)
    yield schema
    with _connect(pg0_db_url) as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        conn.commit()


def _connect(db_url):
    return create_engine(to_libpq_url(db_url), poolclass=NullPool).connect()


def test_schema_mode_migration_leaves_no_extension_in_the_tenant_schema(pg0_db_url, tenant_schema):
    with _connect(pg0_db_url) as conn:
        rows = conn.execute(
            text("SELECT e.extname, n.nspname FROM pg_extension e JOIN pg_namespace n ON n.oid = e.extnamespace")
        ).fetchall()

    misplaced = {name: schema for name, schema in rows if schema == tenant_schema}
    assert misplaced == {}, f"extensions installed inside the tenant schema: {misplaced}"


def test_pg_trgm_operator_resolves_on_a_runtime_connection(pg0_db_url, tenant_schema):
    # The runtime never puts the tenant schema on its search_path (it
    # fully-qualifies tables instead), so `%` has to resolve from public.
    with _connect(pg0_db_url) as conn:
        installed = conn.execute(
            text(
                "SELECT n.nspname FROM pg_extension e "
                "JOIN pg_namespace n ON n.oid = e.extnamespace WHERE e.extname = 'pg_trgm'"
            )
        ).scalar()
        if installed is None:
            pytest.skip("pg_trgm is not available on this PostgreSQL build")

        assert installed == "public"
        conn.execute(text("SELECT set_config('search_path', '\"$user\", public', false)"))
        assert conn.execute(text("SELECT 'abc' % 'abd'")).scalar() is not None

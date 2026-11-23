"""Migrations work when pgvector lives in a schema the migration role cannot move it out of (#6662).

Managed Postgres often installs extensions into a dedicated ``extensions`` schema
owned by an admin role. Relocating them into ``public`` then fails (we don't own
them), and the migration ``search_path`` used to drop that schema, so the first
``VECTOR(...)`` column failed with ``type "vector" does not exist``.
"""

import uuid

import pytest
from sqlalchemy import create_engine, make_url, text

from hindsight_api.migrations import run_migrations


@pytest.mark.parametrize("schema", ["public", "tenant_stranded"])
def test_migrations_resolve_vector_from_an_extension_schema_they_do_not_own(fresh_db_url, schema):
    role = f"hs_app_{uuid.uuid4().hex[:8]}"
    db = make_url(fresh_db_url).database
    admin = create_engine(fresh_db_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(text("DROP EXTENSION IF EXISTS vector CASCADE"))
            conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS extensions"))
            conn.execute(text("CREATE EXTENSION vector SCHEMA extensions"))
            conn.execute(text(f"CREATE ROLE {role} LOGIN PASSWORD 'pw'"))
            conn.execute(text(f'GRANT CREATE ON DATABASE "{db}" TO {role}'))
            conn.execute(text(f"ALTER SCHEMA public OWNER TO {role}"))
            conn.execute(text(f"GRANT USAGE ON SCHEMA extensions TO {role}"))
            conn.execute(text(f'ALTER ROLE {role} SET search_path = "$user", public, extensions'))

        app_url = make_url(fresh_db_url).set(username=role, password="pw").render_as_string(hide_password=False)
        run_migrations(app_url, schema=schema)

        with admin.connect() as conn:
            # The extension stayed where its owner put it, and the columns still got built.
            assert (
                conn.execute(
                    text(
                        "SELECT n.nspname FROM pg_extension e JOIN pg_namespace n ON n.oid = e.extnamespace WHERE extname = 'vector'"
                    )
                ).scalar()
                == "extensions"
            )
            assert (
                conn.execute(
                    text(
                        "SELECT count(*) FROM information_schema.columns "
                        "WHERE table_schema = :s AND table_name = 'memory_units' AND column_name = 'embedding'"
                    ),
                    {"s": schema},
                ).scalar()
                == 1
            )
    finally:
        with admin.connect() as conn:
            conn.execute(text(f"REASSIGN OWNED BY {role} TO CURRENT_USER"))
            conn.execute(text(f"DROP OWNED BY {role}"))
            conn.execute(text(f"DROP ROLE {role}"))
        admin.dispose()

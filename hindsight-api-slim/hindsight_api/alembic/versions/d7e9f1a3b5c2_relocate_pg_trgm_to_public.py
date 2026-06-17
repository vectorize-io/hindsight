"""Relocate pg_trgm into the public schema.

``CREATE EXTENSION IF NOT EXISTS pg_trgm`` (revision ``c1a2b3d4e5f6``) has no
``SCHEMA`` clause, so on a custom-schema install it lands in whatever schema is
first on the ``search_path`` at migration time — typically a tenant schema
rather than ``public``. Its ``%`` operator then only resolves for connections
whose ``search_path`` includes that schema, and retain crashes with
``operator does not exist: text % text`` on external Postgres whose role default
``search_path`` omits it (issue #2270).

This pins pg_trgm to ``public`` so the operator lives in the standard, always-on
location. Idempotent and safe to run per-tenant: pg_trgm is database-global, so
the first schema relocates it and later schemas find it already in ``public``.
The runtime also discovers the extension schema dynamically (see
``build_connection_search_path``), so this migration is belt-and-suspenders
standardisation, not the sole fix.

Revision ID: d7e9f1a3b5c2
Revises: e1f2a3b4c5d6
Create Date: 2026-06-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "d7e9f1a3b5c2"
down_revision: str | Sequence[str] | None = "e1f2a3b4c5d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_upgrade() -> None:
    conn = op.get_bind()
    # pg_trgm may be absent (managed Postgres without the contrib module — the
    # entity resolver falls back to the "full" lookup strategy, see #626) or
    # already in public. Only relocate when it lives somewhere else.
    current_schema = conn.execute(
        sa.text(
            "SELECT n.nspname FROM pg_extension e "
            "JOIN pg_namespace n ON n.oid = e.extnamespace WHERE e.extname = 'pg_trgm'"
        )
    ).scalar()
    if not current_schema or current_schema == "public":
        return
    try:
        conn.execute(sa.text("ALTER EXTENSION pg_trgm SET SCHEMA public"))
    except Exception:
        # Insufficient privileges or a non-relocatable install — leave it where
        # it is; the runtime search_path discovery still makes `%` resolvable.
        conn.execute(sa.text("ROLLBACK"))
        conn.execute(sa.text("BEGIN"))


def _pg_downgrade() -> None:
    # No-op: we don't know which schema pg_trgm originally lived in, and moving
    # it back would risk breaking the very operator resolution this fixed.
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

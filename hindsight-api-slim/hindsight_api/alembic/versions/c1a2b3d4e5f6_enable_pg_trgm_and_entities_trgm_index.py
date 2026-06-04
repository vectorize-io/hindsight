"""Enable pg_trgm extension and add GIN trigram index on entities.canonical_name

Revision ID: c1a2b3d4e5f6
Revises: b4c5d6e7f8a9
Create Date: 2026-03-02

Index is created CONCURRENTLY so the migration does not block writes on entities
during production deployments. CONCURRENTLY requires running outside a transaction
block; see migrations.py for how this is handled safely.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c1a2b3d4e5f6"
down_revision: str | Sequence[str] | None = "b4c5d6e7f8a9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    # pg_trgm ships with most PostgreSQL installations as a contrib module.
    # It enables fast similarity lookups via GIN indexes, used for entity name matching.
    # On managed services (e.g. Azure Flexible Server), the extension may not be
    # available or may require manual enablement.  We gracefully skip the index
    # creation if the extension cannot be loaded — the entity resolver will
    # auto-detect and fall back to the "full" lookup strategy at runtime.  See #626.
    conn = op.get_bind()
    try:
        conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
    except Exception:
        # Extension not available (managed Postgres, insufficient privileges, etc.)
        # Roll back the failed statement and skip index creation.
        conn.execute(sa.text("ROLLBACK"))
        conn.execute(sa.text("BEGIN"))
        return

    schema = _get_schema_prefix()
    # GIN index on canonical_name enables sub-millisecond trigram similarity queries
    # (% operator, similarity()) instead of full-table scans across all bank entities.
    # CREATE INDEX CONCURRENTLY cannot run inside a transaction block.
    with op.get_context().autocommit_block():
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS entities_canonical_name_trgm_idx "
            f"ON {schema}entities USING GIN (canonical_name gin_trgm_ops)"
        )


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}entities_canonical_name_trgm_idx")
    # Note: not dropping pg_trgm extension as other indexes may depend on it


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

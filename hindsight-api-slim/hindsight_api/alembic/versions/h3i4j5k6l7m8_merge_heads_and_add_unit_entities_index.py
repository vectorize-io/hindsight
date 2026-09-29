"""Merge 3 migration heads and add unit_entities composite index

Revision ID: h3i4j5k6l7m8
Revises: a4b5c6d7e8f9, g2h3i4j5k6l7
Create Date: 2026-04-07

Merges three unmerged migration heads into one, and adds a composite index
(entity_id, unit_id) on unit_entities for index-only scans in the LATERAL
entity expansion query.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "h3i4j5k6l7m8"
down_revision: str | Sequence[str] | None = ("a4b5c6d7e8f9", "g2h3i4j5k6l7")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    # Composite index enables index-only scans for entity_id -> unit_id lookups
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_unit_entities_entity_unit ON {schema}unit_entities (entity_id, unit_id)"
    )
    # Drop the now-redundant single-column index
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_unit_entities_entity")


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_unit_entities_entity_unit")
    # Restore the single-column index
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_unit_entities_entity ON {schema}unit_entities (entity_id)")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

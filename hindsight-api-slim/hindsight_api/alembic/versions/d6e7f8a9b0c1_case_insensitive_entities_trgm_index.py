"""Recreate entities trigram index on LOWER(canonical_name) for case-insensitive matching

The previous GIN trigram index on canonical_name was case-sensitive, causing
"Alice" and "alice" to have different trigram sets. This recreates it on
LOWER(canonical_name) so the % operator matches case-insensitively.

Revision ID: d6e7f8a9b0c1
Revises: c5d6e7f8a9b0
Create Date: 2026-03-31
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "d6e7f8a9b0c1"
down_revision: str | Sequence[str] | None = "c5d6e7f8a9b0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    schema = _get_schema_prefix()
    # Drop the old case-sensitive trigram index
    op.execute(f"DROP INDEX IF EXISTS entities_canonical_name_trgm_idx")
    # Create case-insensitive trigram index on LOWER(canonical_name)
    op.execute(
        f"CREATE INDEX IF NOT EXISTS entities_canonical_name_lower_trgm_idx "
        f"ON {schema}entities USING GIN (LOWER(canonical_name) gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS entities_canonical_name_lower_trgm_idx")
    schema = _get_schema_prefix()
    # Restore original case-sensitive index
    op.execute(
        f"CREATE INDEX IF NOT EXISTS entities_canonical_name_trgm_idx "
        f"ON {schema}entities USING GIN (canonical_name gin_trgm_ops)"
    )

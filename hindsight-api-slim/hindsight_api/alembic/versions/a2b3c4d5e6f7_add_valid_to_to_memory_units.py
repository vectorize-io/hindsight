"""Add valid_to validity-window column to memory_units table.

Revision ID: a2b3c4d5e6f7
Revises: z1u2v3w4x5y6
Create Date: 2026-05-02

Adds a nullable ``valid_to TIMESTAMPTZ`` column on ``memory_units`` so that
superseded facts can be soft-retired without losing their timeline. Also
adds a partial index on the bank/fact_type prefix limited to currently
active rows so recall keeps using a small index even as historical data
accumulates.

Recall queries filter out rows where ``valid_to <= now()`` so invalidated
memories no longer surface in semantic / BM25 / graph-spreading search,
while ``GET /memories/{id}`` and ``GET /memories/{id}/history`` still return
them — preserving the audit trail.

See issue #1391 for the full design rationale.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "a2b3c4d5e6f7"
down_revision: str | Sequence[str] | None = "z1u2v3w4x5y6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}memory_units ADD COLUMN IF NOT EXISTS valid_to TIMESTAMPTZ NULL")
    op.execute(
        f"COMMENT ON COLUMN {schema}memory_units.valid_to IS "
        "'NULL = still valid; non-NULL = superseded at this timestamp. "
        "Recall filters out memories with valid_to <= now().'"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_memory_units_active "
        f"ON {schema}memory_units (bank_id, fact_type) WHERE valid_to IS NULL"
    )


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_active")
    op.execute(f"ALTER TABLE {schema}memory_units DROP COLUMN IF EXISTS valid_to")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

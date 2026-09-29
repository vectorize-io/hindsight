"""No-op: observation_sources table is Oracle-only

Originally created the observation_sources junction table for all backends,
but PG uses native array ops on the source_memory_ids column (faster at scale).
Oracle creates this table in the o1a2b3c4d5e6 baseline migration instead.

Kept as a no-op to preserve the Alembic revision chain.

Revision ID: k6l7m8n9o0p1
Revises: i4j5k6l7m8n9
Create Date: 2026-04-24
"""

from collections.abc import Sequence

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "k6l7m8n9o0p1"
down_revision: str | Sequence[str] | None = "i4j5k6l7m8n9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_upgrade() -> None:
    # PG uses source_memory_ids[] array on memory_units — no junction table.
    pass


def _pg_downgrade() -> None:
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

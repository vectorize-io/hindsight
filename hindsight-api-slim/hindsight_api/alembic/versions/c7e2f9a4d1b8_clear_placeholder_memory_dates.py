"""No-op: placeholder dates are read as unknown instead of rewritten

Revision ID: c7e2f9a4d1b8
Revises: f2a9c4e7b1d3
Create Date: 2026-10-20

This revision first cleared year-1 / year-9999 dates already stored in memory_units (#5820).
That UPDATE filtered on four date columns no index leads with, so it scanned the whole table in
every schema at startup to fix a handful of rows. Every reader of those dates (temporal recall,
consolidation's date merging) now treats a placeholder as an unknown bound — see
``hindsight_api.engine.stored_dates`` — so the stored rows are harmless and are left as they are.
The revision stays so databases that already ran it keep a valid history.
"""

from collections.abc import Sequence

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c7e2f9a4d1b8"
down_revision: str | Sequence[str] | None = "f2a9c4e7b1d3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_upgrade() -> None:
    # Intentionally empty on both dialects: see the module docstring.
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect()

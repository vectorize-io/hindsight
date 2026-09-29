"""Merge divergent heads from deferrable FK and cooccurrence backfill

Revision ID: m3rg3h3ad5f6
Revises: 9f8e7d6c5b4a, b5d4e3f2a1c9
Create Date: 2026-05-04
"""

from collections.abc import Sequence

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "m3rg3h3ad5f6"
down_revision: tuple[str, ...] = ("9f8e7d6c5b4a", "b5d4e3f2a1c9")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_upgrade() -> None:
    pass


def _pg_downgrade() -> None:
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

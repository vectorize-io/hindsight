"""Merge divergent heads from the bank-scoped trigram index and reasoning tokens

#4775 and #4745 both branched from d4f8b1c6e903 and landed independently, leaving
two heads. Neither touches the other's objects, so the merge is a no-op on both
dialects (Oracle slot absent on purpose: there is nothing to run).

Revision ID: e5b1c7d3a902
Revises: 7c2e5a9d1f40, a6c4e8f1b203
Create Date: 2026-09-25
"""

from collections.abc import Sequence

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e5b1c7d3a902"
down_revision: tuple[str, ...] = ("7c2e5a9d1f40", "a6c4e8f1b203")
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

"""Merge oracle branch migration head with v0.5.3 merge head

Two independent migration heads existed after merging origin/main into
the database-abstraction branch:

  * ``8c6fa6f7230b`` — merge of v0.5.3 divergent heads (from main)
  * ``d5y6z7a8b9c0`` — backfill mental_models.subtype (from oracle branch)

Both ultimately descend from ``c4x5y6z7a8b9``. This empty merge unifies
them into a single head so Alembic's DAG stays linear.

Revision ID: e6f7g8h9i0j1
Revises: 8c6fa6f7230b, d5y6z7a8b9c0
Create Date: 2026-04-22
"""

from collections.abc import Sequence

revision: str = "e6f7g8h9i0j1"
down_revision: str | Sequence[str] | None = ("8c6fa6f7230b", "d5y6z7a8b9c0")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

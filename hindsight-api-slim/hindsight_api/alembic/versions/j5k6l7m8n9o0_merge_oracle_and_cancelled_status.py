"""Merge oracle branch head with cancelled-status migration

Two independent migration heads existed after merging origin/main into
the database-abstraction branch:

  * ``e6f7g8h9i0j1`` — oracle branch merge (from database-abstraction)
  * ``i4j5k6l7m8n9`` — add cancelled status to async_operations (from main)

Both descend from ``8c6fa6f7230b``. This empty merge unifies them into
a single head so Alembic's DAG stays linear.

Revision ID: j5k6l7m8n9o0
Revises: e6f7g8h9i0j1, i4j5k6l7m8n9
Create Date: 2026-04-24
"""

from collections.abc import Sequence

revision: str = "j5k6l7m8n9o0"
down_revision: str | Sequence[str] | None = ("e6f7g8h9i0j1", "i4j5k6l7m8n9")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

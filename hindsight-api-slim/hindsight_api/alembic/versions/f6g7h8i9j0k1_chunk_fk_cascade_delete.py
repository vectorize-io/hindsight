"""chunk_fk_cascade_delete

Revision ID: f6g7h8i9j0k1
Revises: e5f6g7h8i9j0
Create Date: 2026-03-16 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f6g7h8i9j0k1"
down_revision: str | Sequence[str] | None = "e5f6g7h8i9j0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Change memory_units.chunk_id FK from SET NULL to CASCADE.

    When a document is deleted the CASCADE reaches chunks first; with SET NULL
    the memory_units rows survived with chunk_id = NULL, leaving ghost records.
    Switching to CASCADE ensures they are removed together with their chunk.
    """
    op.drop_constraint("memory_units_chunk_fkey", "memory_units", type_="foreignkey")
    op.create_foreign_key(
        "memory_units_chunk_fkey", "memory_units", "chunks", ["chunk_id"], ["chunk_id"], ondelete="CASCADE"
    )


def downgrade() -> None:
    """Revert to SET NULL behaviour."""
    op.drop_constraint("memory_units_chunk_fkey", "memory_units", type_="foreignkey")
    op.create_foreign_key(
        "memory_units_chunk_fkey", "memory_units", "chunks", ["chunk_id"], ["chunk_id"], ondelete="SET NULL"
    )

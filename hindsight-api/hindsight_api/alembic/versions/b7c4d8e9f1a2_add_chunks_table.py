"""add_chunks_table

Revision ID: b7c4d8e9f1a2
Revises: 5a366d414dce
Create Date: 2025-11-28 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7c4d8e9f1a2"
down_revision: str | Sequence[str] | None = "5a366d414dce"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add chunks table and link memory_units to chunks."""

    # Create chunks table with single text PK (bank_id_document_id_chunk_index)
    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Text(), nullable=False),
        sa.Column("bank_id", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id", "bank_id"],
            ["documents.id", "documents.bank_id"],
            name="chunks_document_fkey",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("chunk_id", name=op.f("pk_chunks")),
    )

    # Add indexes for efficient queries
    op.create_index("idx_chunks_document_id", "chunks", ["document_id"])
    op.create_index("idx_chunks_bank_id", "chunks", ["bank_id"])

    # Add chunk_id column to memory_units (nullable, as existing records won't have chunks)
    op.add_column("memory_units", sa.Column("chunk_id", sa.Text(), nullable=True))

    # Add foreign key constraint to chunks table
    op.create_foreign_key(
        "memory_units_chunk_fkey", "memory_units", "chunks", ["chunk_id"], ["chunk_id"], ondelete="SET NULL"
    )

    # Add index on chunk_id for efficient lookups
    op.create_index("idx_memory_units_chunk_id", "memory_units", ["chunk_id"])


def downgrade() -> None:
    """Remove chunks table and chunk_id from memory_units."""

    # Drop index and foreign key from memory_units
    op.drop_index("idx_memory_units_chunk_id", table_name="memory_units")
    op.drop_constraint("memory_units_chunk_fkey", "memory_units", type_="foreignkey")
    op.drop_column("memory_units", "chunk_id")

    # Drop chunks table indexes and table
    op.drop_index("idx_chunks_bank_id", table_name="chunks")
    op.drop_index("idx_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")

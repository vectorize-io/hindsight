"""add_retain_params_to_documents

Revision ID: c8e5f2a3b4d1
Revises: b7c4d8e9f1a2
Create Date: 2025-12-02 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8e5f2a3b4d1"
down_revision: str | Sequence[str] | None = "b7c4d8e9f1a2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add retain_params JSONB column to documents table."""

    # Add retain_params column to store parameters passed during retain
    op.add_column("documents", sa.Column("retain_params", postgresql.JSONB(), nullable=True))

    # Add index for efficient queries on retain_params
    op.create_index("idx_documents_retain_params", "documents", ["retain_params"], postgresql_using="gin")


def downgrade() -> None:
    """Remove retain_params column from documents table."""

    # Drop index
    op.drop_index("idx_documents_retain_params", table_name="documents")

    # Drop column
    op.drop_column("documents", "retain_params")

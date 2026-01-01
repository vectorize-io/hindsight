"""rename_personality_to_disposition

Revision ID: rename_personality
Revises: d9f6a3b4c5e2
Create Date: 2024-12-04

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import context, op

# revision identifiers, used by Alembic.
revision: str = "rename_personality"
down_revision: str | Sequence[str] | None = "d9f6a3b4c5e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_target_schema() -> str:
    """Get the target schema name (tenant schema or 'public')."""
    schema = context.config.get_main_option("target_schema")
    return schema if schema else "public"


def upgrade() -> None:
    """Rename personality column to disposition in banks table (if it exists)."""
    conn = op.get_bind()
    target_schema = _get_target_schema()

    # Check if 'personality' column exists (old database)
    result = conn.execute(
        sa.text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = 'banks' AND column_name = 'personality'
    """),
        {"schema": target_schema},
    )
    has_personality = result.fetchone() is not None

    # Check if 'disposition' column exists (new database)
    result = conn.execute(
        sa.text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = 'banks' AND column_name = 'disposition'
    """),
        {"schema": target_schema},
    )
    has_disposition = result.fetchone() is not None

    if has_personality and not has_disposition:
        # Old database: rename personality -> disposition
        op.alter_column("banks", "personality", new_column_name="disposition")
    elif not has_personality and not has_disposition:
        # Neither exists (shouldn't happen, but be safe): add disposition column
        op.add_column(
            "banks",
            sa.Column(
                "disposition",
                postgresql.JSONB(astext_type=sa.Text()),
                server_default=sa.text("'{}'::jsonb"),
                nullable=False,
            ),
        )
    # else: disposition already exists, nothing to do


def downgrade() -> None:
    """Revert disposition column back to personality."""
    conn = op.get_bind()
    target_schema = _get_target_schema()
    result = conn.execute(
        sa.text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = 'banks' AND column_name = 'disposition'
    """),
        {"schema": target_schema},
    )
    if result.fetchone():
        op.alter_column("banks", "disposition", new_column_name="personality")

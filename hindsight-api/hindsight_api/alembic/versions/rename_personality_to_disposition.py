"""rename_personality_to_disposition

Revision ID: rename_personality
Revises: d9f6a3b4c5e2
Create Date: 2024-12-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'rename_personality'
down_revision: Union[str, Sequence[str], None] = 'd9f6a3b4c5e2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Rename personality column to disposition in banks table (if it exists)."""
    conn = op.get_bind()

    # Check if 'personality' column exists (old database)
    result = conn.execute(sa.text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'banks' AND column_name = 'personality'
    """))
    has_personality = result.fetchone() is not None

    # Check if 'disposition' column exists (new database)
    result = conn.execute(sa.text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'banks' AND column_name = 'disposition'
    """))
    has_disposition = result.fetchone() is not None

    if has_personality and not has_disposition:
        # Old database: rename personality -> disposition
        op.alter_column('banks', 'personality', new_column_name='disposition')
    elif not has_personality and not has_disposition:
        # Neither exists (shouldn't happen, but be safe): add disposition column
        op.add_column('banks', sa.Column(
            'disposition',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False
        ))
    # else: disposition already exists, nothing to do


def downgrade() -> None:
    """Revert disposition column back to personality."""
    conn = op.get_bind()
    result = conn.execute(sa.text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'banks' AND column_name = 'disposition'
    """))
    if result.fetchone():
        op.alter_column('banks', 'disposition', new_column_name='personality')

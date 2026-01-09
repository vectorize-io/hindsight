"""add_learned_subtype

Revision ID: g2b3c4d5e6f7
Revises: f1a2b3c4d5e6
Create Date: 2026-01-09 00:00:00.000000

Adds 'learned' subtype to mental_models table for learnings from reflection.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "g2b3c4d5e6f7"
down_revision: str | Sequence[str] | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add 'learned' to mental_models subtype constraint."""
    # Drop existing constraint and recreate with 'learned' included
    op.execute("ALTER TABLE mental_models DROP CONSTRAINT IF EXISTS ck_mental_models_subtype")
    op.execute(
        "ALTER TABLE mental_models ADD CONSTRAINT ck_mental_models_subtype "
        "CHECK (subtype IN ('structural', 'emergent', 'learned'))"
    )


def downgrade() -> None:
    """Remove 'learned' from mental_models subtype constraint."""
    # First delete any learned mental models
    op.execute("DELETE FROM mental_models WHERE subtype = 'learned'")

    # Then restore the original constraint
    op.execute("ALTER TABLE mental_models DROP CONSTRAINT IF EXISTS ck_mental_models_subtype")
    op.execute(
        "ALTER TABLE mental_models ADD CONSTRAINT ck_mental_models_subtype "
        "CHECK (subtype IN ('structural', 'emergent'))"
    )

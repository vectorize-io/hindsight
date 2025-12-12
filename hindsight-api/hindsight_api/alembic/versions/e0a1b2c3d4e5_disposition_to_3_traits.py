"""disposition_to_3_traits

Revision ID: e0a1b2c3d4e5
Revises: rename_personality
Create Date: 2024-12-08

Migrate disposition traits from Big Five (openness, conscientiousness, extraversion,
agreeableness, neuroticism, bias_strength with 0-1 float values) to the new 3-trait
system (skepticism, literalism, empathy with 1-5 integer values).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e0a1b2c3d4e5'
down_revision: Union[str, Sequence[str], None] = 'rename_personality'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Convert Big Five disposition to 3-trait disposition."""
    conn = op.get_bind()

    # Update all existing banks to use the new disposition format
    # Convert from old format to new format with reasonable mappings:
    # - skepticism: derived from inverse of agreeableness (skeptical people are less agreeable)
    # - literalism: derived from conscientiousness (detail-oriented people are more literal)
    # - empathy: derived from agreeableness + inverse of neuroticism
    # Default all to 3 (neutral) for simplicity
    conn.execute(sa.text("""
        UPDATE banks
        SET disposition = '{"skepticism": 3, "literalism": 3, "empathy": 3}'::jsonb
        WHERE disposition IS NOT NULL
    """))

    # Update the default for new banks
    conn.execute(sa.text("""
        ALTER TABLE banks
        ALTER COLUMN disposition SET DEFAULT '{"skepticism": 3, "literalism": 3, "empathy": 3}'::jsonb
    """))


def downgrade() -> None:
    """Convert back to Big Five disposition."""
    conn = op.get_bind()

    # Revert to Big Five format with default values
    conn.execute(sa.text("""
        UPDATE banks
        SET disposition = '{"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5, "bias_strength": 0.5}'::jsonb
        WHERE disposition IS NOT NULL
    """))

    # Update the default for new banks
    conn.execute(sa.text("""
        ALTER TABLE banks
        ALTER COLUMN disposition SET DEFAULT '{"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5, "bias_strength": 0.5}'::jsonb
    """))

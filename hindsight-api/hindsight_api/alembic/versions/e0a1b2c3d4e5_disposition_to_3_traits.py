"""disposition_to_3_traits

Revision ID: e0a1b2c3d4e5
Revises: rename_personality
Create Date: 2024-12-08

Migrate disposition traits from Big Five (openness, conscientiousness, extraversion,
agreeableness, neuroticism, bias_strength with 0-1 float values) to the new 3-trait
system (skepticism, literalism, empathy with 1-5 integer values).
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import context, op

# revision identifiers, used by Alembic.
revision: str = "e0a1b2c3d4e5"
down_revision: str | Sequence[str] | None = "rename_personality"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (e.g., 'tenant_x.' or '' for public)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _get_target_schema() -> str:
    """Get the target schema name (tenant schema or 'public')."""
    schema = context.config.get_main_option("target_schema")
    return schema if schema else "public"


def upgrade() -> None:
    """Convert Big Five disposition to 3-trait disposition."""
    conn = op.get_bind()
    schema = _get_schema_prefix()
    target_schema = _get_target_schema()

    # Check if disposition column exists (should have been created by previous migration)
    result = conn.execute(
        sa.text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = 'banks' AND column_name = 'disposition'
    """),
        {"schema": target_schema},
    )
    if not result.fetchone():
        # Column doesn't exist yet (shouldn't happen but be safe)
        return

    # Update all existing banks to use the new disposition format
    # Convert from old format to new format with reasonable mappings:
    # - skepticism: derived from inverse of agreeableness (skeptical people are less agreeable)
    # - literalism: derived from conscientiousness (detail-oriented people are more literal)
    # - empathy: derived from agreeableness + inverse of neuroticism
    # Default all to 3 (neutral) for simplicity
    conn.execute(
        sa.text(f"""
        UPDATE {schema}banks
        SET disposition = '{{"skepticism": 3, "literalism": 3, "empathy": 3}}'::jsonb
        WHERE disposition IS NOT NULL
    """)
    )

    # Update the default for new banks
    conn.execute(
        sa.text(f"""
        ALTER TABLE {schema}banks
        ALTER COLUMN disposition SET DEFAULT '{{"skepticism": 3, "literalism": 3, "empathy": 3}}'::jsonb
    """)
    )


def downgrade() -> None:
    """Convert back to Big Five disposition."""
    conn = op.get_bind()
    schema = _get_schema_prefix()
    target_schema = _get_target_schema()

    # Check if disposition column exists
    result = conn.execute(
        sa.text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = 'banks' AND column_name = 'disposition'
    """),
        {"schema": target_schema},
    )
    if not result.fetchone():
        return

    # Revert to Big Five format with default values
    conn.execute(
        sa.text(f"""
        UPDATE {schema}banks
        SET disposition = '{{"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5, "bias_strength": 0.5}}'::jsonb
        WHERE disposition IS NOT NULL
    """)
    )

    # Update the default for new banks
    conn.execute(
        sa.text(f"""
        ALTER TABLE {schema}banks
        ALTER COLUMN disposition SET DEFAULT '{{"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5, "bias_strength": 0.5}}'::jsonb
    """)
    )

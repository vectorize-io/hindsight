"""add_feedback_signal_tables

Revision ID: f1a2b3c4d5e6
Revises: e0a1b2c3d4e5
Create Date: 2025-12-30

Adds tables for feedback signal tracking:
- fact_usefulness: Aggregate usefulness scores per fact
- usefulness_signals: Individual signal records for audit/analytics
- query_pattern_stats: Pattern tracking for query optimization
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "f1a2b3c4d5e6"
down_revision: str | Sequence[str] | None = "e0a1b2c3d4e5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add feedback signal tables."""

    # Create fact_usefulness table - aggregate scores per fact
    op.create_table(
        "fact_usefulness",
        sa.Column("fact_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("bank_id", sa.Text(), nullable=False),
        sa.Column("usefulness_score", sa.Float(), server_default="0.5", nullable=False),
        sa.Column("signal_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_signal_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "last_decay_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("fact_id", name=op.f("pk_fact_usefulness")),
    )
    op.create_index("idx_fact_usefulness_bank_id", "fact_usefulness", ["bank_id"])
    op.create_index(
        "idx_fact_usefulness_score",
        "fact_usefulness",
        ["bank_id", sa.text("usefulness_score DESC")],
    )

    # Create usefulness_signals table - individual signal records
    op.create_table(
        "usefulness_signals",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("fact_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("bank_id", sa.Text(), nullable=False),
        sa.Column("signal_type", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), server_default="1.0", nullable=False),
        sa.Column("query_hash", sa.Text(), nullable=True),
        sa.Column("context", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_usefulness_signals")),
        sa.CheckConstraint(
            "signal_type IN ('used', 'ignored', 'helpful', 'not_helpful')",
            name="usefulness_signals_type_check",
        ),
        sa.CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="usefulness_signals_confidence_check",
        ),
    )
    op.create_index("idx_usefulness_signals_fact_id", "usefulness_signals", ["fact_id"])
    op.create_index("idx_usefulness_signals_bank_id", "usefulness_signals", ["bank_id"])
    op.create_index(
        "idx_usefulness_signals_created_at",
        "usefulness_signals",
        [sa.text("created_at DESC")],
    )

    # Create query_pattern_stats table - pattern tracking
    op.create_table(
        "query_pattern_stats",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("bank_id", sa.Text(), nullable=False),
        sa.Column("query_hash", sa.Text(), nullable=False),
        sa.Column("query_example", sa.Text(), nullable=True),
        sa.Column("total_signals", sa.Integer(), server_default="0", nullable=False),
        sa.Column("helpful_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("not_helpful_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("used_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("ignored_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_query_pattern_stats")),
        sa.UniqueConstraint("bank_id", "query_hash", name="uq_query_pattern_stats_bank_hash"),
    )
    op.create_index("idx_query_pattern_stats_bank_id", "query_pattern_stats", ["bank_id"])


def downgrade() -> None:
    """Remove feedback signal tables."""
    op.drop_index("idx_query_pattern_stats_bank_id", table_name="query_pattern_stats")
    op.drop_table("query_pattern_stats")

    op.drop_index("idx_usefulness_signals_created_at", table_name="usefulness_signals")
    op.drop_index("idx_usefulness_signals_bank_id", table_name="usefulness_signals")
    op.drop_index("idx_usefulness_signals_fact_id", table_name="usefulness_signals")
    op.drop_table("usefulness_signals")

    op.drop_index("idx_fact_usefulness_score", table_name="fact_usefulness")
    op.drop_index("idx_fact_usefulness_bank_id", table_name="fact_usefulness")
    op.drop_table("fact_usefulness")

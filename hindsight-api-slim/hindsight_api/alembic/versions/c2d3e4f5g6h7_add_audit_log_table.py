"""Add audit_log table for feature usage tracking.

Merge migration that combines the two existing heads (a3b4c5d6e7f8 + c8e5f2a3b4d1).

Stores raw request/response as JSONB for expandability without future migrations.
The metadata JSONB column allows adding arbitrary fields in the future.

Revision ID: c2d3e4f5g6h7
Revises: a3b4c5d6e7f8, c8e5f2a3b4d1
Create Date: 2026-03-26
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c2d3e4f5g6h7"
down_revision: str | Sequence[str] | None = ("a3b4c5d6e7f8", "c8e5f2a3b4d1")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}audit_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            action TEXT NOT NULL,
            transport TEXT NOT NULL,
            bank_id TEXT,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            ended_at TIMESTAMPTZ,
            request JSONB,
            response JSONB,
            metadata JSONB DEFAULT '{{}}'::jsonb
        )
        """
    )

    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_audit_log_action_started ON {schema}audit_log (action, started_at DESC)"
    )
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_audit_log_bank_started ON {schema}audit_log (bank_id, started_at DESC)")
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_audit_log_started ON {schema}audit_log (started_at DESC)")


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()

    op.execute(f"DROP INDEX IF EXISTS {schema}idx_audit_log_started")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_audit_log_bank_started")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_audit_log_action_started")
    op.execute(f"DROP TABLE IF EXISTS {schema}audit_log")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

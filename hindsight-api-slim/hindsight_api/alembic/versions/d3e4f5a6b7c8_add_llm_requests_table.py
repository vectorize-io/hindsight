"""Add llm_requests table for per-bank LLM request tracing.

Stores one row per logical LLM call Hindsight makes (success and failure),
capturing the input messages, model output, token usage (input/output/cached/
total), finish reason, and caller metadata. Disabled by default at the
application layer (HINDSIGHT_API_LLM_TRACE_ENABLED); this migration only
creates the table.

PostgreSQL only — the tracing subsystem is not wired for Oracle, so the Oracle
slot is intentionally absent (mirrors the audit_log table).

Revision ID: d3e4f5a6b7c8
Revises: c1d2e3f4a5b6
Create Date: 2026-06-01
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "d3e4f5a6b7c8"
down_revision: str | Sequence[str] | None = "c1d2e3f4a5b6"
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
        CREATE TABLE IF NOT EXISTS {schema}llm_requests (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            bank_id TEXT,
            operation TEXT,
            scope TEXT,
            -- OTel-style grouping: trace_id is shared by every LLM call of one
            -- operation invocation (e.g. all calls of a single reflect run);
            -- parent_span_id is that operation span; span_id is this call.
            trace_id TEXT,
            span_id TEXT,
            parent_span_id TEXT,
            provider TEXT,
            model TEXT,
            status TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            ended_at TIMESTAMPTZ,
            duration_ms INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cached_tokens INTEGER,
            total_tokens INTEGER,
            input JSONB,
            output JSONB,
            error TEXT,
            llm_info JSONB DEFAULT '{{}}'::jsonb,
            metadata JSONB DEFAULT '{{}}'::jsonb
        )
        """
    )

    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_llm_requests_bank_started ON {schema}llm_requests (bank_id, started_at DESC)"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_llm_requests_status_started ON {schema}llm_requests (status, started_at DESC)"
    )
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_llm_requests_started ON {schema}llm_requests (started_at DESC)")
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_llm_requests_trace ON {schema}llm_requests (bank_id, trace_id, started_at)"
    )


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()

    op.execute(f"DROP INDEX IF EXISTS {schema}idx_llm_requests_started")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_llm_requests_status_started")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_llm_requests_bank_started")
    op.execute(f"DROP TABLE IF EXISTS {schema}llm_requests")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

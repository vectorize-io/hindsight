"""Add supersession_queue — pending fact-supersession checks per bank.

One row per freshly retained world fact (with a real ``occurred_start``) that
the ``fact_supersession`` worker still has to check for contradictions against
existing facts. Enqueued inside the retain Phase 2 transaction; drained
bank-wide by the worker (claim = delete; the task-level retry protocol covers
transient failures, so the queue carries no attempts/state columns — same
lifecycle as ``graph_maintenance_queue``).

Revision ID: f3a4b5c6d7e8
Revises: e7f8a9b0c1d2
Create Date: 2026-06-12
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "f3a4b5c6d7e8"
down_revision: str | Sequence[str] | None = "e7f8a9b0c1d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}supersession_queue (
            id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            bank_id     TEXT NOT NULL,
            memory_id   UUID NOT NULL,
            enqueued_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_supersession_queue_bank ON {schema}supersession_queue (bank_id, enqueued_at)"
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP TABLE IF EXISTS {schema}supersession_queue")


def _oracle_upgrade() -> None:
    op.execute(
        """
        CREATE TABLE supersession_queue (
            id          NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            bank_id     VARCHAR2(255) NOT NULL,
            memory_id   RAW(16) NOT NULL,
            enqueued_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL
        )
        """
    )
    op.execute("CREATE INDEX idx_supersession_queue_bank ON supersession_queue (bank_id, enqueued_at)")


def _oracle_downgrade() -> None:
    op.execute("DROP TABLE supersession_queue")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)

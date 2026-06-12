"""Add graphiti_backflow_poller_state — channel C cursor + diagnostics.

Per deep-dive 5 §3.3: the channel-C polling worker needs per-bank
state to track the cursor (``last_seen_invalid_at``) it last advanced
to, plus diagnostic columns for observability. This is a worker
*runtime* state table — distinct from ``bank_configs`` (business
configuration) and from ``graphiti_outbox`` (drainable work queue):

* No FK to ``banks`` — the poller state is best-effort; a deleted
  bank is harmless to leave behind (the bank would already be gone
  from ``list_banks`` so the poller would never look up the row again).
* No cascade on bank deletion — same reason.
* ``last_poll_truncated`` and ``last_poll_error`` are diagnostic
  columns; the worker's correctness depends only on
  ``last_seen_invalid_at``.

Index choice: a single-row PK lookup per bank per poll, so no
secondary index is needed. ``updated_at`` is informational.

Revision ID: h4i5j6k7l8m9
Revises: g3h4i5j6k7l8
Create Date: 2026-06-12
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "h4i5j6k7l8m9"
down_revision: str | Sequence[str] | None = "g3h4i5j6k7l8"
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
        CREATE TABLE IF NOT EXISTS {schema}graphiti_backflow_poller_state (
            bank_id              TEXT PRIMARY KEY,
            last_seen_invalid_at TIMESTAMPTZ NOT NULL DEFAULT 'epoch'::timestamptz,
            last_poll_at         TIMESTAMPTZ,
            last_poll_edges      INT,
            last_poll_truncated  BOOLEAN,
            last_poll_error      TEXT,
            updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP TABLE IF EXISTS {schema}graphiti_backflow_poller_state")


def _oracle_upgrade() -> None:
    # Oracle 23ai: VARCHAR2(255) for the short text columns, NUMBER(1)
    # for the boolean, NUMBER(10) for the edge count, TIMESTAMP WITH TIME
    # ZONE for the timestamps (matches graphiti_outbox precedent).
    # ``'epoch'`` PG default has no direct Oracle equivalent — the
    # worker side reads NULL as "use epoch" via
    # ``_read_poller_state``/``_EPOCH`` fallback, so we omit the
    # column default entirely on Oracle. The first UPSERT will set
    # the value explicitly.
    op.execute(
        """
        CREATE TABLE graphiti_backflow_poller_state (
            bank_id              VARCHAR2(255) PRIMARY KEY,
            last_seen_invalid_at TIMESTAMP WITH TIME ZONE NOT NULL,
            last_poll_at         TIMESTAMP WITH TIME ZONE,
            last_poll_edges      NUMBER(10),
            last_poll_truncated  NUMBER(1),
            last_poll_error      CLOB,
            updated_at           TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL
        )
        """
    )


def _oracle_downgrade() -> None:
    op.execute("DROP TABLE graphiti_backflow_poller_state")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)

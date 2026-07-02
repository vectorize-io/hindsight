"""Add bank_stats_cache table for distributed get_bank_stats caching

Revision ID: b57a7c9e0d13
Revises: c3f7a1b9d2e4
Create Date: 2026-07-01

get_bank_stats aggregates over memory_links / unit_entities — a multi-second scan
on banks with millions of rows. The result was cached per-process (in-memory), so
every API worker recomputed it once per TTL and the first caller after expiry
stalled. This table backs a shared, cross-process TTL cache: one worker's compute
is written here and served to all the others.

PostgreSQL only. Oracle keeps the in-process cache (the runtime picks the backing
store by dialect), so the Oracle upgrade slot is intentionally absent.
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "b57a7c9e0d13"
down_revision: str | Sequence[str] | None = "c3f7a1b9d2e4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    # One row per bank: payload is the full get_bank_stats result, computed_at
    # drives logical TTL expiry. Rows are overwritten in place (ON CONFLICT), so
    # the table never grows beyond the number of banks and needs no purge job.
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}bank_stats_cache (
            bank_id TEXT PRIMARY KEY,
            payload JSONB NOT NULL,
            computed_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"DROP TABLE IF EXISTS {schema}bank_stats_cache")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)  # oracle slot intentionally absent → no-op


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

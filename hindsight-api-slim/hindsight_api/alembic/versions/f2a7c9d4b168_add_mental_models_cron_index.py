"""Add a partial index for the cron-scheduled mental model discovery sweep.

``mental_models_with_cron()`` (``f4d1c2b3a5e6``, currently installed by
``c8b4e2a71f95``) is a cross-tenant discovery routine: it loops over every schema
holding a ``mental_models`` table and, for each, selects the models carrying a
non-empty ``trigger->>'refresh_cron'``. No index covers that predicate, so each
per-schema probe is a **sequential scan** of that tenant's ``mental_models``
table — paid on every maintenance tick, in every API/worker process, whether or
not the tenant has a single cron-scheduled model.

Cron-scheduled models are rare by construction (the trigger defaults to
``{"refresh_after_consolidation": false}``), so at thousands of tenants the sweep
spends essentially all of its time proving that tenants have nothing to do. A
partial index whose predicate matches the routine's WHERE clause exactly turns a
tenant with no cron-scheduled models into an empty index scan.

``bank_id`` is the indexed column so the routine's projection stays on the
leading column of the index; the predicate is what does the work here.

PostgreSQL only: the maintenance loop and its discovery routines are PG-only
(the Oracle slot is intentionally absent, mirroring ``f4d1c2b3a5e6``), so an
Oracle deployment never runs the scan this index exists to avoid.

Revision ID: f2a7c9d4b168
Revises: c8b4e2a71f95
Create Date: 2026-08-17
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "f2a7c9d4b168"
down_revision: str | Sequence[str] | None = "c8b4e2a71f95"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX = "idx_mental_models_cron"


def _pg_schema_prefix() -> str:
    """Schema-qualifier for PostgreSQL multi-tenant migration runs."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    # Plain (non-CONCURRENT) build: mental_models holds one row per mental model,
    # so this is a sub-second SHARE lock even on large installations — unlike the
    # async_operations indexes in a8c1e4f7b0d3, which needed CONCURRENTLY.
    # The predicate is character-for-character the routine's WHERE clause, which
    # is what lets the planner match the partial index.
    op.execute(
        f"CREATE INDEX IF NOT EXISTS {_INDEX} ON {schema}mental_models (bank_id) "
        "WHERE COALESCE(\"trigger\"->>'refresh_cron', '') <> ''"
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}{_INDEX}")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

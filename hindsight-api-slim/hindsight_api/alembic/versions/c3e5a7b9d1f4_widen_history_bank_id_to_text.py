"""Repair: widen ``*_history.bank_id`` from VARCHAR(64) to TEXT on PostgreSQL.

The original split-history migration (``a7b8c9d0e1f2``) declared
``observation_history.bank_id`` and ``mental_model_history.bank_id`` as
``VARCHAR(64)`` on PostgreSQL. But ``memory_units.bank_id`` — the backfill
source for observations — is ``TEXT`` (unbounded), as are ``banks``,
``documents`` and ``entities``. Any deployment whose ``bank_id`` exceeds 64
characters aborts the backfill ``INSERT`` with::

    psycopg2.errors.StringDataRightTruncation: value too long for type
    character varying(64)

Because the migration runs in ``lifespan`` startup inside a transaction, the
whole migration rolls back and the API never comes up — unrecoverable from the
running container. See https://github.com/vectorize-io/hindsight/issues/2106.

``a7b8c9d0e1f2`` itself has been corrected to create the column as ``TEXT``,
which unblocks deployments that *failed* (the migration rolled back, so it
re-runs the fixed DDL). This forward migration covers deployments that already
*succeeded* with the narrow ``VARCHAR(64)`` column — where editing
``a7b8c9d0e1f2`` has no effect because it will not re-run — by widening the
column in place. ``ALTER COLUMN ... TYPE TEXT`` is a no-op on a column that is
already ``TEXT`` (fresh installs and re-run failures), so every upgrade path
converges on ``TEXT``.

The history tables are per-tenant (they live in each tenant schema, not
``public``), so this runs for every migrated schema via the search-path-aware
prefix — unlike the shared-``public`` routines repaired in ``b2d4f6a8c1e3``.

PostgreSQL only. On Oracle both ``memory_units.bank_id`` and the history
``bank_id`` columns are already ``VARCHAR2(256)`` (consistent, never
truncates), so the Oracle slot is intentionally absent.

Revision ID: c3e5a7b9d1f4
Revises: c9a1b2d3e4f5
Create Date: 2026-06-10
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c3e5a7b9d1f4"
down_revision: str | Sequence[str] | None = "c9a1b2d3e4f5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}observation_history ALTER COLUMN bank_id TYPE TEXT")
    op.execute(f"ALTER TABLE {schema}mental_model_history ALTER COLUMN bank_id TYPE TEXT")


def _pg_downgrade() -> None:
    # No-op: narrowing back to VARCHAR(64) could truncate real data and would
    # re-introduce the bug this migration repairs. The column type is owned by
    # ``a7b8c9d0e1f2``'s lifecycle.
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

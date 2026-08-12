"""Repair: drop the stale global memory_units vector index on per-bank backends.

Revision ID: f2a6d8c4b1e9
Revises: e4a7c1b9d2f6
Create Date: 2026-08-06

Migration d5e6f7a8b9c0 dropped the global ``idx_memory_units_embedding`` for
per-bank backends (every vector search is bank + fact_type scoped and served
by the ``idx_mu_emb_*`` partial indexes; the global index is never chosen by
the planner). However, older versions of the post-migration reconcile
(``ensure_vector_extension``) recreated the index when they found none, so
schemas that were provisioned or reconciled in that window carry it to this
day — paying a second vector graph insertion on every ``memory_units`` write
for an index no query uses.

This repair drops the leftover index. It is intentionally a migration, not
runtime reconcile behavior: ``DROP INDEX`` takes an ACCESS EXCLUSIVE lock on
``memory_units``, which belongs in the versioned, once-per-schema migration
path — not in code that runs at unpredictable times during startup or tenant
provisioning. The reconcile now leaves memory_units vector-index DDL to
migrations entirely on per-bank backends.

ScaNN deployments keep the global index by design (filtered vector search over
a global index; per-bank partial indexes cannot be built safely there), so the
migration is a no-op for them.
"""

import os
from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "f2a6d8c4b1e9"
down_revision: str | Sequence[str] | None = "e4a7c1b9d2f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _configured_vector_extension() -> str:
    ext = os.getenv("HINDSIGHT_API_VECTOR_EXTENSION", "pgvector").lower()
    if ext not in {"pgvector", "pgvectorscale", "vchord", "scann"}:
        raise ValueError(
            f"Invalid HINDSIGHT_API_VECTOR_EXTENSION: {ext}. Must be 'pgvector', 'vchord', 'pgvectorscale', or 'scann'"
        )
    return ext


def _pg_upgrade() -> None:
    # ScaNN uses a global vector index by design — nothing stale to repair.
    if _configured_vector_extension() == "scann":
        return
    schema = _pg_schema_prefix()
    # DROP INDEX needs ACCESS EXCLUSIVE on memory_units. While it waits for
    # in-flight transactions, every new query on the table queues behind it,
    # so on a write-busy schema an unbounded wait can pile up traffic. Fail
    # fast instead: the migration errors, the schema stays below head, and
    # the next migration pass retries — preferable to freezing the table.
    # SET LOCAL scopes the timeout to this migration's transaction.
    op.execute("SET LOCAL lock_timeout = '10s'")
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_embedding")


def _pg_downgrade() -> None:
    # Intentional no-op: recreating a potentially multi-GB vector index that no
    # query uses is not a safe downgrade action. Downgrading past d5e6f7a8b9c0
    # restores the global index for deployments that genuinely need it.
    pass


def upgrade() -> None:
    # PG-only repair: the stale index is a PostgreSQL artifact of the old
    # reconcile; Oracle deployments never had a reconcile that created it.
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

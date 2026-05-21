"""Re-create vchord vector indexes with vector_cosine_ops

Revision ID: b8c9d0e1f2a3
Revises: 86f7a033d372
Create Date: 2026-05-20

vchordrq operator classes are bound 1:1 to operators in PostgreSQL:
vector_l2_ops only matches ``<->``, while every Hindsight ANN query uses
``<=>`` (cosine distance). The previous vchord mapping used vector_l2_ops,
so vchord deployments could never use the index — every ANN query fell
back to a sequential scan with per-row cosine computation.

This migration finds any vchordrq index built with vector_l2_ops in the
target schema and re-creates it with vector_cosine_ops, using
``CREATE INDEX CONCURRENTLY`` so it can run online. It is a no-op when:

* the configured vector extension is not vchord, or
* no matching indexes exist (already on cosine ops).

Only PostgreSQL is affected; the Oracle 23ai dialect uses its own native
vector index and does not depend on this mapping.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from alembic import context, op
from sqlalchemy import text

from hindsight_api._vector_index import configured_vector_extension
from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "b8c9d0e1f2a3"
down_revision: str | Sequence[str] | None = "86f7a033d372"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _rebuild_vchordrq_indexes(old_ops: str, new_ops: str) -> None:
    """Rebuild vchordrq indexes using ``old_ops`` so they use ``new_ops``.

    Each index is rebuilt with CREATE INDEX CONCURRENTLY under a fresh name,
    then the old index is dropped and the new one renamed to take its place.
    Must be called inside an ``autocommit_block()`` because CONCURRENTLY
    cannot run inside a transaction.
    """
    bind = op.get_bind()
    schema = context.config.get_main_option("target_schema") or "public"

    rows = bind.execute(
        text(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE schemaname = :schema "
            "AND indexdef ILIKE '%vchordrq%' "
            "AND indexdef ILIKE :ops_like"
        ),
        {"schema": schema, "ops_like": f"%{old_ops}%"},
    ).fetchall()

    for idx_name, indexdef in rows:
        new_def = indexdef.replace(old_ops, new_ops, 1)
        temp_name = f"{idx_name}__opclass_swap"
        new_def = new_def.replace(idx_name, temp_name, 1)
        new_def = re.sub(
            r"^CREATE\s+INDEX\b",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS",
            new_def,
            count=1,
        )
        op.execute(new_def)
        op.execute(f'DROP INDEX IF EXISTS "{schema}"."{idx_name}"')
        op.execute(f'ALTER INDEX "{schema}"."{temp_name}" RENAME TO "{idx_name}"')


def _pg_upgrade() -> None:
    if configured_vector_extension() != "vchord":
        return
    with op.get_context().autocommit_block():
        _rebuild_vchordrq_indexes("vector_l2_ops", "vector_cosine_ops")


def _pg_downgrade() -> None:
    if configured_vector_extension() != "vchord":
        return
    with op.get_context().autocommit_block():
        _rebuild_vchordrq_indexes("vector_cosine_ops", "vector_l2_ops")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

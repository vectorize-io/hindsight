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


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _rebuild_vchordrq_indexes(old_ops: str, new_ops: str) -> None:
    """Rebuild vchordrq indexes using ``old_ops`` so they use ``new_ops``.

    Each index is rebuilt with CREATE INDEX CONCURRENTLY under a fresh name,
    then the old index is dropped and the new one renamed to take its place.
    Must be called inside an ``autocommit_block()`` because CONCURRENTLY
    cannot run inside a transaction.
    """
    bind = op.get_bind()
    # `or None` collapses both unset and explicit empty-string Alembic options
    # into NULL so the COALESCE below falls back to current_schema() in either
    # case. Without it, an empty-string option would filter on `schemaname = ''`
    # and skip every real schema.
    target_schema = context.config.get_main_option("target_schema") or None
    prefix = _pg_schema_prefix()

    rows = bind.execute(
        text(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE schemaname = COALESCE(:target_schema, current_schema()) "
            "AND indexdef ILIKE '%vchordrq%' "
            "AND indexdef ILIKE :ops_like"
        ),
        {"target_schema": target_schema, "ops_like": f"%{old_ops}%"},
    ).fetchall()

    for idx_name, indexdef in rows:
        # pg_get_indexdef() emits the canonical form `CREATE INDEX <name> ON …`,
        # so <name> is the first textual occurrence — both substitutions below
        # rely on that.
        new_def = indexdef.replace(old_ops, new_ops, 1)
        temp_name = f"{idx_name}__opclass_swap"
        new_def = new_def.replace(idx_name, temp_name, 1)
        new_def = re.sub(
            r"^CREATE\s+INDEX\b",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS",
            new_def,
            count=1,
        )

        # CREATE INDEX CONCURRENTLY can leave the partial index as INVALID if a
        # previous run errored (disk pressure, lock conflict, signal). Without
        # this drop the CONCURRENTLY IF NOT EXISTS below would skip creation,
        # then we'd drop the original and rename the broken index into its
        # place — silently restoring the seq-scan bug this migration fixes.
        op.execute(f'DROP INDEX IF EXISTS {prefix}"{temp_name}"')
        op.execute(new_def)

        # Even on a clean run CONCURRENTLY can finish with indisvalid = false
        # (e.g. constraint violation during the second build scan). Refuse to
        # promote in that case so we never alias an INVALID index over a working
        # one.
        is_valid = bind.execute(
            text(
                "SELECT i.indisvalid "
                "FROM pg_class c "
                "JOIN pg_index i ON c.oid = i.indexrelid "
                "JOIN pg_namespace n ON c.relnamespace = n.oid "
                "WHERE c.relname = :name "
                "  AND n.nspname = COALESCE(:target_schema, current_schema())"
            ),
            {"name": temp_name, "target_schema": target_schema},
        ).scalar()
        if not is_valid:
            raise RuntimeError(
                f"vchordrq index rebuild produced an INVALID index ({temp_name}); "
                "drop it manually and re-run the migration."
            )

        # DROP + RENAME atomically. A crash between the two would leave
        # `temp_name` as a valid orphan and the canonical name missing —
        # next run's `pg_indexes` filter (looking for vector_l2_ops) wouldn't
        # find anything to recover from, so the index would stay gone. PG
        # runs the DO block in its own server-side transaction, so either
        # both succeed or both roll back.
        op.execute(
            f"""
            DO $$
            BEGIN
                DROP INDEX IF EXISTS {prefix}"{idx_name}";
                ALTER INDEX {prefix}"{temp_name}" RENAME TO "{idx_name}";
            END $$;
            """
        )


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

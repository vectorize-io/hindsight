"""Scope the entities trigram index by bank: GIN (bank_id, LOWER(canonical_name)).

The fuzzy candidate probe in ``EntityResolver._resolve_entities_batch_trigram``
filters on ``bank_id = $1 AND LOWER(canonical_name) % LOWER(query)``, but the
partial trigram index from ``b3e8d1c6f4a9`` covers ``LOWER(canonical_name)``
alone. The planner therefore probes it for trigram matches across *every* bank
and only then ANDs the bitmap with ``idx_entities_bank_id``, so each probe costs
in proportion to the whole table, not to the bank being resolved.

Measured on a 11.3M-entity production table (2026-09-25): one probe matched
~60K rows across all banks (~0.55 s) to keep a handful. 20 names took 12 s on
the largest bank (334K entities) and 15 s on a bank holding ~300 — bank size
did not matter. That made entity resolution the retain bottleneck; the async
queue grew faster than the workers drained it.

A ``btree_gin`` composite puts ``bank_id`` inside the GIN index, so the probe
stays in one bank. Same predicate, so the resolver's query needs no change and
the planner picks it on its own. Same production measurement after building
it: 2.6 s on the largest bank, 0.47 s on the ~300-entity one.

Built CONCURRENTLY (autocommit block, invalid-leftover sweep, IF NOT EXISTS —
the ``b3e8d1c6f4a9`` shape) before the old index is dropped, so fuzzy probes
never lose index coverage. Skipped when ``pg_trgm`` is absent (the resolver
uses the "full" strategy, #626) or ``btree_gin`` cannot be installed (the old
index is left in place — slower, still correct).

Oracle has no trigram index, so it is a no-op there.

Revision ID: 7c2e5a9d1f40
Revises: d4f8b1c6e903
Create Date: 2026-09-25
"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

from hindsight_api._pg_extensions import create_extension, extension_schema
from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "7c2e5a9d1f40"
down_revision: str | Sequence[str] | None = "d4f8b1c6e903"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

logger = logging.getLogger(__name__)

_OLD_INDEX = "entities_canonical_name_lower_trgm_nonlabel_idx"
_NEW_INDEX = "entities_bank_lower_name_trgm_nonlabel_idx"


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _drop_invalid_leftover(bind: sa.Connection, index_name: str) -> None:
    """Drop an INVALID index left by a CONCURRENTLY build that errored on a previous run.

    IF NOT EXISTS would otherwise skip it forever, and the step after the build
    drops the other index, leaving fuzzy probes with no usable index at all.
    """
    # `or None` collapses an unset option and an explicit empty string into NULL
    # so the COALESCE below falls back to current_schema() in both cases.
    target_schema = context.config.get_main_option("target_schema") or None
    leftover_invalid = bind.execute(
        sa.text(
            "SELECT NOT i.indisvalid "
            "FROM pg_class c "
            "JOIN pg_index i ON c.oid = i.indexrelid "
            "JOIN pg_namespace n ON c.relnamespace = n.oid "
            "WHERE c.relname = :index_name "
            "  AND n.nspname = COALESCE(:target_schema, current_schema())"
        ),
        {"index_name": index_name, "target_schema": target_schema},
    ).scalar()
    if leftover_invalid:
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_pg_schema_prefix()}{index_name}")


def _pg_upgrade() -> None:
    bind = op.get_bind()
    schema = _pg_schema_prefix()

    if extension_schema(bind, "pg_trgm") is None:
        return

    # btree_gin supplies the GIN opclass for the text bank_id column. Pinned to
    # public like every other extension (#4118). Managed services that don't
    # offer it keep the old index: slower probes, identical results.
    try:
        with bind.begin_nested():
            create_extension(bind, "btree_gin")
    except Exception as exc:
        logger.warning(
            "btree_gin is not available (%s); keeping the bank-agnostic entities trigram index. "
            "Fuzzy entity probes stay correct but scan every bank. Install btree_gin and build %s by hand "
            "to get the bank-scoped index.",
            exc,
            _NEW_INDEX,
        )
        return

    # CREATE/DROP INDEX CONCURRENTLY cannot run inside a transaction block.
    with op.get_context().autocommit_block():
        _drop_invalid_leftover(bind, _NEW_INDEX)
        # The predicate must textually match the resolver's candidate query
        # (`entity_kind != 'label'`) for the planner to choose this index.
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_NEW_INDEX} "
            f"ON {schema}entities USING GIN (bank_id, LOWER(canonical_name) gin_trgm_ops) "
            f"WHERE entity_kind != 'label'"
        )
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}{_OLD_INDEX}")


def _pg_downgrade() -> None:
    bind = op.get_bind()
    schema = _pg_schema_prefix()

    if extension_schema(bind, "pg_trgm") is None:
        return

    # Restore the bank-agnostic index before dropping the scoped one so fuzzy
    # probes keep index coverage throughout. btree_gin stays installed: other
    # objects may depend on it, and an unused extension costs nothing.
    with op.get_context().autocommit_block():
        _drop_invalid_leftover(bind, _OLD_INDEX)
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_OLD_INDEX} "
            f"ON {schema}entities USING GIN (LOWER(canonical_name) gin_trgm_ops) "
            f"WHERE entity_kind != 'label'"
        )
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}{_NEW_INDEX}")


def upgrade() -> None:
    # Oracle fuzzy-matches with a UTL_MATCH scan and has no trigram index.
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

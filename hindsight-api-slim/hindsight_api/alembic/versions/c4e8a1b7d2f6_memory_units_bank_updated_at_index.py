"""Index memory_units(bank_id, updated_at) so the scoped staleness check is cheap.

Mental-model staleness asks one question — *has a memory in this model's scope
been written since the model last read the memories?* — and
``any_memory_updated_since`` answers it with
``WHERE bank_id = $1 AND updated_at > $2 [AND tags ...] LIMIT 1``.

``memory_units`` had no index involving ``updated_at``, so that predicate had
only ``idx_memory_units_bank_id`` to work with, and the *negative* answer — the
common one, since most models are up to date — could not stop early: it read
every row in the bank (or, for a model with no tag scope, seq-scanned the
table). Measured on a 400k-row table with 200k memories in the hot bank: 10.7 ms
for a tagged model that is up to date, 29.8 ms for an untagged one, against
0.05 ms and 0.02 ms with this index.

That cost is why the knowledge-tree and stats surfaces approximated staleness
from one bank-wide watermark instead of asking per page (#3291), and why
reflect's ``search_mental_models`` pays a scan for every model the watermark
cannot prove fresh. The index removes the reason for the approximation.

The composite is ``(bank_id, updated_at DESC)`` rather than ``updated_at``
alone because every reader is bank-scoped; the leading column makes the range
scan local to one bank, and the trailing ``DESC`` also serves
``SELECT MAX(updated_at) ... WHERE bank_id = $1`` (the bank write watermark in
the stats payload) as a jump to the first index entry. The tags GIN index stays
useful and complementary: the planner combines the two, using whichever is more
selective for a given model's scope.

``memory_units`` is the largest table in the schema, so the PostgreSQL build
runs CONCURRENTLY to avoid taking a write lock on it. CONCURRENTLY cannot run
inside a transaction block, so it runs in an ``autocommit_block()``, and
``IF NOT EXISTS`` keeps it idempotent across retries and re-migrated tenant
schemas. A CONCURRENTLY build interrupted partway (lock conflict, disk
pressure, signal) leaves the index behind as *invalid* and ``IF NOT EXISTS``
would then skip it forever, so the upgrade first drops any invalid leftover of
this name.

Revision ID: c4e8a1b7d2f6
Revises: f2a7c9d4b168
Create Date: 2026-08-18
"""

from collections.abc import Sequence

from alembic import context, op
from sqlalchemy import text

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c4e8a1b7d2f6"
down_revision: str | Sequence[str] | None = "f2a7c9d4b168"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PG_INDEX_NAME = "idx_memory_units_bank_updated_at"
# Oracle's baseline names memory_units indexes `idx_mu_*`; keep that convention
# rather than importing the longer PostgreSQL one.
_ORACLE_INDEX_NAME = "idx_mu_bank_updated_at"


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    bind = op.get_bind()
    # `or None` collapses an unset option and an explicit empty string into NULL
    # so the COALESCE below falls back to current_schema() in both cases.
    target_schema = context.config.get_main_option("target_schema") or None
    schema = _pg_schema_prefix()

    with op.get_context().autocommit_block():
        leftover_invalid = bind.execute(
            text(
                "SELECT NOT i.indisvalid "
                "FROM pg_class c "
                "JOIN pg_index i ON c.oid = i.indexrelid "
                "JOIN pg_namespace n ON c.relnamespace = n.oid "
                "WHERE c.relname = :index_name "
                "  AND n.nspname = COALESCE(:target_schema, current_schema())"
            ),
            {"index_name": _PG_INDEX_NAME, "target_schema": target_schema},
        ).scalar()
        if leftover_invalid:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}{_PG_INDEX_NAME}")

        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_PG_INDEX_NAME} "
            f"ON {schema}memory_units(bank_id, updated_at DESC)"
        )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}{_PG_INDEX_NAME}")


def _oracle_upgrade() -> None:
    # Plain CREATE INDEX, as every other Oracle half in this tree does: online
    # builds are an Enterprise-only option, so this takes the ordinary DDL lock,
    # and re-runnability is alembic's version table rather than IF NOT EXISTS.
    op.get_bind().exec_driver_sql(f"CREATE INDEX {_ORACLE_INDEX_NAME} ON memory_units(bank_id, updated_at DESC)")


def _oracle_downgrade() -> None:
    op.get_bind().exec_driver_sql(f"DROP INDEX {_ORACLE_INDEX_NAME}")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)

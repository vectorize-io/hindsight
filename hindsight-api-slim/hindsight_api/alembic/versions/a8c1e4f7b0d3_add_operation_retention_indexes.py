"""Add indexes for terminal cleanup and newest-first operation listing.

Revision ID: a8c1e4f7b0d3
Revises: f2a4b6c8d0e2
Create Date: 2026-07-14
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "a8c1e4f7b0d3"
down_revision: str | Sequence[str] | None = "f2a4b6c8d0e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for PostgreSQL multi-tenant migration runs."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    # These can be large tables in long-running installations. Concurrent DDL
    # keeps operation submission, polling, and status reads available.
    with op.get_context().autocommit_block():
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_async_operations_terminal_cleanup "
            f"ON {schema}async_operations (updated_at, operation_id) "
            "WHERE status IN ('completed', 'failed', 'cancelled')"
        )
        op.execute(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_async_operations_bank_created_desc "
            f"ON {schema}async_operations (bank_id, created_at DESC)"
        )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_async_operations_bank_created_desc")
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {schema}idx_async_operations_terminal_cleanup")


def _oracle_create_index(sql: str) -> None:
    """Create an index idempotently for rerun-safe Oracle migrations."""
    block = (
        "BEGIN "
        "EXECUTE IMMEDIATE :stmt; "
        "EXCEPTION WHEN OTHERS THEN "
        "IF SQLCODE = -955 THEN NULL; ELSE RAISE; END IF; "
        "END;"
    )
    op.get_bind().exec_driver_sql(block, {"stmt": sql})


def _oracle_upgrade() -> None:
    # Oracle migrations run with CURRENT_SCHEMA set to each tenant, so table
    # and index names intentionally remain unqualified here.
    _oracle_create_index(
        "CREATE INDEX idx_async_operations_terminal_cleanup ON async_operations (updated_at, operation_id, status)"
    )
    _oracle_create_index(
        "CREATE INDEX idx_async_operations_bank_created_desc ON async_operations (bank_id, created_at DESC)"
    )


def _oracle_downgrade() -> None:
    op.execute("DROP INDEX idx_async_operations_bank_created_desc")
    op.execute("DROP INDEX idx_async_operations_terminal_cleanup")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)

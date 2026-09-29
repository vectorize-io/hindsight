"""Add CASCADE DELETE FK from async_operations and webhooks to banks.

When a bank is deleted, all its async_operations and webhooks rows are
automatically deleted by the database. This ensures that any in-flight
worker tasks detect the deletion via _check_op_alive() and abort early.

Revision ID: e5f6g7h8i9j0
Revises: d4e5f6g7h8i9
Create Date: 2026-03-11
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e5f6g7h8i9j0"
down_revision: str | Sequence[str] | None = "d4e5f6g7h8i9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()

    # Remove orphaned async_operations rows whose bank no longer exists
    # (can happen because there was no FK before this migration).
    op.execute(
        f"""
        DELETE FROM {schema}async_operations
        WHERE bank_id IS NOT NULL
          AND bank_id NOT IN (SELECT bank_id FROM {schema}banks)
        """
    )

    # Remove orphaned webhooks rows whose bank no longer exists.
    op.execute(
        f"""
        DELETE FROM {schema}webhooks
        WHERE bank_id IS NOT NULL
          AND bank_id NOT IN (SELECT bank_id FROM {schema}banks)
        """
    )

    # Add FK with ON DELETE CASCADE so that deleting a bank automatically
    # cleans up all its pending/processing operations and webhook configs.
    op.execute(
        f"""
        ALTER TABLE {schema}async_operations
            ADD CONSTRAINT fk_async_operations_bank_id
            FOREIGN KEY (bank_id) REFERENCES {schema}banks(bank_id)
            ON DELETE CASCADE
        """
    )

    op.execute(
        f"""
        ALTER TABLE {schema}webhooks
            ADD CONSTRAINT fk_webhooks_bank_id
            FOREIGN KEY (bank_id) REFERENCES {schema}banks(bank_id)
            ON DELETE CASCADE
        """
    )


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    op.execute(f"ALTER TABLE {schema}async_operations DROP CONSTRAINT IF EXISTS fk_async_operations_bank_id")
    op.execute(f"ALTER TABLE {schema}webhooks DROP CONSTRAINT IF EXISTS fk_webhooks_bank_id")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

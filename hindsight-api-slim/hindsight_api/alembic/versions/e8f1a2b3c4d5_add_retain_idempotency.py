"""Add optional async retain idempotency identity.

Revision ID: e8f1a2b3c4d5
Revises: d7b2f8a1c934
Create Date: 2026-07-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e8f1a2b3c4d5"
down_revision: str | Sequence[str] | None = "d7b2f8a1c934"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_upgrade() -> None:
    op.add_column("async_operations", sa.Column("idempotency_key", sa.Text(), nullable=True))
    op.add_column("async_operations", sa.Column("request_fingerprint", sa.String(length=64), nullable=True))
    op.create_unique_constraint(
        "uq_async_operations_idempotency",
        "async_operations",
        ["bank_id", "operation_type", "idempotency_key"],
    )


def _pg_downgrade() -> None:
    op.drop_constraint("uq_async_operations_idempotency", "async_operations", type_="unique")
    op.drop_column("async_operations", "request_fingerprint")
    op.drop_column("async_operations", "idempotency_key")


def _oracle_upgrade() -> None:
    # Oracle defaults VARCHAR2 length semantics to bytes. The API accepts 256
    # Unicode characters, so declare character semantics explicitly.
    op.execute("ALTER TABLE async_operations ADD idempotency_key VARCHAR2(256 CHAR)")
    op.add_column("async_operations", sa.Column("request_fingerprint", sa.String(length=64), nullable=True))
    op.create_unique_constraint(
        "uq_async_operations_idempotency",
        "async_operations",
        ["bank_id", "operation_type", "idempotency_key"],
    )


def _oracle_downgrade() -> None:
    op.drop_constraint("uq_async_operations_idempotency", "async_operations", type_="unique")
    op.drop_column("async_operations", "request_fingerprint")
    op.drop_column("async_operations", "idempotency_key")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)

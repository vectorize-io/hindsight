"""Index the Oracle foreign keys no index covers, so deletes stop deadlocking

Revision ID: d3f8a1c6e9b2
Revises: c7e2f9a4d1b8
Create Date: 2026-11-09

Oracle only treats a foreign key as indexed when an index *leads* with all its
columns. ``chunks``, ``memory_units`` and ``invalidated_memory_units`` reference
``documents(id, bank_id)`` through ``(document_id, bank_id)`` but only carry an
index on ``document_id`` (or none), so every ``DELETE FROM documents`` takes a
table-level share lock on all three children for the statement. Retain deletes a
document's memories (holding a row-exclusive lock on ``memory_units``) and then the
document row, so two retains in *different* banks each hold the lock the other's
share lock waits for: ORA-00060, surfaced as an HTTP 500 (#6434).
``knowledge_pages`` has the same gap on ``parent_id`` and
``(mental_model_id, bank_id)``.

The composite indexes replace the single-column ``document_id`` ones, which they
make redundant (they still serve ``document_id``-only lookups).

Oracle only: PostgreSQL takes no table lock to enforce a foreign key, so an
unindexed one there costs a scan, not a deadlock.
"""

from collections.abc import Sequence

from alembic import op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "d3f8a1c6e9b2"
down_revision: str | Sequence[str] | None = "c7e2f9a4d1b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEXES = (
    "CREATE INDEX idx_chunks_document_bank ON chunks (document_id, bank_id)",
    "CREATE INDEX idx_mu_document_bank ON memory_units (document_id, bank_id)",
    "CREATE INDEX idx_imu_document_bank ON invalidated_memory_units (document_id, bank_id)",
    "CREATE INDEX idx_kp_parent ON knowledge_pages (parent_id)",
    "CREATE INDEX idx_kp_mental_model ON knowledge_pages (mental_model_id, bank_id)",
)
_REPLACED = ("idx_chunks_document_id", "idx_mu_document_id")


def _oracle_exec_ignoring(sql: str, *codes: int) -> None:
    """Run one DDL statement, swallowing the given ORA- codes so a rerun is a no-op."""
    tolerated = " OR ".join(f"SQLCODE = {code}" for code in codes)
    block = (
        f"BEGIN EXECUTE IMMEDIATE :stmt; EXCEPTION WHEN OTHERS THEN IF {tolerated} THEN NULL; ELSE RAISE; END IF; END;"
    )
    op.get_bind().exec_driver_sql(block, {"stmt": sql})


def _oracle_upgrade() -> None:
    # Unqualified names: Oracle migrations run with CURRENT_SCHEMA set to each tenant.
    # -955: name already used; -1408: that column list is already indexed.
    for sql in _INDEXES:
        _oracle_exec_ignoring(sql, -955, -1408)
    # -1418: index does not exist.
    for name in _REPLACED:
        _oracle_exec_ignoring(f"DROP INDEX {name}", -1418)


def _oracle_downgrade() -> None:
    _oracle_exec_ignoring("CREATE INDEX idx_chunks_document_id ON chunks (document_id)", -955, -1408)
    _oracle_exec_ignoring("CREATE INDEX idx_mu_document_id ON memory_units (document_id)", -955, -1408)
    for sql in _INDEXES:
        _oracle_exec_ignoring(f"DROP INDEX {sql.split()[2]}", -1418)


def upgrade() -> None:
    run_for_dialect(oracle=_oracle_upgrade)  # PG slot intentionally absent, see docstring


def downgrade() -> None:
    run_for_dialect(oracle=_oracle_downgrade)

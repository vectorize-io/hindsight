"""Make every foreign key over bank_id DEFERRABLE INITIALLY IMMEDIATE.

Revision ID: b8d2f4a6c1e3
Revises: f3a5b7c9d1e2
Create Date: 2026-09-15

``hindsight-admin rename-bank`` rewrites ``bank_id`` on every bank-scoped table
in one transaction. Several FKs carry ``bank_id`` inside a composite key
(``memory_units``/``chunks``/``document_attachments`` → ``documents(id, bank_id)``,
``knowledge_pages``/``mental_model_history`` → ``mental_models(id, bank_id)``),
so no update order satisfies an immediate check: moving the parent strands the
children and moving the children first points them at a parent that does not
exist yet. Deferring the checks to the end of the rename is the only way through
that does not drop and recreate the constraints (ACCESS EXCLUSIVE on the shared
tables plus a full revalidation scan of every bank in the schema).

``INITIALLY IMMEDIATE`` keeps the check timing every other write sees today;
only a transaction that asks with ``SET CONSTRAINTS ... DEFERRED`` changes it.
``ALTER CONSTRAINT`` touches only the catalog — no table rewrite, no revalidation.

Constraints are found from the catalog, not listed, so every bank_id FK present
at this revision is covered. A FK added later without DEFERRABLE makes
rename-bank refuse with its name, and ``test_admin_rename_bank`` fails in CI.

PG-only: rename-bank is an admin command, and the admin CLI is PostgreSQL-only.
Oracle can only change deferrability by recreating the constraint; do that when
rename reaches the API. Oracle slot intentionally absent → no-op there.
"""

from collections.abc import Sequence

from alembic import context, op
from sqlalchemy import text

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "b8d2f4a6c1e3"
down_revision: str | Sequence[str] | None = "f3a5b7c9d1e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# FKs whose own columns include bank_id, in the migrated schema.
_BANK_ID_FKS = text(
    """
    SELECT c.conname, c.conrelid::regclass::text AS tbl
    FROM pg_constraint c
    JOIN pg_namespace n ON n.oid = c.connamespace
    WHERE c.contype = 'f'
      AND n.nspname = COALESCE(:schema, current_schema())
      AND c.condeferrable = :deferrable
      AND EXISTS (
          SELECT 1 FROM pg_attribute a
          WHERE a.attrelid = c.conrelid AND a.attnum = ANY (c.conkey) AND a.attname = 'bank_id'
      )
    """
)


def _set_deferrable(from_deferrable: bool, clause: str) -> None:
    schema = context.config.get_main_option("target_schema")
    bind = op.get_bind()
    for conname, tbl in bind.execute(_BANK_ID_FKS, {"schema": schema, "deferrable": from_deferrable}).fetchall():
        quoted = '"' + conname.replace('"', '""') + '"'
        op.execute(f"ALTER TABLE {tbl} ALTER CONSTRAINT {quoted} {clause}")


def _pg_upgrade() -> None:
    _set_deferrable(False, "DEFERRABLE INITIALLY IMMEDIATE")


def _pg_downgrade() -> None:
    _set_deferrable(True, "NOT DEFERRABLE")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

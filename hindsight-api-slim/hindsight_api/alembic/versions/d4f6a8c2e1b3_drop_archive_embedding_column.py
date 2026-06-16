"""Drop the embedding column from the curation archive (invalidated_memory_units).

The archive is cold storage, never a recall surface, so it has no business
keeping an embedding. Earlier curation code copied the live row's embedding into
``invalidated_memory_units`` on invalidate; the engine now leaves it out on
invalidate and recomputes it on revert, so the column is dead weight.

Dropping it makes "the archive holds no embedding" a schema-enforced invariant
rather than a convention the move queries have to honour, and removes a latent
failure mode (#2209): after an embedding-model switch the live tables are
re-dimensioned but the archive was not, so a stale old-dimension embedding in
the archive tripped a vector-dimension mismatch on the INSERT … SELECT
round-trip. With no column at all, there is nothing to mismatch.

The creation sites no longer add the column (the PG ``LIKE`` clone in
c9a1b2d3e4f5 drops it; the Oracle baseline omits it), so on a fresh database
this migration is a no-op (DROP ... IF EXISTS / Oracle ORA-00904 swallow). It
does the real work on databases created before the column was removed there.

DROP COLUMN is a metadata-only operation on both PostgreSQL and Oracle 23ai (no
table rewrite), so it is cheap even across many tenant schemas. The downgrade
re-adds an unconstrained vector column (any dimension) — empty, since the
embeddings are intentionally discarded.

Revision ID: d4f6a8c2e1b3
Revises: a1d3f5b7c9e2
Create Date: 2026-06-15
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "d4f6a8c2e1b3"
down_revision: str | Sequence[str] | None = "a1d3f5b7c9e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}invalidated_memory_units DROP COLUMN IF EXISTS embedding")


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    # Unconstrained `vector` (no dimension) so the re-added column accepts any
    # model's embeddings; it comes back empty regardless.
    op.execute(f"ALTER TABLE {schema}invalidated_memory_units ADD COLUMN IF NOT EXISTS embedding vector")


def _oracle_upgrade() -> None:
    # Oracle has no `DROP COLUMN IF EXISTS`; swallow ORA-00904 (column does not
    # exist) so the migration is idempotent and safe on a fresh schema whose
    # baseline already omits the column.
    op.execute(
        """
        BEGIN
            EXECUTE IMMEDIATE 'ALTER TABLE invalidated_memory_units DROP COLUMN embedding';
        EXCEPTION WHEN OTHERS THEN
            IF SQLCODE != -904 THEN RAISE; END IF;
        END;
        """
    )


def _oracle_downgrade() -> None:
    # Swallow ORA-01430 (column already exists) for idempotency.
    op.execute(
        """
        BEGIN
            EXECUTE IMMEDIATE 'ALTER TABLE invalidated_memory_units ADD (embedding VECTOR)';
        EXCEPTION WHEN OTHERS THEN
            IF SQLCODE != -1430 THEN RAISE; END IF;
        END;
        """
    )


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)

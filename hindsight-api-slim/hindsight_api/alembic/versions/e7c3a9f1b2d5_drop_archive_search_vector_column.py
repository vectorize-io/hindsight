"""Drop the search_vector column from the curation archive (invalidated_memory_units).

The archive is cold storage, never a recall surface, and carries no text-search
index. Like ``embedding`` (dropped in d4f6a8c2e1b3), ``search_vector`` is a
recall-surface column whose type follows the configured text-search backend, so
it has no business living on the archive. Earlier curation code copied the live
row's ``search_vector`` into ``invalidated_memory_units`` on invalidate; the
engine now leaves it out on invalidate and recomputes it on revert, so the
column is dead weight.

Dropping it removes a latent failure mode (#2503): under a non-native backend
(pgroonga / pg_textsearch / pg_search / vchord) ``ensure_text_search_extension``
reconciles ``memory_units.search_vector`` to ``text`` / ``bm25vector`` but never
touched the archive, which the ``LIKE memory_units`` clone (c9a1b2d3e4f5) created
as ``tsvector``. The type mismatch then broke the curation INSERT … SELECT
round-trip:

    column "search_vector" is of type tsvector but expression is of type text

With no column at all, there is nothing to mismatch. Unlike ``embedding`` (whose
creation sites already omit it), the ``LIKE`` clone still adds ``search_vector``,
so this migration does real work on both fresh and existing PostgreSQL databases.

DROP COLUMN is a metadata-only operation on both PostgreSQL and Oracle 23ai (no
table rewrite), so it is cheap even across many tenant schemas. The downgrade
re-adds an empty ``tsvector`` column (its original creation type).

Revision ID: e7c3a9f1b2d5
Revises: b57a7c9e0d13
Create Date: 2026-07-02
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e7c3a9f1b2d5"
down_revision: str | Sequence[str] | None = "b57a7c9e0d13"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}invalidated_memory_units DROP COLUMN IF EXISTS search_vector")


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    # Re-add as the original tsvector creation type; comes back empty regardless.
    op.execute(f"ALTER TABLE {schema}invalidated_memory_units ADD COLUMN IF NOT EXISTS search_vector tsvector")


def _oracle_upgrade() -> None:
    # Oracle has no `DROP COLUMN IF EXISTS`; swallow ORA-00904 (column does not
    # exist) so the migration is idempotent and safe on a schema whose baseline
    # may already omit the column.
    op.execute(
        """
        BEGIN
            EXECUTE IMMEDIATE 'ALTER TABLE invalidated_memory_units DROP COLUMN search_vector';
        EXCEPTION WHEN OTHERS THEN
            IF SQLCODE != -904 THEN RAISE; END IF;
        END;
        """
    )


def _oracle_downgrade() -> None:
    # Swallow ORA-01430 (column already exists) for idempotency. Oracle stores
    # search_vector as CLOB (see the Oracle baseline), so re-add it as CLOB.
    op.execute(
        """
        BEGIN
            EXECUTE IMMEDIATE 'ALTER TABLE invalidated_memory_units ADD (search_vector CLOB)';
        EXCEPTION WHEN OTHERS THEN
            IF SQLCODE != -1430 THEN RAISE; END IF;
        END;
        """
    )


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)

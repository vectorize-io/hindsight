"""Drop stored embeddings from the curation archive (invalidated_memory_units).

The archive is cold storage, never a recall surface, so it has no business
keeping embeddings. Earlier curation code copied the live row's embedding into
``invalidated_memory_units`` on invalidate; the engine now NULLs it on invalidate
and recomputes it on revert. This migration clears any embeddings that earlier
versions already stored so they don't linger.

Beyond reclaiming space, this removes a latent failure mode (#2209): after an
embedding-model switch the live tables are re-dimensioned but the archive is not,
so a stale old-dimension embedding in the archive would trip a vector-dimension
mismatch on revert. With the column held at NULL, its declared dimension no
longer matters.

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
    # Table may not exist on very old schemas; guard with to_regclass.
    op.execute(
        f"""
        DO $$ BEGIN
            IF to_regclass('{schema}invalidated_memory_units') IS NOT NULL THEN
                UPDATE {schema}invalidated_memory_units SET embedding = NULL WHERE embedding IS NOT NULL;
            END IF;
        END $$;
        """
    )


def _pg_downgrade() -> None:
    # One-way data cleanup: the embeddings are not recoverable, and the archive
    # is meant to hold none. Nothing to undo.
    pass


def upgrade() -> None:
    # PG-only: the Oracle baseline archive likewise carries no embeddings; there
    # is nothing to clear there.
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

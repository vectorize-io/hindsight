"""Add invalidated_memory_units table for curation (edit/invalidate).

Curation keeps the recall hot-path (``memory_units``) clean by *moving*
invalidated facts into a sibling archive table rather than flagging them in
place. If a row is in ``memory_units`` it is live; if it is in
``invalidated_memory_units`` it has been retired. Recall/consolidation/graph
queries never need a state predicate — the rows simply aren't there.

The archive mirrors ``memory_units`` column-for-column — except ``embedding``,
which it never keeps: the archive is cold storage, never a recall surface, and
revert recomputes the embedding from the unit's text/dates/entities. Keeping no
archive vector also means a later embedding-model switch (which re-dimensions
``memory_units``) can't trip a dimension mismatch on the move (#2209). Plus:
- ``invalidation_reason``  optional free text recorded on invalidate
- ``invalidated_at``       when it was retired
- ``entity_ids``           snapshot of the unit's entity associations, so revert
                           can restore them (``unit_entities`` is cascade-deleted
                           when the live row is removed)

This migration also adds ``edited_at`` to ``memory_units``: set whenever a user
edits a memory's fields (text, context, dates, fact_type, entities) via curation.
NULL means never manually modified; a non-NULL value answers "has the user ever
changed this?" with the time of the last edit (distinct from ``updated_at``,
which background operations also bump). It is added to ``memory_units`` *before*
the archive is cloned below, so the archive inherits the column and the marker
travels with a fact when it is invalidated.

Revision ID: c9a1b2d3e4f5
Revises: b2d4f6a8c1e3
Create Date: 2026-06-03
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c9a1b2d3e4f5"
down_revision: str | Sequence[str] | None = "b2d4f6a8c1e3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    # Add edited_at to the live table FIRST so the archive's LIKE clone below
    # inherits it (keeps the two tables column-for-column identical for round-trip).
    op.execute(f"ALTER TABLE {schema}memory_units ADD COLUMN IF NOT EXISTS edited_at TIMESTAMPTZ")
    # LIKE ... INCLUDING DEFAULTS clones every memory_units column (incl.
    # edited_at) so an invalidated row can move back verbatim. We deliberately
    # omit indexes/constraints — the archive is cold storage, not a recall
    # surface; only the lookups below need indexing.
    op.execute(
        f"CREATE TABLE IF NOT EXISTS {schema}invalidated_memory_units (LIKE {schema}memory_units INCLUDING DEFAULTS)"
    )
    # ...then drop the inherited embedding: the archive never stores one (revert
    # recomputes it), so it isn't created here only to be dropped again later by
    # d4f6a8c2e1b3. That migration still runs as a no-op (DROP ... IF EXISTS) on
    # fresh DBs and does the real drop on DBs created before this column was removed.
    op.execute(f"ALTER TABLE {schema}invalidated_memory_units DROP COLUMN IF EXISTS embedding")
    op.execute(
        f"ALTER TABLE {schema}invalidated_memory_units "
        f"ADD COLUMN IF NOT EXISTS invalidation_reason TEXT, "
        f"ADD COLUMN IF NOT EXISTS invalidated_at TIMESTAMPTZ DEFAULT now(), "
        f"ADD COLUMN IF NOT EXISTS entity_ids UUID[]"
    )
    op.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_invalidated_mu_id ON {schema}invalidated_memory_units (id)")
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_invalidated_mu_bank "
        f"ON {schema}invalidated_memory_units (bank_id, invalidated_at)"
    )
    # Deleting a document (or bank) should clear its archived facts too, mirroring
    # the memory_units → documents cascade.
    op.execute(
        f"""
        DO $$ BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'invalidated_mu_document_fkey') THEN
            ALTER TABLE {schema}invalidated_memory_units
                ADD CONSTRAINT invalidated_mu_document_fkey
                FOREIGN KEY (document_id, bank_id)
                REFERENCES {schema}documents(id, bank_id) ON DELETE CASCADE;
        END IF; END $$;
        """
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    # Drops the archive (and its inherited edited_at) wholesale, then removes
    # edited_at from the live table.
    op.execute(f"DROP TABLE IF EXISTS {schema}invalidated_memory_units")
    op.execute(f"ALTER TABLE {schema}memory_units DROP COLUMN IF EXISTS edited_at")


def upgrade() -> None:
    # PG-only: Oracle gets the table from the baseline snapshot, matching the
    # convention used by sibling column/index migrations in this tree.
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

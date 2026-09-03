"""Add memory_units.attachment_ids — which attachment a *fact* came from.

``document_attachments`` records that a document carries an attachment; that is
enough for lifecycle, but not for provenance. Extraction runs one LLM call per
chunk, and a chunk holding a screenshot also holds the prose around it, so a
chunk's attachments could only ever be shown against *every* fact the chunk
produced — a policy paragraph and a screenshot both attributed to the diagram
nobody read.

So the extractor is asked which attachments each fact came from (the prompt
numbers them, ``from_attachments`` carries the answer) and the resolved edge is
stored here. Facts stated in the surrounding text get an empty array, which is
the point: an attachment shown beside a memory means the model looked at it to
produce that memory.

A **column, not a junction table**. These ids behave exactly like ``tags``: a
short array read with the unit and never queried on its own. A table would add
an FK, an index, a second write and a join to every read, and — because it would
have to live in Postgres — would silently do nothing for a memory store that
owns its own records. Carried on the memory, it travels with whatever store
holds it.

Holds ``short_id`` rather than the full digest because short_id is what a
placeholder carries and therefore what the extractor's numbers resolve to; it is
unique per bank (see the unique index in e2f4a6c8b0d1).

Revision ID: f3b5c7d9e1a2
Revises: e2f4a6c8b0d1
Create Date: 2026-09-03
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "f3b5c7d9e1a2"
down_revision: str | Sequence[str] | None = "e2f4a6c8b0d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    # Defaults to empty rather than NULL so a reader never has to distinguish
    # "no attachments" from "written before this column existed" — both mean the
    # fact came from text.
    op.execute(
        f"ALTER TABLE {schema}memory_units "
        f"ADD COLUMN IF NOT EXISTS attachment_ids TEXT[] NOT NULL DEFAULT '{{}}'::text[]"
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}memory_units DROP COLUMN IF EXISTS attachment_ids")


def _oracle_upgrade() -> None:
    # Oracle has no array type in this tree's dialect surface, so the ids are a
    # JSON array in a CLOB — the same shape `tags` and `observation_scopes`
    # already take on this backend.
    op.execute(
        "ALTER TABLE memory_units ADD (attachment_ids CLOB DEFAULT '[]' NOT NULL "
        "CONSTRAINT mu_attachment_ids_json CHECK (attachment_ids IS JSON))"
    )


def _oracle_downgrade() -> None:
    op.execute("ALTER TABLE memory_units DROP COLUMN attachment_ids")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)

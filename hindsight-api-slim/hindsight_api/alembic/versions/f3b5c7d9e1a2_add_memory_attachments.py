"""Add memory_attachments — which attachment a *fact* actually came from.

``document_attachments`` records that a document carries an attachment; that is
enough for lifecycle, but not for provenance. Extraction runs one LLM call per
chunk, and a chunk holding a screenshot also holds the prose around it, so a
chunk's attachments could only ever be shown against *every* fact the chunk
produced — a policy paragraph and a screenshot both ending up attributed to the
diagram nobody read.

So the extractor is asked which attachments each fact came from (the prompt
numbers them, ``from_attachments`` carries the answer) and the resolved edge is
stored per unit here. Facts stated in the surrounding text get no row at all,
which is the point: an attachment shown beside a memory now means the model
looked at it to produce that memory.

Keyed by ``short_id`` rather than the full digest because short_id is what a
placeholder carries and therefore what the extractor's numbers resolve to; it is
unique per bank (see the unique index in e2f4a6c8b0d1), so it is a real key and
not a lossy convenience.

Revision ID: f3b5c7d9e1a2
Revises: e2f4a6c8b0d1
Create Date: 2026-09-02
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
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}memory_attachments (
            bank_id TEXT NOT NULL,
            unit_id UUID NOT NULL,
            short_id VARCHAR(12) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            CONSTRAINT pk_memory_attachments PRIMARY KEY (unit_id, short_id),
            CONSTRAINT fk_memory_attachments_unit FOREIGN KEY (unit_id)
                REFERENCES {schema}memory_units(id) ON DELETE CASCADE
        )
        """
    )
    # The read path's only question: "for these units, which attachments?" —
    # asked once per recall/list page with the page's unit ids.
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_memory_attachments_bank_unit ON {schema}memory_attachments (bank_id, unit_id)"
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_attachments_bank_unit")
    op.execute(f"DROP TABLE IF EXISTS {schema}memory_attachments")


def _oracle_upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_attachments (
            bank_id VARCHAR2(256) NOT NULL,
            unit_id RAW(16) NOT NULL,
            short_id VARCHAR2(12) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT pk_memory_attachments PRIMARY KEY (unit_id, short_id),
            CONSTRAINT fk_memory_attachments_unit FOREIGN KEY (unit_id)
                REFERENCES memory_units(id) ON DELETE CASCADE
        )
        """
    )
    op.execute("CREATE INDEX idx_memory_attachments_bank_unit ON memory_attachments (bank_id, unit_id)")


def _oracle_downgrade() -> None:
    op.execute("DROP TABLE memory_attachments CASCADE CONSTRAINTS")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)

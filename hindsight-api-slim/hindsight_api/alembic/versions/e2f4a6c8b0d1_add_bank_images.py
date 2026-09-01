"""Add bank_images table (inline images retained with document content).

A retain item's ``content`` may now be an ordered list of text and image blocks.
The blocks are flattened at the API boundary into one canonical body in which
each image is an atomic placeholder, and the bytes are written to file storage
content-addressed by their sha256 — so identical images dedupe across documents
and re-ingests, and ``documents.original_text`` stays plain text (which is what
keeps content_hash idempotency, update_mode=append and chunk-delta re-extraction
working unchanged).

This table holds the metadata a placeholder cannot carry: what the bytes are and
where they live. It is keyed **by bank and hash, not by document**, which is the
whole point:

- The document→image edge already exists, in the text. A fact links to its chunk,
  the chunk's text names image hashes, and this table turns a hash into something
  servable. Nothing has to keep a second copy of that edge in sync across the
  append, delta-re-extraction and reprocess paths.
- A row is written once per distinct image per bank, at the ingress, before the
  document row exists. So retain does not have to thread image metadata through
  its pipeline just to satisfy a foreign key.

The blob itself is deliberately NOT stored here: it lives behind the FileStorage
abstraction, so an operator on S3/GCS/Azure keeps image bytes out of the database
exactly as they already do for uploaded files. ``storage_key`` is what links the
two, and is kept as a column rather than recomputed so a future change to the key
layout cannot orphan existing blobs.

Rows are reclaimed with their bank via the FK. Reclaiming an individual blob whose
last referencing document was deleted is a separate sweep, not a cascade — the
reference lives in text, so it cannot be expressed as one.

Revision ID: e2f4a6c8b0d1
Revises: d1e2f3a4b5c6
Create Date: 2026-09-01
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e2f4a6c8b0d1"
down_revision: str | Sequence[str] | None = "d1e2f3a4b5c6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    # (bank_id, image_hash) is the PK: content-addressing means the same image
    # retained a hundred times across a bank's documents is one row and one blob.
    # An insert is therefore an idempotent upsert, which is what lets the ingress
    # write it before knowing whether the retain will ultimately commit.
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}bank_images (
            bank_id TEXT NOT NULL,
            image_hash VARCHAR(64) NOT NULL,
            media_type TEXT NOT NULL,
            byte_size BIGINT NOT NULL,
            storage_key TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            CONSTRAINT pk_bank_images PRIMARY KEY (bank_id, image_hash),
            CONSTRAINT fk_bank_images_bank FOREIGN KEY (bank_id)
                REFERENCES {schema}banks(bank_id) ON DELETE CASCADE
        )
        """
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP TABLE IF EXISTS {schema}bank_images")


def _oracle_upgrade() -> None:
    # bank_id is VARCHAR2 rather than PG's TEXT because it is a PK column, and
    # Oracle cannot index a CLOB — the same trade the other bank-scoped tables in
    # this tree make.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS bank_images (
            bank_id VARCHAR2(256) NOT NULL,
            image_hash VARCHAR2(64) NOT NULL,
            media_type VARCHAR2(64) NOT NULL,
            byte_size NUMBER NOT NULL,
            storage_key VARCHAR2(1024) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT pk_bank_images PRIMARY KEY (bank_id, image_hash),
            CONSTRAINT fk_bank_images_bank FOREIGN KEY (bank_id)
                REFERENCES banks(bank_id) ON DELETE CASCADE
        )
        """
    )


def _oracle_downgrade() -> None:
    op.execute("DROP TABLE bank_images CASCADE CONSTRAINTS")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)

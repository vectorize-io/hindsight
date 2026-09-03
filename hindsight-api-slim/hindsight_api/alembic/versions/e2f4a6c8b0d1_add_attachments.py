"""Add attachments and document_attachments (inline retain attachments).

A retain item's ``content`` may be an ordered list of text, image and file
blocks. The blocks are flattened at the API boundary into one canonical body in
which each attachment is an atomic placeholder, and the bytes are written to file
storage content-addressed by their sha256 — so an identical attachment dedupes
across documents and re-ingests, and ``documents.original_text`` stays plain text
(which is what keeps content_hash idempotency, update_mode=append and chunk-delta
re-extraction working unchanged).

Two tables, because they answer two different questions:

``attachments`` — *what is this attachment?* Keyed by ``(bank_id,
attachment_hash)``, so one row per distinct attachment per bank however many
documents carry it. Document text references it by ``short_id``, a prefix of the
digest, which is all a placeholder can afford to carry; a UNIQUE index on that
prefix turns the astronomically unlikely collision into a failed insert rather
than a placeholder that silently resolves to somebody else's file.

``document_attachments`` — *which documents still reference it?* Derived from the
placeholders in the canonical text each time a document is written, so it cannot
drift from the text that is the source of truth. It exists for lifecycle: rows
die with their document via the composite FK, and a blob is reclaimed once no row
in the bank references its hash. Deriving the edge rather than storing a second
copy of it is what lets append, delta re-extraction and reprocess stay unaware of
attachments entirely.

The bytes are deliberately NOT stored here: they live behind the FileStorage
abstraction, so an operator on S3/GCS/Azure keeps them out of the database
exactly as they already do for uploaded files.

Revision ID: e2f4a6c8b0d1
Revises: d1e2f3a4b5c6
Create Date: 2026-09-02
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
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}attachments (
            bank_id TEXT NOT NULL,
            attachment_hash VARCHAR(64) NOT NULL,
            short_id VARCHAR(12) NOT NULL,
            media_type TEXT NOT NULL,
            byte_size BIGINT NOT NULL,
            storage_key TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'image',
            filename TEXT,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            CONSTRAINT pk_attachments PRIMARY KEY (bank_id, attachment_hash),
            CONSTRAINT fk_attachments_bank FOREIGN KEY (bank_id)
                REFERENCES {schema}banks(bank_id) ON DELETE CASCADE
        )
        """
    )
    # Document text references an attachment by `short_id`, so that is what
    # resolution looks up — and it must identify exactly one attachment.
    op.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS uq_attachments_short_id ON {schema}attachments (bank_id, short_id)")

    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}document_attachments (
            bank_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            attachment_hash VARCHAR(64) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            CONSTRAINT pk_document_attachments PRIMARY KEY (bank_id, document_id, attachment_hash),
            CONSTRAINT fk_document_attachments_document FOREIGN KEY (document_id, bank_id)
                REFERENCES {schema}documents(id, bank_id) ON DELETE CASCADE
        )
        """
    )
    # "Does any document in this bank still reference this hash?" — the question
    # the reclaim path asks after every document delete.
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_document_attachments_bank_hash "
        f"ON {schema}document_attachments (bank_id, attachment_hash)"
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_document_attachments_bank_hash")
    op.execute(f"DROP TABLE IF EXISTS {schema}document_attachments")
    op.execute(f"DROP INDEX IF EXISTS {schema}uq_attachments_short_id")
    op.execute(f"DROP TABLE IF EXISTS {schema}attachments")


def _oracle_upgrade() -> None:
    # bank_id/document_id are VARCHAR2 rather than PG's TEXT because they are PK
    # columns, and Oracle cannot index a CLOB — the same trade the other
    # bank-scoped tables in this tree make.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS attachments (
            bank_id VARCHAR2(256) NOT NULL,
            attachment_hash VARCHAR2(64) NOT NULL,
            short_id VARCHAR2(12) NOT NULL,
            media_type VARCHAR2(255) NOT NULL,
            byte_size NUMBER NOT NULL,
            storage_key VARCHAR2(1024) NOT NULL,
            kind VARCHAR2(16) DEFAULT 'image' NOT NULL,
            filename VARCHAR2(1024),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT pk_attachments PRIMARY KEY (bank_id, attachment_hash),
            CONSTRAINT fk_attachments_bank FOREIGN KEY (bank_id)
                REFERENCES banks(bank_id) ON DELETE CASCADE
        )
        """
    )
    op.execute("CREATE UNIQUE INDEX uq_attachments_short_id ON attachments (bank_id, short_id)")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS document_attachments (
            bank_id VARCHAR2(256) NOT NULL,
            document_id VARCHAR2(512) NOT NULL,
            attachment_hash VARCHAR2(64) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT pk_document_attachments PRIMARY KEY (bank_id, document_id, attachment_hash),
            CONSTRAINT fk_document_attachments_document FOREIGN KEY (document_id, bank_id)
                REFERENCES documents(id, bank_id) ON DELETE CASCADE
        )
        """
    )
    op.execute("CREATE INDEX idx_document_attachments_bank_hash ON document_attachments (bank_id, attachment_hash)")


def _oracle_downgrade() -> None:
    op.execute("DROP TABLE document_attachments CASCADE CONSTRAINTS")
    op.execute("DROP TABLE attachments CASCADE CONSTRAINTS")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)

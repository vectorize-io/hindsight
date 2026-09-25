"""Give every attachment an owning document, and drop document_attachments.

An attachment used to be identified by ``(bank_id, attachment_hash)``: one row
and one blob per distinct attachment per bank, with ``document_attachments``
recording which documents referenced it. That edge needs a SQL ``documents``
row, so a bank whose documents live in a memories store wrote none — and the
reclaim sweep, which asks "does any row still reference this hash?", could not
tell "unreferenced" from "referenced by a document it cannot see". It bailed out
for those banks, so nothing ever reclaimed their attachments.

So the document becomes the owner. ``attachments`` is keyed
``(bank_id, document_id, attachment_hash)``, the ``filename`` moves onto it
(a filename describes the reference, not the bytes), and
``document_attachments`` goes away with the "is anyone else still using this?"
question that needed it. Two documents carrying the same image now hold two
rows; cross-document dedup is given up on purpose, because it was the only
reason a delete had to ask that question.

**The bytes are not moved.** A new attachment is written under a per-document
key, but a migrated row keeps the bank-scoped ``storage_key`` it already has —
copying every blob in every bank is a large, failure-prone job to buy a
property that costs nothing to carry instead: the delete path drops a blob only
once no surviving ``attachments`` row names that key. Legacy rows sharing one
key therefore free it when the last of them goes, and everything retained after
this migration has a key of its own.

**Attachments with no document are dropped.** A store-owned bank writes
``attachments`` rows but no ``document_attachments`` rows, so its existing
attachments have no knowable owner; the same is true of a row left behind by a
retain that never committed. They are deleted rather than kept as rows nothing
could ever reclaim. Their blobs are left in file storage — this migration does
not touch it — and are swept when the bank is deleted.

Revision ID: a7c2e9f41b60
Revises: f3a5b7c9d1e2
Create Date: 2026-09-17
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "a7c2e9f41b60"
down_revision: str | Sequence[str] | None = "f3a5b7c9d1e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


#: One referencing document per attachment — the one the existing row is handed
#: to. Every *other* referencing document gets a row of its own afterwards.
#: ``ROW_NUMBER`` rather than PG's ``DISTINCT ON`` so the same text runs on both.
_FIRST_OWNER = """
    SELECT bank_id, attachment_hash, document_id, filename FROM (
        SELECT bank_id, attachment_hash, document_id, filename,
               ROW_NUMBER() OVER (
                   PARTITION BY bank_id, attachment_hash ORDER BY document_id
               ) AS rn
        FROM {schema}document_attachments
    ) ranked WHERE rn = 1
"""

#: The remaining referencing documents, each taking a copy of the blob metadata —
#: including the storage key, which they share until the last of them is deleted.
_EXTRA_OWNERS = """
    INSERT INTO {schema}attachments
        (bank_id, document_id, attachment_hash, short_id, media_type, byte_size,
         storage_key, kind, created_at, filename)
    SELECT da.bank_id, da.document_id, a.attachment_hash, a.short_id, a.media_type,
           a.byte_size, a.storage_key, a.kind, a.created_at, da.filename
    FROM {schema}document_attachments da
    JOIN {schema}attachments a
      ON a.bank_id = da.bank_id AND a.attachment_hash = da.attachment_hash
    WHERE da.document_id <> a.document_id
"""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}attachments ADD COLUMN IF NOT EXISTS document_id TEXT")
    op.execute(f"ALTER TABLE {schema}attachments ADD COLUMN IF NOT EXISTS filename TEXT")

    # Both go before the backfill: the extra-owner insert writes a second row for
    # a (bank_id, attachment_hash) the old key would refuse, and two documents
    # may legitimately carry the same short id.
    op.execute(f"ALTER TABLE {schema}attachments DROP CONSTRAINT IF EXISTS pk_attachments")
    op.execute(f"DROP INDEX IF EXISTS {schema}uq_attachments_short_id")

    op.execute(
        f"""
        UPDATE {schema}attachments a
        SET document_id = owner.document_id, filename = owner.filename
        FROM ({_FIRST_OWNER.format(schema=schema)}) owner
        WHERE a.bank_id = owner.bank_id AND a.attachment_hash = owner.attachment_hash
        """
    )
    op.execute(f"DELETE FROM {schema}attachments WHERE document_id IS NULL")
    op.execute(_EXTRA_OWNERS.format(schema=schema))

    op.execute(f"ALTER TABLE {schema}attachments ALTER COLUMN document_id SET NOT NULL")
    op.execute(
        f"ALTER TABLE {schema}attachments ADD CONSTRAINT pk_attachments "
        f"PRIMARY KEY (bank_id, document_id, attachment_hash)"
    )
    # A placeholder is resolved in the context of the document that carries it,
    # so the id must identify exactly one attachment *within* that document.
    op.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS uq_attachments_short_id "
        f"ON {schema}attachments (bank_id, document_id, short_id)"
    )

    op.execute(f"DROP TABLE IF EXISTS {schema}document_attachments")


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}document_attachments (
            bank_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            attachment_hash VARCHAR(64) NOT NULL,
            filename TEXT,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            CONSTRAINT pk_document_attachments PRIMARY KEY (bank_id, document_id, attachment_hash),
            CONSTRAINT fk_document_attachments_document FOREIGN KEY (document_id, bank_id)
                REFERENCES {schema}documents(id, bank_id) ON DELETE CASCADE
        )
        """
    )
    # Only documents that still have a SQL row can take the FK back; a store-owned
    # bank's attachments lose their edge, which is the state this migration found
    # them in.
    op.execute(
        f"""
        INSERT INTO {schema}document_attachments (bank_id, document_id, attachment_hash, filename, created_at)
        SELECT a.bank_id, a.document_id, a.attachment_hash, a.filename, a.created_at
        FROM {schema}attachments a
        JOIN {schema}documents d ON d.id = a.document_id AND d.bank_id = a.bank_id
        ON CONFLICT DO NOTHING
        """
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_document_attachments_bank_hash "
        f"ON {schema}document_attachments (bank_id, attachment_hash)"
    )

    # Collapse back to one row per attachment. The survivor keeps its own
    # storage_key, so a document whose copy is dropped now points at another
    # document's bytes — identical content, since the key is the content hash.
    op.execute(f"ALTER TABLE {schema}attachments DROP CONSTRAINT IF EXISTS pk_attachments")
    op.execute(f"DROP INDEX IF EXISTS {schema}uq_attachments_short_id")
    op.execute(
        f"""
        DELETE FROM {schema}attachments a
        USING ({_FIRST_OWNER.format(schema=schema)}) owner
        WHERE a.bank_id = owner.bank_id AND a.attachment_hash = owner.attachment_hash
          AND a.document_id <> owner.document_id
        """
    )
    op.execute(f"ALTER TABLE {schema}attachments DROP COLUMN IF EXISTS document_id")
    op.execute(f"ALTER TABLE {schema}attachments DROP COLUMN IF EXISTS filename")
    op.execute(f"ALTER TABLE {schema}attachments ADD CONSTRAINT pk_attachments PRIMARY KEY (bank_id, attachment_hash)")
    op.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS uq_attachments_short_id ON {schema}attachments (bank_id, short_id)")


def _oracle_upgrade() -> None:
    op.execute("ALTER TABLE attachments ADD (document_id VARCHAR2(512), filename VARCHAR2(1024))")
    op.execute("ALTER TABLE attachments DROP CONSTRAINT pk_attachments")
    op.execute("DROP INDEX uq_attachments_short_id")

    # MERGE rather than PG's UPDATE ... FROM, which Oracle does not have.
    op.execute(
        f"""
        MERGE INTO attachments a
        USING ({_FIRST_OWNER.format(schema="")}) owner
        ON (a.bank_id = owner.bank_id AND a.attachment_hash = owner.attachment_hash)
        WHEN MATCHED THEN UPDATE SET a.document_id = owner.document_id, a.filename = owner.filename
        """
    )
    op.execute("DELETE FROM attachments WHERE document_id IS NULL")
    op.execute(_EXTRA_OWNERS.format(schema=""))

    op.execute("ALTER TABLE attachments MODIFY (document_id VARCHAR2(512) NOT NULL)")
    op.execute(
        "ALTER TABLE attachments ADD CONSTRAINT pk_attachments PRIMARY KEY (bank_id, document_id, attachment_hash)"
    )
    op.execute("CREATE UNIQUE INDEX uq_attachments_short_id ON attachments (bank_id, document_id, short_id)")

    op.execute("DROP TABLE document_attachments CASCADE CONSTRAINTS")


def _oracle_downgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS document_attachments (
            bank_id VARCHAR2(256) NOT NULL,
            document_id VARCHAR2(512) NOT NULL,
            attachment_hash VARCHAR2(64) NOT NULL,
            filename VARCHAR2(1024),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT pk_document_attachments PRIMARY KEY (bank_id, document_id, attachment_hash),
            CONSTRAINT fk_document_attachments_document FOREIGN KEY (document_id, bank_id)
                REFERENCES documents(id, bank_id) ON DELETE CASCADE
        )
        """
    )
    op.execute(
        """
        INSERT INTO document_attachments (bank_id, document_id, attachment_hash, filename, created_at)
        SELECT a.bank_id, a.document_id, a.attachment_hash, a.filename, a.created_at
        FROM attachments a
        JOIN documents d ON d.id = a.document_id AND d.bank_id = a.bank_id
        ON CONFLICT DO NOTHING
        """
    )
    op.execute("CREATE INDEX idx_document_attachments_bank_hash ON document_attachments (bank_id, attachment_hash)")

    op.execute("ALTER TABLE attachments DROP CONSTRAINT pk_attachments")
    op.execute("DROP INDEX uq_attachments_short_id")
    op.execute(
        f"""
        DELETE FROM attachments a
        WHERE EXISTS (
            SELECT 1 FROM ({_FIRST_OWNER.format(schema="")}) owner
            WHERE a.bank_id = owner.bank_id AND a.attachment_hash = owner.attachment_hash
              AND a.document_id <> owner.document_id
        )
        """
    )
    op.execute("ALTER TABLE attachments DROP (document_id, filename)")
    op.execute("ALTER TABLE attachments ADD CONSTRAINT pk_attachments PRIMARY KEY (bank_id, attachment_hash)")
    op.execute("CREATE UNIQUE INDEX uq_attachments_short_id ON attachments (bank_id, short_id)")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)

"""Knowledge banks: documents and multimodal passages for search.

A knowledge bank is a ``banks`` row with ``kind = 'knowledge'``, so the async
operations FK, audit and the delete cascade work unchanged. Its own API lives
under ``/v1/default/knowledge-banks``.

* ``kb_documents`` — one row per file or text the caller wrote.
* ``kb_passages`` — the search unit: a PDF page, an image, an audio or video
  window, or a text passage, with its embedding. The embedding column is
  untyped here; the knowledge store pins it to the configured model's
  dimension (and builds the vector index) on first use, because that model is
  not the memory banks' one.

Knowledge banks do not ship on Oracle in v1, but Oracle still gets ``banks.kind``:
the memory-bank list filters on it, and that query runs on both dialects.

Revision ID: k7b1c2d3e4f5
Revises: 05867433a58c
Create Date: 2026-11-04
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "k7b1c2d3e4f5"
down_revision: str | Sequence[str] | None = "05867433a58c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"ALTER TABLE {schema}banks ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'memory'")
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}kb_documents (
            bank_id TEXT NOT NULL REFERENCES {schema}banks(bank_id) ON DELETE CASCADE,
            document_id TEXT NOT NULL,
            filename TEXT,
            content_type TEXT NOT NULL,
            media_type TEXT NOT NULL,
            language TEXT,
            size_bytes BIGINT NOT NULL DEFAULT 0,
            storage_key TEXT,
            content_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT,
            passage_count INT NOT NULL DEFAULT 0,
            tags TEXT[] NOT NULL DEFAULT '{{}}',
            -- Set on a document an ingestion strategy split out of another one (knowledge/ingest_strategy.py).
            source_document_id TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (bank_id, document_id)
        )
    """)
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_kb_documents_tags ON {schema}kb_documents USING GIN (tags)")
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_kb_documents_source ON {schema}kb_documents (bank_id, source_document_id) "
        "WHERE source_document_id IS NOT NULL"
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_kb_documents_updated ON {schema}kb_documents (bank_id, updated_at DESC)"
    )
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}kb_passages (
            bank_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            idx INT NOT NULL,
            modality TEXT NOT NULL,
            text TEXT,
            page INT,
            start_ms INT,
            end_ms INT,
            media_key TEXT,
            embedding vector NOT NULL,
            -- Keyword arm: the passage text stemmed in its document's language (set on insert).
            search_vector tsvector,
            PRIMARY KEY (bank_id, document_id, idx),
            FOREIGN KEY (bank_id, document_id) REFERENCES {schema}kb_documents(bank_id, document_id) ON DELETE CASCADE
        )
    """)
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_kb_passages_bank ON {schema}kb_passages (bank_id)")
    op.execute(f"CREATE INDEX IF NOT EXISTS idx_kb_passages_search ON {schema}kb_passages USING GIN (search_vector)")
    # Extra vectors of a passage, beside its primary one in kb_passages: a PDF page's band crops,
    # its whole text and its small text units. Search scores a page from all of them.
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}kb_passage_vectors (
            bank_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            idx INT NOT NULL,
            role TEXT NOT NULL,
            n INT NOT NULL,
            embedding vector NOT NULL,
            PRIMARY KEY (bank_id, document_id, idx, role, n),
            FOREIGN KEY (bank_id, document_id, idx)
                REFERENCES {schema}kb_passages(bank_id, document_id, idx) ON DELETE CASCADE
        )
    """)
    # A document too big to hold in memory is ingested in pieces: each piece of its next version is
    # staged here, out of sight of every read, and its commit moves them into kb_passages in one
    # transaction (knowledge/store.py, PassageStage). The search text is made there, once the
    # document's language is known.
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}kb_staged_passages (
            bank_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            idx INT NOT NULL,
            modality TEXT NOT NULL,
            text TEXT,
            page INT,
            start_ms INT,
            end_ms INT,
            media_key TEXT,
            embedding vector NOT NULL,
            PRIMARY KEY (bank_id, document_id, idx),
            FOREIGN KEY (bank_id, document_id) REFERENCES {schema}kb_documents(bank_id, document_id) ON DELETE CASCADE
        )
    """)
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS {schema}kb_staged_passage_vectors (
            bank_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            idx INT NOT NULL,
            role TEXT NOT NULL,
            n INT NOT NULL,
            embedding vector NOT NULL,
            PRIMARY KEY (bank_id, document_id, idx, role, n),
            FOREIGN KEY (bank_id, document_id, idx)
                REFERENCES {schema}kb_staged_passages(bank_id, document_id, idx) ON DELETE CASCADE
        )
    """)


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP TABLE IF EXISTS {schema}kb_staged_passage_vectors")
    op.execute(f"DROP TABLE IF EXISTS {schema}kb_staged_passages")
    op.execute(f"DROP TABLE IF EXISTS {schema}kb_passage_vectors")
    op.execute(f"DROP TABLE IF EXISTS {schema}kb_passages")
    op.execute(f"DROP TABLE IF EXISTS {schema}kb_documents")
    op.execute(f"ALTER TABLE {schema}banks DROP COLUMN IF EXISTS kind")


def _oracle_upgrade() -> None:
    # Only the column the shared bank queries read; the kb_* tables are PostgreSQL-only in v1.
    op.execute("ALTER TABLE banks ADD (kind VARCHAR2(32) DEFAULT 'memory' NOT NULL)")


def _oracle_downgrade() -> None:
    op.execute("ALTER TABLE banks DROP COLUMN kind")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)

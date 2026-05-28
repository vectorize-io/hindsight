"""Add text_signals column to memory_units for enriched BM25 indexing.

text_signals stores a denormalized space-separated string of entity names
(and future signals) to improve full-text search recall without polluting
the stored fact text.

- vchord: text_signals included in tokenize() at insert time
- native: search_vector GENERATED column regenerated to include text_signals
- pg_textsearch: no change (index only supports a single base column)
- pg_search: BM25 index dropped and recreated to include text_signals

Revision ID: a2b3c4d5e6f7
Revises: z1u2v3w4x5y6
Create Date: 2026-02-28
"""

import os
from collections.abc import Sequence

from alembic import context, op

from hindsight_api._pg_search import (
    PG_SEARCH_TOKENIZER_ENV,
    normalize_pg_search_tokenizer,
    pg_search_bm25_columns,
)
from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "a2b3c4d5e6f7"
down_revision: str | Sequence[str] | None = "aa2b3c4d5e6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _detect_text_search_extension() -> str:
    return os.getenv("HINDSIGHT_API_TEXT_SEARCH_EXTENSION", "native").lower()


def _pg_search_tokenizer() -> str:
    return normalize_pg_search_tokenizer(os.getenv(PG_SEARCH_TOKENIZER_ENV))


def _pg_upgrade() -> None:
    schema = _get_schema_prefix()
    table = f"{schema}memory_units"
    text_search_ext = _detect_text_search_extension()

    # Add text_signals column (nullable TEXT, populated at retain time)
    op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS text_signals TEXT")

    if text_search_ext == "native":
        # Native PostgreSQL: drop and recreate the GENERATED tsvector column to include text_signals
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS search_vector")
        op.execute(f"""
            ALTER TABLE {table}
            ADD COLUMN search_vector tsvector
            GENERATED ALWAYS AS (
                to_tsvector('english',
                    COALESCE(text, '') || ' ' ||
                    COALESCE(context, '') || ' ' ||
                    COALESCE(text_signals, '')
                )
            ) STORED
        """)
        # Recreate GIN index (was dropped with the column)
        op.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_memory_units_text_search
            ON {table} USING gin(search_vector)
        """)
    elif text_search_ext == "pg_search":
        # ParadeDB pg_search: drop the existing BM25 index and recreate it
        # to include text_signals alongside text and context.
        bm25_cols = pg_search_bm25_columns("id", ("text", "context", "text_signals"), _pg_search_tokenizer())
        op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_text_search")
        op.execute(f"""
            CREATE INDEX idx_memory_units_text_search ON {table}
            USING bm25 ({bm25_cols})
            WITH (key_field='id')
        """)

    # vchord: tokenize() call in fact_storage.py is updated to include text_signals at insert time
    # pg_textsearch: no change — index operates on the base `text` column only


def _pg_downgrade() -> None:
    schema = _get_schema_prefix()
    table = f"{schema}memory_units"
    text_search_ext = _detect_text_search_extension()

    if text_search_ext == "native":
        op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_text_search")
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS search_vector")
        op.execute(f"""
            ALTER TABLE {table}
            ADD COLUMN search_vector tsvector
            GENERATED ALWAYS AS (
                to_tsvector('english', COALESCE(text, '') || ' ' || COALESCE(context, ''))
            ) STORED
        """)
        op.execute(f"""
            CREATE INDEX idx_memory_units_text_search
            ON {table} USING gin(search_vector)
        """)
    elif text_search_ext == "pg_search":
        # Restore the original (id, text, context) BM25 index without text_signals.
        bm25_cols = pg_search_bm25_columns("id", ("text", "context"), _pg_search_tokenizer())
        op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_text_search")
        op.execute(f"""
            CREATE INDEX idx_memory_units_text_search ON {table}
            USING bm25 ({bm25_cols})
            WITH (key_field='id')
        """)

    op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS text_signals")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

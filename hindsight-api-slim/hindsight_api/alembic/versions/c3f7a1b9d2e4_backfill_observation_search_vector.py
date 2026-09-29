"""Backfill search_vector for native-backend observations.

Observations created or updated by the consolidator landed with a NULL
``search_vector`` under the ``native`` text-search backend: the
single-row INSERT/UPDATE paths in ``consolidator.py`` never populated the
tsvector (only the batch raw-fact path in ``ops_postgresql.insert_facts_batch``
did). Those observations were therefore invisible to the BM25 retrieval arm
until they were re-written by a later consolidation pass. The writer is fixed
in the same change set (all four consolidator sites now call
``to_tsvector($lang, COALESCE(text, ''))``); this migration repairs the
historical residue so existing observations become BM25-searchable without a
re-ingest.

Scope mirrors the writer fix exactly:
  * Only the ``native`` backend is touched. The gate is the column *type*:
    under ``native`` ``search_vector`` is a regular (non-generated) tsvector
    column; under ``vchord`` it is a ``bm25vector`` and under
    ``pg_textsearch`` / ``pgroonga`` / ``pg_search`` it is a dummy ``text``
    column. ``_is_regular_tsvector`` is true only for ``native``, so every
    other backend is a no-op.
  * The tsvector is built from the observation's own ``text`` only — matching
    the consolidator INSERT/UPDATE paths (entity / source / temporal signals
    are intentionally excluded; the other retrieval arms cover those).
  * Only ``fact_type = 'observation'`` rows with a NULL ``search_vector`` are
    rewritten. Raw facts already carry a populated tsvector, and the
    ``IS NULL`` predicate makes the migration idempotent and re-runnable.

The configured ``HINDSIGHT_API_TEXT_SEARCH_EXTENSION_NATIVE_LANGUAGE`` is used
so backfilled rows are lexically identical to newly-created observations. The
value is validated as a PG identifier (mirroring
``HindsightConfig.validate``) before being embedded as a SQL literal.

This is a single UPDATE per schema: it locks the targeted observation rows for
its duration. It is one-time and only touches unpopulated rows, so subsequent
online writes (which now carry the tsvector via the writer fix) are unaffected.

Oracle slot is intentionally absent: the consolidator INSERT/UPDATE paths that
this repairs are PostgreSQL-specific (``ops_postgresql``), and the native
tsvector ``search_vector`` column only exists on PostgreSQL. There is no Oracle
residue to repair.

Revision ID: c3f7a1b9d2e4
Revises: f4d1c2b3a5e6
Create Date: 2026-06-29
"""

import os
import re
from collections.abc import Sequence

from alembic import context, op
from sqlalchemy import Connection, text

from hindsight_api.alembic._dialect import run_for_dialect
from hindsight_api.config import (
    DEFAULT_TEXT_SEARCH_EXTENSION_NATIVE_LANGUAGE,
    ENV_TEXT_SEARCH_EXTENSION_NATIVE_LANGUAGE,
)

revision: str = "c3f7a1b9d2e4"
down_revision: str | Sequence[str] | None = "f4d1c2b3a5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Matches HindsightConfig.validate(): a tsvector regconfig name embedded as a
# SQL literal must be a bare PG identifier.
_PG_IDENTIFIER = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def _schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _schema_name() -> str:
    return (context.config.get_main_option("target_schema") or "public").strip('"')


def _native_language() -> str:
    """Configured native tsvector language, validated as a PG identifier."""
    lang = os.getenv(
        ENV_TEXT_SEARCH_EXTENSION_NATIVE_LANGUAGE,
        DEFAULT_TEXT_SEARCH_EXTENSION_NATIVE_LANGUAGE,
    )
    if not _PG_IDENTIFIER.fullmatch(lang):
        return DEFAULT_TEXT_SEARCH_EXTENSION_NATIVE_LANGUAGE
    return lang


def _is_regular_tsvector(conn: Connection, schema: str, table: str) -> bool:
    """True iff ``schema.table.search_vector`` is a non-generated tsvector column.

    This is the ``native`` backend signature. ``vchord`` (bm25vector) and
    ``pg_textsearch`` / ``pgroonga`` / ``pg_search`` (dummy text column) all
    fail this check, so the backfill is a no-op for them.
    """
    row = conn.execute(
        text(
            """
            SELECT is_generated, udt_name
            FROM information_schema.columns
            WHERE table_schema = :schema
              AND table_name = :table
              AND column_name = 'search_vector'
            """
        ),
        {"schema": schema, "table": table},
    ).fetchone()
    if not row:
        return False
    is_generated, udt_name = row[0], row[1]
    return udt_name == "tsvector" and is_generated != "ALWAYS"


def _pg_upgrade() -> None:
    conn = op.get_bind()
    schema_name = _schema_name()
    if not _is_regular_tsvector(conn, schema_name, "memory_units"):
        # Non-native backend (or column absent) — nothing to backfill.
        return
    schema_prefix = _schema_prefix()
    lang = _native_language()
    op.execute(
        f"""
        UPDATE {schema_prefix}memory_units
        SET search_vector = to_tsvector('{lang}'::regconfig, COALESCE(text, ''))
        WHERE fact_type = 'observation' AND search_vector IS NULL
        """
    )


def _pg_downgrade() -> None:
    # No-op: backfilled rows are indistinguishable from observations that were
    # populated by the post-fix writer, and reverting either to NULL would
    # re-break BM25 retrieval. The column simply stays populated.
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

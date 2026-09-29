"""Drop GENERATED expression on tsvector search_vector columns.

The search_vector tsvector column was originally GENERATED ALWAYS with a
hardcoded ``to_tsvector('english', ...)`` expression. To support configurable
``HINDSIGHT_API_TEXT_SEARCH_EXTENSION_NATIVE_LANGUAGE``, we convert it to a
regular tsvector column that the application populates at INSERT time via
``to_tsvector($lang, ...)``.

Existing rows retain their English-derived lexemes — switching the configured
language only affects newly-written rows. Users who need to backfill existing
rows in a different language can run an admin UPDATE after this migration.

Only the ``native`` text-search backend is affected. ``vchord``, ``pg_textsearch``,
and ``pgroonga`` use other column types or no column at all.

Revision ID: p4q5r6s7t8u9
Revises: 86f7a033d372
Create Date: 2026-05-08
"""

from collections.abc import Sequence
from dataclasses import dataclass

from alembic import context, op
from sqlalchemy import Connection, text

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "p4q5r6s7t8u9"
down_revision: str | Sequence[str] | None = "86f7a033d372"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


@dataclass(frozen=True)
class _TsvectorTableSpec:
    """Native-backend tsvector table targeted by this migration.

    ``upgrade`` is a one-way DROP EXPRESSION; ``downgrade`` re-attaches the
    original GENERATED expression so the schema returns to the state created
    by the initial migration (and a2b3c4d5e6f7_add_text_signals_column for
    memory_units).
    """

    table: str
    generated_expression: str


# Tables that may have a GENERATED tsvector ``search_vector`` column under the
# native backend. Note: the ``learnings`` table was dropped in
# p1k2l3m4n5o6_new_knowledge_architecture and ``pinned_reflections`` was renamed
# to ``reflections`` in the same migration.
_NATIVE_TSVECTOR_TABLES: tuple[_TsvectorTableSpec, ...] = (
    _TsvectorTableSpec(
        table="memory_units",
        generated_expression=(
            "to_tsvector('english', COALESCE(text, '') || ' ' || "
            "COALESCE(context, '') || ' ' || COALESCE(text_signals, ''))"
        ),
    ),
    _TsvectorTableSpec(
        table="reflections",
        generated_expression="to_tsvector('english', COALESCE(name, '') || ' ' || content)",
    ),
)


def _schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _is_generated_tsvector(conn: Connection, schema: str, table: str) -> bool:
    """Return True iff ``schema.table.search_vector`` is a GENERATED tsvector column."""
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
    return is_generated == "ALWAYS" and udt_name == "tsvector"


def _is_regular_tsvector(conn: Connection, schema: str, table: str) -> bool:
    """Return True iff ``schema.table.search_vector`` is a non-generated tsvector column."""
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


def _table_exists(conn: Connection, schema: str, table: str) -> bool:
    return bool(
        conn.execute(
            text(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = :schema AND table_name = :table
                """
            ),
            {"schema": schema, "table": table},
        ).fetchone()
    )


def _pg_upgrade() -> None:
    schema_prefix = _schema_prefix()
    schema_name = (context.config.get_main_option("target_schema") or "public").strip('"')
    conn = op.get_bind()

    for spec in _NATIVE_TSVECTOR_TABLES:
        if not _table_exists(conn, schema_name, spec.table):
            continue
        if not _is_generated_tsvector(conn, schema_name, spec.table):
            # Either the column doesn't exist (non-native backend) or it's
            # already a regular tsvector — nothing to do.
            continue
        op.execute(f"ALTER TABLE {schema_prefix}{spec.table} ALTER COLUMN search_vector DROP EXPRESSION")


def _pg_downgrade() -> None:
    schema_prefix = _schema_prefix()
    schema_name = (context.config.get_main_option("target_schema") or "public").strip('"')
    conn = op.get_bind()

    for spec in _NATIVE_TSVECTOR_TABLES:
        if not _table_exists(conn, schema_name, spec.table):
            continue
        # Only restore the GENERATED expression if a non-generated tsvector
        # column exists — otherwise the table is on a different backend.
        if not _is_regular_tsvector(conn, schema_name, spec.table):
            continue
        # Drop and recreate to re-attach the GENERATED expression. Index will be
        # recreated by re-running ensure_text_search_extension on next startup.
        op.execute(f"DROP INDEX IF EXISTS {schema_prefix}idx_{spec.table}_text_search")
        op.execute(f"ALTER TABLE {schema_prefix}{spec.table} DROP COLUMN search_vector")
        op.execute(
            f"ALTER TABLE {schema_prefix}{spec.table} "
            f"ADD COLUMN search_vector tsvector GENERATED ALWAYS AS ({spec.generated_expression}) STORED"
        )
        op.execute(f"CREATE INDEX idx_{spec.table}_text_search ON {schema_prefix}{spec.table} USING gin(search_vector)")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

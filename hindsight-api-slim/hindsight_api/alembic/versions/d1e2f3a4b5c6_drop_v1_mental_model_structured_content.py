"""Drop schema-v1 mental model structured_content

``mental_models.structured_content`` held a typed block tree (paragraph /
bullet_list / ordered_list / code / table) that was *parsed* out of the
document's markdown. That parse was lossy for every construct the union could
not express — nested lists, list continuation lines, blockquotes, hard line
breaks, horizontal rules, HTML, indented code, table alignment, a table row
missing an outer pipe — and the loss was a fixed point, so a document could
never recover on its own (#3361).

Schema v2 stores each block as a verbatim markdown fragment instead, so the
v1 rows are a strictly worse copy of the same document: the row's ``content``
is the faithful text, and re-deriving the structure from it is lossless. There
is nothing in a v1 blob worth converting, so this drops them and lets the next
refresh rebuild the structure from ``content`` — a one-time, lossless import.

``content`` itself is deliberately untouched: it is what users read, and no
refresh has to run for it to keep working.

Revision ID: d1e2f3a4b5c6
Revises: c4e8a1b7d2f6
Create Date: 2026-08-19
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "d1e2f3a4b5c6"
down_revision: str | Sequence[str] | None = "c4e8a1b7d2f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA_VERSION = "2"


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    # ``IS DISTINCT FROM`` so a blob with no ``version`` key (NULL on the left)
    # is cleared too — an untagged shape is by definition not v2.
    op.execute(
        f"""
        UPDATE {schema}mental_models
        SET structured_content = NULL
        WHERE structured_content IS NOT NULL
          AND (structured_content ->> 'version') IS DISTINCT FROM '{SCHEMA_VERSION}'
        """
    )


def _pg_downgrade() -> None:
    # The v1 blobs are gone and are not reconstructible — nor worth
    # reconstructing: the code that read them no longer exists, and rolling back
    # leaves every model rebuilding its baseline from ``content``, which is what
    # a NULL already means.
    pass


def _oracle_upgrade() -> None:
    # ``structured_content`` is a CLOB with an ``IS JSON`` check, so the version
    # is read with JSON_VALUE rather than PG's ``->>``. JSON_VALUE returns NULL
    # both for a missing key and for unparseable JSON; either way it is not v2.
    op.execute(
        f"""
        UPDATE mental_models
        SET structured_content = NULL
        WHERE structured_content IS NOT NULL
          AND (
              JSON_VALUE(structured_content, '$.version') IS NULL
              OR JSON_VALUE(structured_content, '$.version') <> '{SCHEMA_VERSION}'
          )
        """
    )


def _oracle_downgrade() -> None:
    # See _pg_downgrade.
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)

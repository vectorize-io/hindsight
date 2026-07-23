"""Add knowledge_pages table (knowledge-base hierarchy).

The knowledge base organizes synthesized mental models into a navigable tree of
**folders** and **pages**. A page references the mental model that holds its
content (``mental_model_id``); a folder is a pure container (``mental_model_id``
NULL). Hierarchy is a single self-referential ``parent_id`` so folders can nest
arbitrarily. Content stays in ``mental_models`` — this table is metadata + tree
structure only.

Revision ID: a9b8c7d6e5f4
Revises: d7b2f8a1c934
Create Date: 2026-06-25
"""

from collections.abc import Sequence

from alembic import context, op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "a9b8c7d6e5f4"
down_revision: str | Sequence[str] | None = "d7b2f8a1c934"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_schema_prefix() -> str:
    """Schema-qualifier for raw SQL on PG (multi-tenant search_path)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _pg_upgrade() -> None:
    schema = _pg_schema_prefix()
    # parent_id self-FK cascades so deleting a folder row removes its whole
    # subtree of rows in one shot. The mental_model FK is composite (matches the
    # mental_models (id, bank_id) PK) and cascades too, so deleting a page's
    # mental model removes the page row — folders skip the FK because a NULL
    # column in a composite FK is not enforced (MATCH SIMPLE).
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}knowledge_pages (
            id VARCHAR(64) NOT NULL,
            bank_id TEXT NOT NULL,
            parent_id VARCHAR(64),
            kind VARCHAR(16) NOT NULL,
            name TEXT NOT NULL,
            mental_model_id VARCHAR(64),
            sort_order INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            CONSTRAINT pk_knowledge_pages PRIMARY KEY (id),
            CONSTRAINT ck_knowledge_pages_kind CHECK (kind IN ('folder', 'page')),
            CONSTRAINT fk_kp_bank FOREIGN KEY (bank_id)
                REFERENCES {schema}banks(bank_id) ON DELETE CASCADE,
            CONSTRAINT fk_kp_parent FOREIGN KEY (parent_id)
                REFERENCES {schema}knowledge_pages(id) ON DELETE CASCADE,
            CONSTRAINT fk_kp_mm FOREIGN KEY (mental_model_id, bank_id)
                REFERENCES {schema}mental_models(id, bank_id) ON DELETE CASCADE
        )
        """
    )
    op.execute(
        f"CREATE INDEX IF NOT EXISTS idx_kp_bank_parent ON {schema}knowledge_pages (bank_id, parent_id, sort_order)"
    )


def _pg_downgrade() -> None:
    schema = _pg_schema_prefix()
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_kp_bank_parent")
    op.execute(f"DROP TABLE IF EXISTS {schema}knowledge_pages")


def _oracle_upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_pages (
            id VARCHAR2(64) NOT NULL,
            bank_id VARCHAR2(256) NOT NULL,
            parent_id VARCHAR2(64),
            kind VARCHAR2(16) NOT NULL,
            name CLOB NOT NULL,
            mental_model_id VARCHAR2(64),
            sort_order NUMBER DEFAULT 0 NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT pk_knowledge_pages PRIMARY KEY (id),
            CONSTRAINT ck_knowledge_pages_kind CHECK (kind IN ('folder', 'page')),
            CONSTRAINT fk_kp_bank FOREIGN KEY (bank_id)
                REFERENCES banks(bank_id) ON DELETE CASCADE,
            CONSTRAINT fk_kp_parent FOREIGN KEY (parent_id)
                REFERENCES knowledge_pages(id) ON DELETE CASCADE,
            CONSTRAINT fk_kp_mm FOREIGN KEY (mental_model_id, bank_id)
                REFERENCES mental_models(id, bank_id) ON DELETE CASCADE
        )
        """
    )
    op.execute("CREATE INDEX idx_kp_bank_parent ON knowledge_pages (bank_id, parent_id, sort_order)")


def _oracle_downgrade() -> None:
    op.execute("DROP TABLE knowledge_pages CASCADE CONSTRAINTS")


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)

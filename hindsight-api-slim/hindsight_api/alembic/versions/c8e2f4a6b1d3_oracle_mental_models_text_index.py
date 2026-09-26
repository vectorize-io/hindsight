"""Oracle Text index for knowledge-page search

``search_knowledge_pages`` fuses a BM25 arm over ``mental_models`` with the vector
arm. PostgreSQL has had ``idx_mental_models_text_search`` since the knowledge pages
landed; Oracle never got a counterpart, so the arm had nothing to run against. This
adds the Oracle Text (``CTXSYS.CONTEXT``) index the Oracle branch of
``knowledge_bm25_arm`` queries with ``CONTAINS``/``SCORE`` — the same shape the
baseline gives ``memory_units(text)``. ``SYNC (ON COMMIT)`` keeps it current without
a maintenance job.

PostgreSQL: no-op (the index already exists there).

Revision ID: c8e2f4a6b1d3
Revises: e5b1c7d3a902
Create Date: 2026-09-26
"""

from collections.abc import Sequence

from alembic import op

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c8e2f4a6b1d3"
down_revision: str | Sequence[str] | None = "e5b1c7d3a902"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# PL/SQL wrappers so reruns are safe: ORA-00955 (already exists) on create,
# ORA-01418 (does not exist) on drop.
_CREATE = (
    "BEGIN "
    "EXECUTE IMMEDIATE '"
    "CREATE INDEX idx_mental_models_text_search ON mental_models(content) "
    "INDEXTYPE IS CTXSYS.CONTEXT "
    "PARAMETERS (''SYNC (ON COMMIT)'')"
    "'; "
    "EXCEPTION WHEN OTHERS THEN "
    "IF SQLCODE = -955 THEN NULL; ELSE RAISE; END IF; "
    "END;"
)

_DROP = (
    "BEGIN "
    "EXECUTE IMMEDIATE 'DROP INDEX idx_mental_models_text_search'; "
    "EXCEPTION WHEN OTHERS THEN "
    "IF SQLCODE = -1418 THEN NULL; ELSE RAISE; END IF; "
    "END;"
)


def _oracle_upgrade() -> None:
    bind = op.get_bind()
    bind.exec_driver_sql("ALTER SESSION SET DDL_LOCK_TIMEOUT = 30")
    bind.exec_driver_sql(_CREATE)


def _oracle_downgrade() -> None:
    op.get_bind().exec_driver_sql(_DROP)


def upgrade() -> None:
    run_for_dialect(pg=None, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=None, oracle=_oracle_downgrade)

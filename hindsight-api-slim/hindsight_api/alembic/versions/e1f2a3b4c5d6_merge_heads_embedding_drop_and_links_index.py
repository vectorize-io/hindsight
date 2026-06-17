"""Merge two divergent migration heads.

``d4f6a8c2e1b3`` (drop the curation-archive embedding column) and
``2071c7518f88`` (add the memory_links(bank_id, link_type) index) were authored
in parallel off the same parent (``a1d3f5b7c9e2``) and merged independently,
leaving the DAG with two heads. This is a no-op merge that re-unifies them so
``alembic upgrade head`` is unambiguous again (enforced by
``tests/test_alembic_dag.py::test_single_head``).

Revision ID: e1f2a3b4c5d6
Revises: d4f6a8c2e1b3, 2071c7518f88
Create Date: 2026-06-16
"""

from collections.abc import Sequence

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "e1f2a3b4c5d6"
down_revision: str | Sequence[str] | None = ("d4f6a8c2e1b3", "2071c7518f88")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_upgrade() -> None:
    # Pure DAG merge — both parents already applied their schema changes.
    pass


def _pg_downgrade() -> None:
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

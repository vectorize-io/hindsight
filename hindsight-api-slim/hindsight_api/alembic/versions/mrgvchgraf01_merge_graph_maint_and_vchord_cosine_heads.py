"""Merge divergent heads from graph maintenance queue and vchord cosine opclass

Two unrelated PRs landed on main with parents on the same revision and never
got a merge revision: ``b5a4c3e2f1d8`` (add_graph_maintenance_queue) and
``b8c9d0e1f2a3`` (vchord_cosine_opclass). ``tests/test_alembic_dag.py::test_single_head``
catches the divergence and recommends the canonical fix
``alembic merge heads -m '<reason>'`` — that's what this file is.

No schema change; the parents already applied their own migrations
independently. This file only restores a single-head DAG so future authors
have an unambiguous parent to attach to.

Revision ID: mrgvchgraf01
Revises: b5a4c3e2f1d8, b8c9d0e1f2a3
Create Date: 2026-05-29
"""

from collections.abc import Sequence

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "mrgvchgraf01"
down_revision: tuple[str, ...] = ("b5a4c3e2f1d8", "b8c9d0e1f2a3")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_upgrade() -> None:
    pass


def _pg_downgrade() -> None:
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade)

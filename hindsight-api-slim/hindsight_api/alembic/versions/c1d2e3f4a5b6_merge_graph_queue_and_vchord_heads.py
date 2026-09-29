"""Merge graph_maintenance_queue and vchord_cosine_opclass heads.

Revision ID: c1d2e3f4a5b6
Revises: b5a4c3e2f1d8, b8c9d0e1f2a3
Create Date: 2026-05-29

PRs #1668 (vchord cosine opclass) and #1772 (async link recompute) both
branched off the same parent and were merged onto main without rebasing,
leaving two parallel Alembic heads. This is a structural merge revision
with no schema changes — its only job is to unify the DAG so
``alembic upgrade head`` is unambiguous again.
"""

from collections.abc import Sequence

from hindsight_api.alembic._dialect import run_for_dialect

revision: str = "c1d2e3f4a5b6"
down_revision: str | Sequence[str] | None = ("b5a4c3e2f1d8", "b8c9d0e1f2a3")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _pg_upgrade() -> None:
    pass


def _pg_downgrade() -> None:
    pass


def _oracle_upgrade() -> None:
    pass


def _oracle_downgrade() -> None:
    pass


def upgrade() -> None:
    run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


def downgrade() -> None:
    run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)

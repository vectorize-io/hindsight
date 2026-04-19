"""Merge divergent migration heads for v0.5.3

v0.5.3 shipped with two migration heads that were never unified:

  * ``c4x5y6z7a8b9`` — delta-refresh chain
    (``add_last_refreshed_source_query`` ->
     ``add_structured_content_to_mental_models`` ->
     ``backsweep_orphan_observations_v2``)

  * ``h3i4j5k6l7m8`` — per-bank vector indexes / audit log chain
    (the ``merge_heads_and_add_unit_entities_index`` subtree)

Both fork from ``z1u2v3w4x5y6``. Upgrades from v0.5.2 still succeed — the
walker applies the three c4x5 revisions and leaves the database stamped at
both heads — but the result is a split DAG: ``alembic upgrade head``
(singular) is ambiguous, and any future migration has to pick one head as
its parent, orphaning the other.

This revision linearises the DAG into a single head. It has no schema
effect.

Revision ID: 8c6fa6f7230b
Revises: c4x5y6z7a8b9, h3i4j5k6l7m8
Create Date: 2026-04-18
"""

from collections.abc import Sequence

revision: str = "8c6fa6f7230b"
down_revision: str | Sequence[str] | None = ("c4x5y6z7a8b9", "h3i4j5k6l7m8")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

"""Schema invariant: memory_links FKs to memory_units must be deferred.

Locks in the fix from migration ``9f8e7d6c5b4a``. The two FK constraints
(``from_unit_id`` and ``to_unit_id``) reference ``memory_units(id)`` with
``ON DELETE CASCADE`` and must be ``DEFERRABLE INITIALLY DEFERRED`` so an
INSERT into ``memory_links`` does not race with a cascading DELETE on the
referenced ``memory_units`` row (e.g. delta-retain superseding chunks
which CASCADEs through chunks → memory_units → memory_links).

If a future migration accidentally reverts the deferral, or someone
recreates the constraint with the SQLAlchemy default (NOT DEFERRABLE),
this test fails — preserving the intended concurrency semantics.

The behaviour test (concurrent INSERT + cascading DELETE no longer
deadlocks) is hard to write deterministically because PG's deadlock
detector is racy; this schema-shape test is the durable guard.
"""

import asyncpg
import pytest

_TARGET_FKS = (
    "fk_memory_links_from_unit_id_memory_units",
    "fk_memory_links_to_unit_id_memory_units",
)


@pytest.mark.asyncio
async def test_memory_links_to_memory_units_fks_are_deferred(pg0_db_url):
    """Both memory_links → memory_units FKs must be deferrable + initially deferred."""
    conn = await asyncpg.connect(pg0_db_url)
    try:
        rows = await conn.fetch(
            """
            SELECT conname,
                   condeferrable,
                   condeferred,
                   confdeltype
            FROM pg_constraint
            WHERE conname = ANY($1::text[])
              AND conrelid = 'public.memory_links'::regclass
            ORDER BY conname
            """,
            list(_TARGET_FKS),
        )
    finally:
        await conn.close()

    by_name = {r["conname"]: r for r in rows}
    missing = set(_TARGET_FKS) - by_name.keys()
    assert not missing, (
        f"FK constraints missing on memory_links: {sorted(missing)}. "
        "Initial schema migration did not run, or the constraint was "
        "renamed without updating this test."
    )

    for fk_name in _TARGET_FKS:
        r = by_name[fk_name]
        assert r["condeferrable"] is True, (
            f"FK {fk_name} must be DEFERRABLE — migration 9f8e7d6c5b4a "
            "set this to eliminate INSERT-vs-cascade-DELETE deadlocks. "
            "Was the constraint recreated without DEFERRABLE INITIALLY "
            "DEFERRED?"
        )
        assert r["condeferred"] is True, (
            f"FK {fk_name} must be INITIALLY DEFERRED — DEFERRABLE alone "
            "does nothing unless callers explicitly SET CONSTRAINTS "
            "DEFERRED in every transaction. We chose INITIALLY DEFERRED "
            "to make every INSERT safe by default."
        )
        # ON DELETE CASCADE is still required so deleting a memory_unit
        # cleans up its links (otherwise we accumulate dangling rows).
        # 'c' = CASCADE in pg_constraint.confdeltype. asyncpg returns the
        # PG ``char`` type as bytes, so compare against b'c'.
        assert r["confdeltype"] == b"c", (
            f"FK {fk_name} must keep ON DELETE CASCADE — the deferred-FK "
            "fix only changes WHEN the constraint is checked, not what "
            "happens when the parent row is deleted."
        )

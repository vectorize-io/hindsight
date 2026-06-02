"""Regression for #1882 — the ``to_unit_id`` side of the memory_links FK race.

Mirror of #1795 (which covered ``from_unit_id``). Temporal / ANN link inserts
reference a **pre-existing neighbor** unit as ``to_unit_id`` and bulk-insert with
``skip_exists_check=True`` (see ``create_temporal_links_batch_per_fact`` /
``compute_semantic_links_ann``). The FK
``fk_memory_links_to_unit_id_memory_units`` is ``DEFERRABLE INITIALLY DEFERRED``
(migration ``9f8e7d6c5b4a``), so the INSERT takes **no lock** on the parent
``memory_units`` row — the check is pushed to COMMIT. A concurrent committed
DELETE of that neighbor (consolidation pruning observation units at
``consolidator.py``; document re-tracking at ``fact_storage.py``) then makes the
deferred check fail at COMMIT with the exact violation reported in #1882.

This is normally a timing race, but it is made **fully deterministic** here by
driving the two connections by hand (no sleeps):

  1. conn A opens a txn and inserts a temporal link ``(from, neighbor)`` via the
     real ``_bulk_insert_links`` path — but does NOT commit.
  2. conn B deletes the neighbor unit and commits.
  3. conn A commits.

Pre-fix: step 2 succeeds instantly (A holds no lock on the neighbor), the
neighbor is gone, and step 3 raises ``ForeignKeyViolationError`` → this test
fails, reproducing #1882.

Post-fix (A locks referenced parents ``FOR KEY SHARE`` before inserting): step 2
blocks on A's lock and is cut off by ``statement_timeout``; the neighbor
survives and step 3 commits cleanly → this test passes.
"""

import uuid

import asyncpg
import pytest

from hindsight_api.engine.db.ops_postgresql import PostgreSQLOps
from hindsight_api.engine.db.postgresql import PostgresConnection
from hindsight_api.engine.retain.link_utils import _bulk_insert_links


async def _insert_unit(conn: asyncpg.Connection, bank_id: str) -> str:
    """Insert one committed memory_unit (autocommit) and return its id as text."""
    return await conn.fetchval(
        """
        INSERT INTO memory_units (bank_id, text, event_date, fact_type)
        VALUES ($1, 'unit', now(), 'world')
        RETURNING id::text
        """,
        bank_id,
    )


@pytest.mark.asyncio
async def test_to_unit_id_survives_concurrent_neighbor_delete(pg0_db_url):
    bank_id = f"fk-to-race-{uuid.uuid4().hex}"
    ops = PostgreSQLOps()

    # Three independent connections: a setup/cleanup conn, the retain txn (A),
    # and the concurrent deleter (B). Independent connections give us full,
    # deterministic control over transaction boundaries.
    setup = await asyncpg.connect(pg0_db_url)
    conn_a = await asyncpg.connect(pg0_db_url)
    conn_b = await asyncpg.connect(pg0_db_url)
    try:
        # Two committed units: `from_id` plays the freshly-inserted fact,
        # `neighbor_id` plays the pre-existing temporal neighbor (to_unit_id).
        from_id = await _insert_unit(setup, bank_id)
        neighbor_id = await _insert_unit(setup, bank_id)

        # --- Step 1: conn A inserts the temporal link, does NOT commit ---
        tx_a = conn_a.transaction()
        await tx_a.start()
        await _bulk_insert_links(
            PostgresConnection(conn_a),
            [(from_id, neighbor_id, "temporal", 1.0, None)],
            bank_id=bank_id,
            skip_exists_check=True,  # the racy path: no EXISTS guard
            ops=ops,
        )

        # --- Step 2: conn B deletes the neighbor and commits (autocommit) ---
        # statement_timeout bounds the post-fix path, where A holds FOR KEY
        # SHARE on the neighbor and this DELETE must block.
        await conn_b.execute("SET statement_timeout = '1000ms'")
        neighbor_deleted = False
        try:
            await conn_b.execute("DELETE FROM memory_units WHERE id = $1", uuid.UUID(neighbor_id))
            neighbor_deleted = True
        except asyncpg.QueryCanceledError:
            # Blocked on A's lock until statement_timeout — the fixed behavior.
            neighbor_deleted = False

        # --- Step 3: conn A commits — deferred FK check fires here ---
        fk_error: Exception | None = None
        try:
            await tx_a.commit()
        except asyncpg.ForeignKeyViolationError as exc:
            fk_error = exc
            await conn_a.execute("ROLLBACK")

        assert fk_error is None, (
            "memory_links.to_unit_id FK violation at COMMIT (#1882): a concurrent "
            "DELETE removed the referenced neighbor unit between the deferred-FK "
            "link INSERT and COMMIT. The link insert must lock referenced parent "
            f"units (FOR KEY SHARE) or drop vanished ones. neighbor_deleted="
            f"{neighbor_deleted}. {fk_error}"
        )

        # Post-fix end state: the neighbor was protected, so the link persists.
        assert not neighbor_deleted, (
            "Expected the concurrent DELETE to block on A's FOR KEY SHARE lock "
            "and be cut off by statement_timeout; it succeeded instead, meaning "
            "the parent row was not locked before the link insert."
        )
        link_count = await setup.fetchval(
            "SELECT count(*) FROM memory_links WHERE from_unit_id = $1 AND to_unit_id = $2",
            uuid.UUID(from_id),
            uuid.UUID(neighbor_id),
        )
        assert link_count == 1
    finally:
        await setup.execute("DELETE FROM memory_links WHERE bank_id = $1", bank_id)
        await setup.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
        await setup.close()
        await conn_a.close()
        await conn_b.close()

"""Oracle: a link to a neighbor deleted since it was read must not fail the batch (#6440).

Retain reads its link neighbors (ANN / temporal) before the write transaction. If a
concurrent writer deletes a neighbor in between, Oracle's immediate FK used to fail the
whole link insert with ORA-02291, and the retain task retried forever. The insert now
drops links to vanished units and locks the live ones until commit, like PostgreSQL.
"""

import asyncio
import uuid
from datetime import UTC, datetime

import pytest

from hindsight_api.engine.db import create_database_backend
from hindsight_api.engine.memories.pg.links import _bulk_insert_links

pytestmark = pytest.mark.oracle


@pytest.fixture(autouse=True)
def _oracle_backend(monkeypatch):
    """Table names resolve per backend from config; point it at Oracle for these tests."""
    from hindsight_api.config import clear_config_cache

    monkeypatch.setenv("HINDSIGHT_API_DATABASE_BACKEND", "oracle")
    clear_config_cache()
    yield
    clear_config_cache()


async def _insert_unit(conn, bank_id: str) -> str:
    unit_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO memory_units (id, bank_id, text, event_date, fact_type) VALUES ($1, $2, 'unit', $3, 'world')",
        unit_id,
        bank_id,
        datetime.now(UTC),
    )
    return str(unit_id)


async def _link_targets(conn, from_id: str) -> set[str]:
    # Raw memory_links read: this drives the link insert below the engine, with a
    # concurrent delete forged on a second connection — no engine read covers that.
    rows = await conn.fetch("SELECT to_unit_id FROM memory_links WHERE from_unit_id = $1", uuid.UUID(from_id))
    return {str(row["to_unit_id"]) for row in rows}


@pytest.mark.asyncio
async def test_link_to_deleted_neighbor_is_dropped_and_live_neighbor_is_locked(oracle_db_url):
    backend = create_database_backend("oracle")
    await backend.initialize(oracle_db_url, min_size=1, max_size=3)
    bank_id = f"test-6440-{uuid.uuid4().hex[:8]}"
    try:
        async with backend.acquire() as conn:
            await conn.execute("INSERT INTO banks (bank_id, name) VALUES ($1, $2)", bank_id, bank_id)
            from_id = await _insert_unit(conn, bank_id)
            gone_id = await _insert_unit(conn, bank_id)
            live_id = await _insert_unit(conn, bank_id)
            # The neighbor vanishes after the (simulated) ANN read, before the insert.
            await conn.execute("DELETE FROM memory_units WHERE id = $1", uuid.UUID(gone_id))

        delete_live: asyncio.Task | None = None
        async with backend.transaction() as tx:
            await _bulk_insert_links(
                tx,
                [(from_id, gone_id, "semantic", 0.9, None), (from_id, live_id, "semantic", 0.8, None)],
                bank_id=bank_id,
                ops=backend.ops,
            )

            # A concurrent delete of the live neighbor must wait for this commit.
            async def _delete_live() -> None:
                async with backend.acquire() as other:
                    await other.execute("DELETE FROM memory_units WHERE id = $1", uuid.UUID(live_id))

            delete_live = asyncio.create_task(_delete_live())
            await asyncio.sleep(1.0)
            assert not delete_live.done(), "the live neighbor must stay locked until the link insert commits"

        await delete_live
        async with backend.acquire() as conn:
            assert await _link_targets(conn, from_id) == set(), "the delete cascades to the committed link"
    finally:
        async with backend.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)
        await backend.shutdown()


@pytest.mark.asyncio
async def test_links_land_when_every_neighbor_is_live(oracle_db_url):
    backend = create_database_backend("oracle")
    await backend.initialize(oracle_db_url, min_size=1, max_size=2)
    bank_id = f"test-6440-{uuid.uuid4().hex[:8]}"
    try:
        async with backend.acquire() as conn:
            await conn.execute("INSERT INTO banks (bank_id, name) VALUES ($1, $2)", bank_id, bank_id)
            from_id = await _insert_unit(conn, bank_id)
            gone_id = await _insert_unit(conn, bank_id)
            live_id = await _insert_unit(conn, bank_id)
            await conn.execute("DELETE FROM memory_units WHERE id = $1", uuid.UUID(gone_id))

        async with backend.transaction() as tx:
            await _bulk_insert_links(
                tx,
                [(from_id, gone_id, "semantic", 0.9, None), (from_id, live_id, "semantic", 0.8, None)],
                bank_id=bank_id,
                ops=backend.ops,
            )

        async with backend.acquire() as conn:
            assert await _link_targets(conn, from_id) == {live_id}
    finally:
        async with backend.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", bank_id)
            await conn.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)
        await backend.shutdown()

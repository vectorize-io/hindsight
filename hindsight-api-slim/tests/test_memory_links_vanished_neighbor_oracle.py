"""Oracle: a link to a neighbor deleted since it was read must not fail the batch (#6440).

Retain reads its link neighbors (ANN / temporal) before the write transaction. If a
concurrent writer deletes a neighbor in between, Oracle's immediate FK used to fail the
whole link insert with ORA-02291, and the retain task retried forever. The insert now
drops just the links to vanished units, like PostgreSQL; a link that lands makes a
concurrent delete of its endpoint wait until commit.
"""

import asyncio
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.db import create_database_backend
from hindsight_api.engine.db.base import DatabaseConnection
from hindsight_api.engine.db.ops_oracle import OracleOps
from hindsight_api.engine.memories.pg.links import _bulk_insert_links

_FROM, _GONE, _LIVE = (str(uuid.UUID(int=i)) for i in (1, 2, 3))


@pytest.mark.asyncio
async def test_vanished_endpoint_drops_only_that_link() -> None:
    """Mechanics, no database: a parent-key-not-found on a unit FK skips that one
    link and the rest are still inserted; any other error still raises."""
    fk_gone = Exception("ORA-02291: integrity constraint (HS.FK_ML_TO) violated - parent key not found")

    async def _execute(_query: str, _from_id: str, to_id: str, *_rest: object) -> None:
        if to_id == _GONE:
            raise fk_gone

    conn = AsyncMock()
    conn.execute.side_effect = _execute

    links = [(_FROM, _GONE, "semantic", 0.9, None), (_FROM, _LIVE, "semantic", 0.8, None)]
    await OracleOps().bulk_insert_links(conn, "memory_links", links, "bank-1", str(uuid.UUID(int=0)))

    inserted = [call.args[1:3] for call in conn.execute.await_args_list]
    assert inserted == [(_FROM, _GONE), (_FROM, _LIVE)], "every link is attempted, one statement each"

    for other in (
        Exception("ORA-02291: integrity constraint (HS.FK_ML_ENTITY) violated - parent key not found"),
        Exception("ORA-00060: deadlock detected while waiting for resource"),
    ):
        conn.execute.side_effect = other
        with pytest.raises(Exception, match=str(other)[:9]):
            await OracleOps().bulk_insert_links(conn, "memory_links", links, "bank-1", str(uuid.UUID(int=0)))


@pytest.fixture
def _oracle_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Table names resolve per backend from config; point it at Oracle for these tests."""
    from hindsight_api.config import clear_config_cache

    monkeypatch.setenv("HINDSIGHT_API_DATABASE_BACKEND", "oracle")
    clear_config_cache()
    yield
    clear_config_cache()


async def _insert_unit(conn: DatabaseConnection, bank_id: str) -> str:
    unit_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO memory_units (id, bank_id, text, event_date, fact_type) VALUES ($1, $2, 'unit', $3, 'world')",
        unit_id,
        bank_id,
        datetime.now(UTC),
    )
    return str(unit_id)


async def _link_targets(conn: DatabaseConnection, from_id: str) -> set[str]:
    # Raw memory_links read: this drives the link insert below the engine, with a
    # concurrent delete forged on a second connection — no engine read covers that.
    rows = await conn.fetch("SELECT to_unit_id FROM memory_links WHERE from_unit_id = $1", uuid.UUID(from_id))
    return {str(row["to_unit_id"]) for row in rows}


@pytest.mark.oracle
@pytest.mark.usefixtures("_oracle_backend")
@pytest.mark.asyncio
async def test_delete_of_a_linked_neighbor_waits_for_the_link_commit(oracle_db_url: str) -> None:
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


@pytest.mark.oracle
@pytest.mark.usefixtures("_oracle_backend")
@pytest.mark.asyncio
async def test_only_the_vanished_neighbors_link_is_dropped(oracle_db_url: str) -> None:
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

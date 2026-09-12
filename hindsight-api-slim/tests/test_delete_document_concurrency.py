"""Schedule real SQL interleavings through the engine's delete methods (#4251).

The LLM cannot prescribe exact co-source IDs or transaction interleavings, so seed
that internal state directly. Only authentication and post-commit background work
are stubbed; acquisition, transactions, cascades and observation cleanup are real.
"""

import asyncio
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncpg
import pytest
import pytest_asyncio

from hindsight_api.engine.db.postgresql import PostgresConnection, PostgreSQLBackend
from hindsight_api.engine.memory_engine import MemoryEngine

pytestmark = [pytest.mark.asyncio, pytest.mark.memory_backend_incompatible]


@dataclass
class DeleteRace:
    setup: asyncpg.Connection
    banks: list[str]
    engines: list[MemoryEngine]
    second_pid: int


async def seed_sources(conn: asyncpg.Connection, bank: str) -> None:
    await conn.execute("INSERT INTO banks(bank_id) VALUES($1)", bank)
    sources = [uuid.uuid4(), uuid.uuid4()]
    for document_id, unit_id in zip(["a", "b"], sources):
        await conn.execute("INSERT INTO documents(id, bank_id) VALUES($1, $2)", document_id, bank)
        await conn.execute(
            "INSERT INTO memory_units(id, bank_id, document_id, text, fact_type) VALUES($1, $2, $3, 'fact', 'world')",
            unit_id,
            bank,
            document_id,
        )
    # No entities or links: their maintenance queue locks would hide the
    # observation-source cycle. Both documents are explicit sources of this row.
    await conn.execute(
        "INSERT INTO memory_units(bank_id, text, fact_type, source_memory_ids) "
        "VALUES($1, 'shared observation', 'observation', $2)",
        bank,
        sources,
    )


@pytest_asyncio.fixture
async def delete_race(pg0_db_url):
    setup = await asyncpg.connect(pg0_db_url)
    banks = [f"delete-race-{uuid.uuid4().hex}" for _ in range(2)]
    backends = []
    engines = []
    try:
        for bank in banks:
            await seed_sources(setup, bank)
        for _ in range(2):
            backend = PostgreSQLBackend()
            await backend.initialize(pg0_db_url, min_size=1, max_size=1, command_timeout=15)
            backends.append(backend)
            engine = MemoryEngine.__new__(MemoryEngine)
            engine._initialized = True
            engine._backend = backend
            engine._operation_validator = None
            engine._tenant_extension = None
            engine._authenticate_tenant = AsyncMock(return_value="public")
            engine._bank_stats_cache = SimpleNamespace(invalidate=AsyncMock())
            engine._config_resolver = SimpleNamespace(
                resolve_full_config=AsyncMock(return_value=SimpleNamespace(enable_auto_consolidation=False))
            )
            engine._submit_refreshes_for_retracted_grounding = AsyncMock()
            engine.submit_async_graph_maintenance = AsyncMock()
            engine._submit_vector_index_maintenance_quietly = AsyncMock()
            engines.append(engine)
        async with backends[1].acquire() as conn:
            pid = await conn.fetchval("SELECT pg_backend_pid()")
        yield DeleteRace(setup, banks, engines, pid)
    finally:
        for backend in backends:
            await backend.shutdown()
        for bank in banks:
            await setup.execute("DELETE FROM memory_units WHERE bank_id=$1", bank)
            await setup.execute("DELETE FROM documents WHERE bank_id=$1", bank)
            await setup.execute("DELETE FROM banks WHERE bank_id=$1", bank)
        await setup.close()


async def wait_for_lock(race: DeleteRace) -> None:
    # Observe a real wait rather than relying on a scheduler-dependent sleep.
    async with asyncio.timeout(10):
        while (
            await race.setup.fetchval("SELECT wait_event_type FROM pg_stat_activity WHERE pid=$1", race.second_pid)
            != "Lock"
        ):
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("rollback", [False, True])
async def test_shared_observation_deletes_complete_after_commit_or_rollback(delete_race, request_context, rollback):
    race = delete_race
    first, second = race.engines
    paused, release = asyncio.Event(), asyncio.Event()
    original = first._delete_stale_observations_for_memories

    async def pause_after_delete(conn, bank_id, fact_ids):
        paused.set()
        await release.wait()
        if rollback:
            raise RuntimeError("injected transaction rollback")
        return await original(conn, bank_id, fact_ids)

    first._delete_stale_observations_for_memories = pause_after_delete
    tasks = [asyncio.create_task(first.delete_document("a", race.banks[0], request_context=request_context))]
    try:
        await asyncio.wait_for(paused.wait(), 10)
        tasks.append(asyncio.create_task(second.delete_document("b", race.banks[0], request_context=request_context)))
        await wait_for_lock(race)
    finally:
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 20)
    assert results[1] == {"document_deleted": 1, "memory_units_deleted": 1}
    if rollback:
        assert isinstance(results[0], RuntimeError) and str(results[0]) == "injected transaction rollback"
        assert await first.get_document("a", race.banks[0], request_context=request_context) is not None
    else:
        assert results[0] == {"document_deleted": 1, "memory_units_deleted": 1}
        assert await first.get_document("a", race.banks[0], request_context=request_context) is None
    assert await second.get_document("b", race.banks[0], request_context=request_context) is None
    assert await second.delete_document("b", race.banks[0], request_context=request_context) == {
        "document_deleted": 0,
        "memory_units_deleted": 0,
    }


async def test_other_bank_can_delete_while_first_bank_is_paused(delete_race, request_context):
    race = delete_race
    first, second = race.engines
    paused, release = asyncio.Event(), asyncio.Event()
    original = first._delete_stale_observations_for_memories

    async def pause_after_delete(conn, bank_id, fact_ids):
        paused.set()
        await release.wait()
        return await original(conn, bank_id, fact_ids)

    first._delete_stale_observations_for_memories = pause_after_delete
    task = asyncio.create_task(first.delete_document("a", race.banks[0], request_context=request_context))
    try:
        await asyncio.wait_for(paused.wait(), 10)
        result = await asyncio.wait_for(second.delete_document("b", race.banks[1], request_context=request_context), 10)
        assert result["document_deleted"] == 1
        assert await second.get_document("a", race.banks[1], request_context=request_context) is not None
    finally:
        release.set()
        await asyncio.wait_for(task, 20)


@pytest.mark.parametrize("clear_observations", [False, True])
async def test_bank_cleanup_and_document_delete_use_same_lock_order(
    delete_race, request_context, monkeypatch, clear_observations
):
    race = delete_race
    first, second = race.engines
    paused, release = asyncio.Event(), asyncio.Event()
    execute = PostgresConnection.execute
    pause_query = (
        "DELETE FROM public.memory_units WHERE bank_id = $1 AND fact_type = 'observation'"
        if clear_observations
        else "DELETE FROM public.documents WHERE bank_id = $1"
    )

    async def pause_after_documents(conn, query, *args, **kwargs):
        result = await execute(conn, query, *args, **kwargs)
        if query == pause_query and args == (race.banks[0],):
            paused.set()
            await release.wait()
        return result

    monkeypatch.setattr(PostgresConnection, "execute", pause_after_documents)
    # Index cleanup is after commit; the assertion concerns the SQL deletion order.
    monkeypatch.setattr("hindsight_api.engine.retain.bank_utils.drop_bank_vector_indexes", AsyncMock())
    cleanup = first.clear_observations if clear_observations else first.delete_bank
    tasks = [asyncio.create_task(cleanup(race.banks[0], request_context=request_context))]
    try:
        await asyncio.wait_for(paused.wait(), 10)
        tasks.append(asyncio.create_task(second.delete_document("b", race.banks[0], request_context=request_context)))
        await wait_for_lock(race)
    finally:
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 20)
    if clear_observations:
        assert results[0] == {"deleted_count": 1}
        assert results[1] == {"document_deleted": 1, "memory_units_deleted": 1}
    else:
        assert results[0]["bank_deleted"] is True
        assert results[1] == {"document_deleted": 0, "memory_units_deleted": 0}

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
from hindsight_api.engine.retain.fact_storage import handle_document_tracking

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


async def test_reingest_and_delete_of_observation_sources_both_complete(delete_race, request_context):
    # Re-ingest does not take the bank lock (it would serialise retains), so the
    # sweep itself must not deadlock against a document delete's cascade.
    race = delete_race
    first, second = race.engines
    paused, release = asyncio.Event(), asyncio.Event()
    original = first._delete_stale_observations_for_memories
    sweeps = 0

    async def pause_after_cascade(conn, bank_id, fact_ids):
        # delete_document sweeps before and after its cascade; pause at the second,
        # while it holds document a's rows, the shared observation and b's source.
        nonlocal sweeps
        sweeps += 1
        if sweeps == 2:
            paused.set()
            await release.wait()
        return await original(conn, bank_id, fact_ids)

    async def reingest_b() -> None:
        backend = second._backend
        async with backend.acquire() as conn:
            async with conn.transaction():
                await handle_document_tracking(
                    conn, race.banks[0], "b", "replacement", is_first_batch=True, ops=backend.ops
                )

    first._delete_stale_observations_for_memories = pause_after_cascade
    tasks = [asyncio.create_task(first.delete_document("a", race.banks[0], request_context=request_context))]
    try:
        await asyncio.wait_for(paused.wait(), 10)
        tasks.append(asyncio.create_task(reingest_b()))
        await wait_for_lock(race)
    finally:
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 20)
    assert results == [{"document_deleted": 1, "memory_units_deleted": 1}, None]
    assert await first.get_document("a", race.banks[0], request_context=request_context) is None
    assert await first.get_document("b", race.banks[0], request_context=request_context) is not None


async def test_reingest_and_delete_of_linked_documents_both_complete(delete_race, request_context, monkeypatch):
    # Every a×b pair is linked both ways (temporal links are written as pairs), so the
    # delete's cascade and the re-ingest's cascade reach the same link rows from
    # opposite endpoints. The executor-chosen cascade order cannot be paused mid
    # statement, so hold each transaction at its first link-touching statement — the
    # cascade itself, or the ordered link delete that now precedes it — until the other
    # arrives, and start them together.
    race = delete_race
    first, second = race.engines
    bank = race.banks[0]
    cascades = (
        "DELETE FROM public.documents WHERE id = $1 AND bank_id = $2 RETURNING id",
        "DELETE FROM public.memory_units WHERE document_id = $1 AND bank_id = $2",
    )
    barrier = asyncio.Barrier(2)
    arrived: set[int] = set()
    execute, fetchval = PostgresConnection.execute, PostgresConnection.fetchval

    async def rendezvous(conn, query: str) -> None:
        if (query.strip() in cascades or "WITH matched_links" in query) and id(conn) not in arrived:
            arrived.add(id(conn))
            await asyncio.wait_for(barrier.wait(), 10)

    async def rendezvous_execute(conn, query, *args, **kwargs):
        await rendezvous(conn, query)
        return await execute(conn, query, *args, **kwargs)

    async def rendezvous_fetchval(conn, query, *args, **kwargs):
        await rendezvous(conn, query)
        return await fetchval(conn, query, *args, **kwargs)

    monkeypatch.setattr(PostgresConnection, "execute", rendezvous_execute)
    monkeypatch.setattr(PostgresConnection, "fetchval", rendezvous_fetchval)

    async def reingest_b() -> None:
        backend = second._backend
        async with backend.acquire() as conn:
            async with conn.transaction():
                await handle_document_tracking(conn, bank, "b", "replacement", is_first_batch=True, ops=backend.ops)

    for _ in range(3):
        await race.setup.execute("DELETE FROM memory_units WHERE bank_id=$1", bank)
        await race.setup.execute("DELETE FROM documents WHERE bank_id=$1", bank)
        units = {}
        for document_id in ("a", "b"):
            await race.setup.execute("INSERT INTO documents(id, bank_id) VALUES($1, $2)", document_id, bank)
            units[document_id] = [uuid.uuid4() for _ in range(100)]
            await race.setup.execute(
                "INSERT INTO memory_units(id, bank_id, document_id, text, fact_type) "
                "SELECT u, $2, $3, 'fact', 'world' FROM unnest($1::uuid[]) u",
                units[document_id],
                bank,
                document_id,
            )
        pairs = [(x, y) for x in units["a"] for y in units["b"]]
        await race.setup.execute(
            "INSERT INTO memory_links(from_unit_id, to_unit_id, link_type, weight, bank_id) "
            "SELECT f, t, 'temporal', 1.0, $3 FROM unnest($1::uuid[], $2::uuid[]) AS u(f, t)",
            [f for f, _ in pairs] + [t for _, t in pairs],
            [t for _, t in pairs] + [f for f, _ in pairs],
            bank,
        )
        results = await asyncio.wait_for(
            asyncio.gather(
                first.delete_document("a", bank, request_context=request_context),
                reingest_b(),
                return_exceptions=True,
            ),
            30,
        )
        assert results == [{"document_deleted": 1, "memory_units_deleted": 100}, None]


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

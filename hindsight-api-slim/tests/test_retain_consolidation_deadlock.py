"""Consolidation vs the bank's other memory writers must not deadlock (#5433).

Consolidation used to share-lock its batch, then lock the observation it rewrites, then stamp
the batch. A delta retain locks its document, sweeps (locks) the observations on its
re-chunked facts, then relabels the document's surviving facts — which can be the batch. On
plain Postgres the row locks alone closed the cycle (``delete``/``update``); on pgvectorscale
the DiskANN index's transaction-wide insert lock closed it even when only the stamp met the
relabel (``stamp``). Consolidation now takes its documents and then all its rows up front, in
the writers' order — so ``delete_document`` must take its document before its sweep too.

Real SQL on two connections through the real consolidation write transaction; only the LLM,
the recall and the point where the two transactions interleave are stubbed.
"""

import asyncio
import types
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from hindsight_api.engine.consolidation import consolidator as C
from hindsight_api.engine.db.postgresql import PostgreSQLBackend
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.response_models import MemoryFact
from hindsight_api.engine.retain.chunk_storage import delete_chunks_by_ids
from hindsight_api.engine.retain.fact_storage import update_memory_units_metadata_and_tags
from hindsight_api.engine.schema import fq_table

pytestmark = [pytest.mark.asyncio, pytest.mark.memory_backend_incompatible]

_WAITING = "SELECT wait_event_type = 'Lock' FROM pg_stat_activity WHERE pid=$1"
_IDLE_IN_TX = "SELECT state = 'idle in transaction' FROM pg_stat_activity WHERE pid=$1"
_DOC = "transcript"


async def _until(watch: asyncpg.Pool, pid: int, sql: str, timeout: float) -> None:
    """Wait until ``sql`` holds for backend ``pid``, or ``timeout`` (the other side is blocked)."""
    try:
        async with asyncio.timeout(timeout):
            while not await watch.fetchval(sql, pid):
                await asyncio.sleep(0.01)
    except TimeoutError:
        pass


@dataclass
class _Race:
    setup: asyncpg.Connection
    # Both transactions poll each other's state at once: a pool, not the one setup connection.
    watch: asyncpg.Pool
    bank: str
    consolidation_pool: PostgreSQLBackend
    writer_pool: PostgreSQLBackend
    consolidation_pid: int
    writer_pid: int
    other: uuid.UUID
    pending: uuid.UUID
    obs: uuid.UUID
    embedding: str
    memories: list[dict]


@asynccontextmanager
async def _race(db_url: str) -> AsyncIterator[_Race]:
    """One transcript of three chunks: `pending` is consolidation's batch; the observation
    stands on the last chunk's fact and on `other`, so a sweep of the last chunk requeues
    `other` — on pgvectorscale, an insert into the DiskANN index.

    Seeded with SQL because the race needs exact internal state no public call can set up: an
    observation over chosen sources, which facts are pending, chunk ids a delta retain replaces.
    """
    setup = await asyncpg.connect(db_url)
    bank = f"deadlock-5433-{uuid.uuid4().hex}"
    other, pending, old_unit, obs = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    dim = await setup.fetchval(
        "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'memory_units'::regclass AND attname = 'embedding'"
    )
    embedding = "[" + ",".join(["0.1"] * dim) + "]"
    await setup.execute("INSERT INTO banks(bank_id) VALUES($1)", bank)
    await setup.execute("INSERT INTO documents(id, bank_id) VALUES($1, $2)", _DOC, bank)
    for idx, (chunk, unit) in enumerate([("a", other), ("b", pending), ("last", old_unit)]):
        await setup.execute(
            "INSERT INTO chunks(chunk_id, document_id, bank_id, chunk_index, chunk_text) VALUES($1,$2,$3,$4,'t')",
            f"{bank}-{chunk}",
            _DOC,
            bank,
            idx,
        )
        await setup.execute(
            "INSERT INTO memory_units(id, bank_id, document_id, chunk_id, text, fact_type, embedding) "
            "VALUES($1,$2,$3,$4,'fact','world',$5::vector)",
            unit,
            bank,
            _DOC,
            f"{bank}-{chunk}",
            embedding,
        )
    await setup.execute("UPDATE memory_units SET consolidated_at = NOW() WHERE id = ANY($1::uuid[])", [old_unit, other])
    await setup.execute(
        "INSERT INTO memory_units(id, bank_id, text, fact_type, source_memory_ids) VALUES($1,$2,'obs','observation',$3)",
        obs,
        bank,
        [old_unit, other],
    )
    # On pgvectorscale, the bank's partial DiskANN index (what vector-index maintenance builds).
    diskann = await setup.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vectorscale'")
    if diskann:
        await setup.execute(
            f'CREATE INDEX "idx_5433_{bank[-12:]}" ON memory_units USING diskann (embedding vector_cosine_ops) '
            f"WITH (num_neighbors = 50) WHERE fact_type = 'world' AND bank_id = '{bank}'"
        )
    row = await setup.fetchrow("SELECT updated_at FROM memory_units WHERE id=$1", pending)
    watch = await asyncpg.create_pool(db_url, min_size=1, max_size=3)
    backends: list[PostgreSQLBackend] = []
    try:
        pids = []
        for _ in range(2):
            b = PostgreSQLBackend()
            await b.initialize(db_url, min_size=1, max_size=1, command_timeout=15)
            backends.append(b)
            async with b.acquire() as c:
                pids.append(await c.fetchval("SELECT pg_backend_pid()"))
        yield _Race(
            setup=setup,
            watch=watch,
            bank=bank,
            consolidation_pool=backends[0],
            writer_pool=backends[1],
            consolidation_pid=pids[0],
            writer_pid=pids[1],
            other=other,
            pending=pending,
            obs=obs,
            embedding=embedding,
            memories=[{"id": str(pending), "text": "fact", "tags": [], "updated_at": row["updated_at"]}],
        )
    finally:
        for b in backends:
            await b.shutdown()
        await watch.close()
        await setup.execute("DELETE FROM memory_units WHERE bank_id=$1", bank)
        await setup.execute("DELETE FROM chunks WHERE bank_id=$1", bank)
        await setup.execute("DELETE FROM documents WHERE bank_id=$1", bank)
        await setup.execute("DELETE FROM banks WHERE bank_id=$1", bank)
        if diskann:
            await setup.execute(f'DROP INDEX IF EXISTS "idx_5433_{bank[-12:]}"')
        await setup.close()


def _llm_result(race: _Race, action: str) -> "C._BatchLLMResult":
    return {
        "delete": C._BatchLLMResult(deletes=[C._DeleteAction(observation_id=str(race.obs))]),
        "update": C._BatchLLMResult(
            updates=[C._UpdateAction(observation_id=str(race.obs), text="obs v2", source_fact_ids=[str(race.pending)])]
        ),
        "stamp": C._BatchLLMResult(),
    }[action]


async def _consolidate(
    race: _Race,
    action: str,
    *,
    pause: Callable[[], Awaitable[None]] | None = None,
    before_locks: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """``_process_memory_batch`` for `pending`, its LLM answering ``action``. Inside the write
    transaction, ``before_locks`` runs before it takes any lock, ``pause`` after its first reads."""
    real_changed_since = C._sources_changed_since_read
    store = get_memories()
    real_lock_set = store.lock_consolidation_write_set

    async def changed_since_then_pause(conn, bank_id, mems):
        result = await real_changed_since(conn, bank_id, mems)
        if pause is not None:
            await pause()
        return result

    async def wait_then_lock_set(**kwargs):
        if before_locks is not None:
            await before_locks()
        await real_lock_set(**kwargs)

    obs_fact = MemoryFact(id=str(race.obs), text="obs", fact_type="observation", tags=[])
    with (
        patch.object(C, "_sources_changed_since_read", new=changed_since_then_pause),
        patch.object(store, "lock_consolidation_write_set", new=wait_then_lock_set),
        patch.object(
            C,
            "_find_related_observations",
            new=AsyncMock(return_value=types.SimpleNamespace(results=[obs_fact], source_facts={})),
        ),
        patch.object(C, "_consolidate_batch_with_llm", new=AsyncMock(return_value=_llm_result(race, action))),
        patch.object(C, "_effective_scope_limit", return_value=-1),
        patch.object(C, "_config_for_scope", side_effect=lambda config, _tags: config),
        patch.object(C, "_dedup_active", return_value=False),
        patch.object(C, "triage_model", return_value=None),
        patch.object(
            C,
            "_embed_observation_texts",
            new=AsyncMock(side_effect=lambda _engine, texts, _perf=None: {t: race.embedding for t in texts}),
        ),
        # The recalls' query vectors are embedded up front in one call (#5769); the recall
        # itself is stubbed above, so any vector of the right shape will do (#5797).
        patch.object(
            C.embedding_utils,
            "generate_embeddings_batch",
            new=AsyncMock(side_effect=lambda _embeddings, texts, **_kw: [race.embedding for _ in texts]),
        ),
    ):
        await C._process_memory_batch(
            pool=race.consolidation_pool,
            memory_engine=types.SimpleNamespace(_backend=race.consolidation_pool, embeddings=object()),
            llm_config=object(),
            bank_id=race.bank,
            memories=race.memories,
            request_context=object(),
            config=types.SimpleNamespace(),
            mark_consolidated_ids=[race.pending],
        )


@pytest.mark.parametrize("action", ["delete", "update", "stamp"])
async def test_delta_retain_and_consolidation_do_not_deadlock(pg0_db_url, action):
    async with _race(pg0_db_url) as race:
        swept = asyncio.Event()

        async def let_retain_ahead() -> None:
            # Past its sweep, or (stamp) blocked in its relabel. Bounded: a retain waiting on
            # this transaction never gets there.
            if action == "stamp":
                await _until(race.watch, race.writer_pid, _WAITING, 2)
            else:
                try:
                    await asyncio.wait_for(swept.wait(), 2)
                except TimeoutError:
                    pass

        async def delta_retain() -> None:
            # The delta branch of the retain orchestrator, in its order: the document row first.
            async with race.writer_pool.acquire() as conn, conn.transaction():
                await get_memories().lock_document_hash(
                    conn=conn, fq_table=fq_table, bank_id=race.bank, document_id=_DOC
                )
                await delete_chunks_by_ids(conn, [f"{race.bank}-last"], race.bank, ops=race.writer_pool.ops)
                swept.set()
                if action != "stamp":
                    await _until(race.watch, race.consolidation_pid, _WAITING, 2)
                await update_memory_units_metadata_and_tags(
                    conn, race.bank, _DOC, ["session:2"], {"turns": 2}, ops=race.writer_pool.ops
                )

        task_c = asyncio.create_task(_consolidate(race, action, pause=let_retain_ahead))
        await _until(race.watch, race.consolidation_pid, _IDLE_IN_TX, 10)
        task_r = asyncio.create_task(delta_retain())
        results = await asyncio.gather(task_c, task_r, return_exceptions=True)
        assert [r for r in results if isinstance(r, BaseException)] == []

        async with race.writer_pool.acquire() as conn:
            stored = await get_memories().get_memories(
                conn=conn, fq_table=fq_table, bank_id=race.bank, unit_ids=[str(race.other), str(race.pending)]
            )
        by_id = {m.unit_id: m for m in stored}
        assert {uid: m.tags for uid, m in by_id.items()} == {
            str(race.other): ["session:2"],
            str(race.pending): ["session:2"],
        }
        # The retain committed second, and its retag requeues what consolidation just stamped.
        assert by_id[str(race.pending)].consolidated_at is None


def _engine(backend: PostgreSQLBackend) -> MemoryEngine:
    """Just enough engine for ``delete_document``: real SQL, no post-commit work."""
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._initialized = True
    engine._backend = backend
    engine._operation_validator = None
    engine._tenant_extension = None
    engine._authenticate_tenant = AsyncMock(return_value="public")
    engine._bank_stats_cache = types.SimpleNamespace(invalidate=AsyncMock())
    engine._config_resolver = types.SimpleNamespace(
        resolve_full_config=AsyncMock(return_value=types.SimpleNamespace(enable_auto_consolidation=False))
    )
    engine._submit_refreshes_for_retracted_grounding = AsyncMock()
    engine.submit_async_graph_maintenance = AsyncMock()
    engine._submit_vector_index_maintenance_quietly = AsyncMock()
    return engine


async def test_delete_document_and_consolidation_do_not_deadlock(pg0_db_url, request_context):
    # The delete sweeps the document's facts first and deletes its row last; consolidation now
    # takes that row before the facts. Without the delete taking its row up front, consolidation
    # holds the row and waits on the swept facts while the delete waits on the row.
    async with _race(pg0_db_url) as race:
        engine = _engine(race.writer_pool)
        swept = asyncio.Event()
        real_sweep = engine._delete_stale_observations_for_memories

        async def sweep_then_let_consolidation_in(conn, bank_id, fact_ids):
            result = await real_sweep(conn, bank_id, fact_ids)
            if not swept.is_set():
                swept.set()
                await _until(race.watch, race.consolidation_pid, _WAITING, 2)
            return result

        engine._delete_stale_observations_for_memories = sweep_then_let_consolidation_in
        # Consolidation past its unlocked preflight, in its write transaction, before any lock.
        in_write = asyncio.Event()

        async def hold_until_swept() -> None:
            in_write.set()
            await asyncio.wait_for(swept.wait(), 10)

        task_c = asyncio.create_task(_consolidate(race, "update", before_locks=hold_until_swept))
        async with asyncio.timeout(10):
            await in_write.wait()
        task_d = asyncio.create_task(engine.delete_document(_DOC, race.bank, request_context=request_context))
        results = await asyncio.gather(task_d, task_c, return_exceptions=True)
        assert [r for r in results if isinstance(r, BaseException)] == []
        assert results[0] == {"document_deleted": 1, "memory_units_deleted": 3}

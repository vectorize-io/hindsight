"""Per-bank vector-index DDL is serialized per table within a process.

Concurrent index DDL on the shared ``memory_units`` table deadlocks by design:
DROP INDEX CONCURRENTLY holds ShareUpdateExclusive while waiting out every
other session whose snapshot could still see the index — including other
sessions' queued index DDL. CI's end-of-run teardown (all xdist workers
deleting their banks at once) forms exactly that cycle, and the delete path's
default retry budget (~2.4s) could not outlast the storm (run 31195108586,
test-api 3/3). Advisory locks are banned in this codebase (poolers), so the
fix is an in-process asyncio lock on ``PostgreSQLOps`` plus a much larger
jittered retry budget for the cross-process residue.

Two layers are proven here:

* unit (fake conn): create and drop DDL for one table never interleave, the
  create and drop sides contend on the same lock, and the lock is released
  when a statement raises;
* integration: a many-bank concurrent ``delete_bank`` storm — the CI failure
  shape — completes without ``DeadlockDetectedError``.
"""

import asyncio
import uuid

import pytest

from hindsight_api.engine.db.ops_postgresql import PostgreSQLOps
from hindsight_api.engine.retain.bank_utils import _BANK_INDEX_FACT_TYPES, _bank_index_name, _vector_index_clause

_SCHEMA = "public"
_INDEX_CLAUSE = "USING hnsw (embedding vector_cosine_ops)"


class _OverlapTrackingConn:
    """Fake DatabaseConnection asserting no two DDL statements ever overlap.

    Each execute parks on the event loop long enough that unserialized callers
    would interleave, and records the highest number of in-flight statements.
    """

    def __init__(self, fail_on: str | None = None):
        self.calls: list[str] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._fail_on = fail_on

    async def execute(self, query, *args, **kwargs):
        self.calls.append(query)
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(0.01)
            if self._fail_on and self._fail_on in query:
                raise RuntimeError(f"boom on: {query}")
        finally:
            self._in_flight -= 1
        return "OK"


async def test_concurrent_create_and_drop_on_one_table_serialize():
    ops = PostgreSQLOps()
    conn = _OverlapTrackingConn()
    table = f"{_SCHEMA}.memory_units"

    await asyncio.gather(
        ops.drop_bank_vector_indexes(conn, _SCHEMA, uuid.uuid4().hex, _BANK_INDEX_FACT_TYPES),
        ops.drop_bank_vector_indexes(conn, _SCHEMA, uuid.uuid4().hex, _BANK_INDEX_FACT_TYPES),
        # The drop side reconstructs the create side's fq-table key from
        # `schema`, so a create must queue behind the drops too.
        ops.create_bank_vector_indexes(conn, table, "bank-1", uuid.uuid4().hex, _INDEX_CLAUSE, _BANK_INDEX_FACT_TYPES),
    )

    assert len(conn.calls) == 3 * len(_BANK_INDEX_FACT_TYPES)
    assert conn.max_in_flight == 1, "vector-index DDL statements overlapped despite the per-table lock"


async def test_different_tables_do_not_contend():
    ops = PostgreSQLOps()
    assert ops._index_ddl_lock("a.memory_units") is ops._index_ddl_lock("a.memory_units")
    assert ops._index_ddl_lock("a.memory_units") is not ops._index_ddl_lock("b.memory_units")


async def test_lock_released_when_ddl_raises():
    ops = PostgreSQLOps()
    conn = _OverlapTrackingConn(fail_on="DROP INDEX CONCURRENTLY")
    with pytest.raises(RuntimeError):
        await ops.drop_bank_vector_indexes(conn, _SCHEMA, uuid.uuid4().hex, _BANK_INDEX_FACT_TYPES)

    # A subsequent create must not hang on a lock the failed drop never released.
    await asyncio.wait_for(
        ops.create_bank_vector_indexes(
            conn, f"{_SCHEMA}.memory_units", "bank-1", uuid.uuid4().hex, _INDEX_CLAUSE, _BANK_INDEX_FACT_TYPES
        ),
        timeout=2.0,
    )


async def test_concurrent_bank_delete_storm_does_not_deadlock(memory, request_context):
    """The CI failure shape: every worker tears down its banks at once.

    Eight banks (24 partial indexes) dropped concurrently previously wedged
    into a DROP INDEX CONCURRENTLY wait cycle; serialized behind the per-table
    lock the storm must complete without DeadlockDetectedError.
    """
    if _vector_index_clause() is None:
        pytest.skip("backend does not use per-bank vector indexes")

    bank_ids = [f"test-ddl-lock-{uuid.uuid4().hex[:8]}" for _ in range(8)]
    for bank_id in bank_ids:
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        internal_ids = {
            bank_id: str(await conn.fetchval("SELECT internal_id FROM banks WHERE bank_id = $1", bank_id))
            for bank_id in bank_ids
        }

    await asyncio.gather(*(memory.delete_bank(bank_id, request_context=request_context) for bank_id in bank_ids))

    async with backend.acquire() as conn:
        for bank_id in bank_ids:
            for ft in _BANK_INDEX_FACT_TYPES:
                idx = _bank_index_name(ft, internal_ids[bank_id])
                assert not await conn.fetchval(
                    "SELECT 1 FROM pg_indexes WHERE schemaname = $1 AND indexname = $2", _SCHEMA, idx
                ), f"index {idx} survived delete_bank"

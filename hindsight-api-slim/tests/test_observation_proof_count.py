"""A new observation's proof_count is its number of distinct live sources (#4955).

Consolidation's create path used to store a literal 1, so an observation built from
several facts kept proof_count=1 until an update recounted it, and ranked lower in
recall than equally supported ones.
"""

import importlib
import uuid
from types import SimpleNamespace

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.memory_engine import MemoryEngine
from tests.consolidation_actions import create_observation
from tests.test_observation_invalidation import _ensure_bank, _get_memory, _insert_memory


@pytest.mark.asyncio
async def test_created_observation_counts_each_live_source_once(memory: MemoryEngine, request_context: RequestContext):
    bank_id = f"test-create-proof-count-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        a = await _insert_memory(memory, conn, bank_id, "Alice changed the timeout in app.yaml.")
        b = await _insert_memory(memory, conn, bank_id, "Alice changed the retry count in app.yaml.")
        c = await _insert_memory(memory, conn, bank_id, "Alice changed the log level in app.yaml.")
    dead = uuid.uuid4()  # stands in for a source deleted concurrently

    result = await create_observation(
        pool=backend,
        memory_engine=memory,
        bank_id=bank_id,
        source_memory_ids=[a, b, a, dead, c],
        observation_text="Alice keeps tuning app.yaml.",
    )

    assert result["action"] == "created"
    async with acquire_with_retry(backend) as conn:
        stored = await _get_memory(conn, bank_id, result["observation_id"])
    assert [str(s) for s in stored.source_memory_ids] == [str(a), str(b), str(c)]
    assert stored.proof_count == 3

    await memory.delete_bank(bank_id, request_context=request_context)


class _Rollback(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_migration_recounts_undercounted_observations(
    memory: MemoryEngine, request_context: RequestContext, monkeypatch: pytest.MonkeyPatch
):
    if memory._database_backend_type != "postgresql":
        pytest.skip("seeds memory_units rows with PostgreSQL SQL")
    migration = importlib.import_module("hindsight_api.alembic.versions.f3a9c1e7b5d2_repair_observation_proof_count")

    bank_id = f"test-repair-proof-count-{uuid.uuid4().hex[:8]}"
    await _ensure_bank(memory, bank_id, request_context)
    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        a = await _insert_memory(memory, conn, bank_id, "Bob moved to Berlin.")
        b = await _insert_memory(memory, conn, bank_id, "Bob works in Berlin.")
    statements: list[str] = []
    monkeypatch.setattr(migration, "op", SimpleNamespace(execute=statements.append))
    monkeypatch.setattr(migration, "context", SimpleNamespace(config=SimpleNamespace(get_main_option=lambda _: None)))
    migration._pg_upgrade()
    assert len(statements) == 1

    stuck, sourceless = uuid.uuid4(), uuid.uuid4()
    # Rolled back at the end: the repair is table-wide, and other tests' rows must not change.
    with pytest.raises(_Rollback):
        async with acquire_with_retry(backend) as conn, conn.transaction():
            # Raw INSERT on purpose: it forges the undercounted rows the old create path left
            # behind, which no engine write produces any more.
            await conn.executemany(
                "INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, source_memory_ids, proof_count) "
                "VALUES ($1, $2, $3, 'observation', NOW(), $4, 1)",
                [
                    (stuck, bank_id, "Bob lives and works in Berlin.", [a, b, a]),
                    (sourceless, bank_id, "Bob is a person.", []),
                ],
            )
            await conn.execute(statements[0])
            assert (await _get_memory(conn, bank_id, stuck)).proof_count == 2
            assert (await _get_memory(conn, bank_id, sourceless)).proof_count == 1
            raise _Rollback

    await memory.delete_bank(bank_id, request_context=request_context)

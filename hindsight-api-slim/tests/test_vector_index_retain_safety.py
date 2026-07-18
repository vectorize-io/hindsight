from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine


@pytest.mark.asyncio
async def test_retain_safety_check_is_ttl_gated_and_only_wakes_maintenance(monkeypatch) -> None:
    import hindsight_api.engine.memory_engine as memory_engine_mod
    import hindsight_api.engine.retain.bank_utils as bank_utils
    import hindsight_api.engine.vector_index_reconcile as reconcile_mod

    queries = 0
    wakeups = 0

    @asynccontextmanager
    async def fake_acquire_with_retry(_backend, **_kwargs):
        yield SimpleNamespace()

    class Maintenance:
        def request_vector_index_reconcile(self):
            nonlocal wakeups
            wakeups += 1

    async def unhealthy(_conn, schema, bank_id):
        nonlocal queries
        queries += 1
        assert schema == "tenant_a"
        assert bank_id == "restored-bank"
        return False

    engine = SimpleNamespace(
        _backend=None,
        _database_backend_type="postgresql",
        _maintenance_loop=Maintenance(),
        _vector_index_check_last={},
    )
    cfg = SimpleNamespace(vector_index_reconcile_interval_seconds=3600)
    monkeypatch.setattr(memory_engine_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(memory_engine_mod, "get_current_schema", lambda: "tenant_a")
    monkeypatch.setattr(memory_engine_mod, "acquire_with_retry", fake_acquire_with_retry)
    monkeypatch.setattr(bank_utils, "_vector_index_clause", lambda: "USING hnsw (embedding vector_cosine_ops)")
    monkeypatch.setattr(reconcile_mod, "bank_vector_indexes_healthy", unhealthy)

    await MemoryEngine._maybe_request_vector_index_reconcile(engine, "restored-bank")
    await MemoryEngine._maybe_request_vector_index_reconcile(engine, "restored-bank")

    assert queries == 1
    assert wakeups == 1

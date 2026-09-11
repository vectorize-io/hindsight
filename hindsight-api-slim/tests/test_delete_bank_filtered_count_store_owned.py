"""``delete_bank(fact_type=...)`` must count what it deletes where the memories live.

For a store-owned bank the SQL ``memory_units`` table is intentionally empty; the rows
are removed through ``store.delete_where``. The filtered branch used to count the SQL
table anyway and reported ``memory_units_deleted: 0`` (#4307 review), which the
``DELETE /banks/{bank}/memories?type=...`` endpoint now exposes as ``deleted_count``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import hindsight_api.engine.memories as memories_mod
from hindsight_api import RequestContext
from hindsight_api.engine import memory_engine as engine_mod
from hindsight_api.engine.memory_engine import MemoryEngine


class _StoreOwnedMemories:
    """A store that keeps memories outside SQL and records the delete it is asked for."""

    def __init__(self, counts: dict[str, int]):
        self.counts = counts
        self.deleted: list = []

    def store_owned_for(self, bank_id: str) -> bool:
        return True

    async def scan_memories(self, *, conn, fq_table, bank_id, fact_types, limit):
        n = sum(self.counts.get(t, 0) for t in fact_types)
        return SimpleNamespace(memories=[SimpleNamespace(unit_id=f"u{i}") for i in range(n)])

    async def count_memories(self, *, conn, fq_table, bank_id):
        return dict(self.counts)

    async def delete_where(self, bank_id, predicate):
        self.deleted.append((bank_id, predicate))


class _FakeConn:
    """Answers every SQL statement with 'nothing there', like the empty SQL side of a store-owned bank."""

    def __init__(self):
        self.executed: list[str] = []

    @asynccontextmanager
    async def transaction(self):
        yield self

    async def execute(self, sql, *args):
        self.executed.append(sql)

    async def fetch(self, sql, *args):
        return []

    async def fetchval(self, sql, *args):
        return None


def _stub_engine() -> MemoryEngine:
    engine = object.__new__(MemoryEngine)
    engine._authenticate_tenant = AsyncMock()
    engine._operation_validator = None
    engine._get_backend = AsyncMock(return_value=MagicMock())
    engine._tenant_extension = None
    engine._bank_stats_cache = MagicMock(invalidate=AsyncMock())
    engine._delete_stale_observations_for_memories = AsyncMock(return_value=0)
    return engine


@pytest.mark.asyncio
@pytest.mark.parametrize("fact_type", ["world", "observation"])
async def test_filtered_delete_counts_store_owned_memories(monkeypatch, fact_type):
    store = _StoreOwnedMemories({"world": 3, "experience": 2, "observation": 4})
    monkeypatch.setattr(memories_mod, "get_memories", lambda: store)
    conn = _FakeConn()

    @asynccontextmanager
    async def _acquire(_backend, *args, **kwargs):
        yield conn

    monkeypatch.setattr(engine_mod, "acquire_with_retry", _acquire)
    engine = _stub_engine()

    result = await engine.delete_bank(
        "bank-x", fact_type=fact_type, delete_bank_profile=False, request_context=RequestContext(api_key="k")
    )

    assert result["memory_units_deleted"] == store.counts[fact_type]
    assert result["entities_deleted"] == 0
    assert len(store.deleted) == 1
    bank_id, predicate = store.deleted[0]
    assert bank_id == "bank-x"
    assert list(predicate.fact_types) == [fact_type]

"""``_store_read_conn`` takes a pool connection only for a bank whose store reads SQL.

A store-owned bank answers these reads by itself, so a pooled connection would be held for the
whole remote call and used for nothing. The property asserted is that the pool is never touched
for such a bank, not merely that ``None`` comes out: a helper that acquired and then yielded
``None`` would still hold the slot.

Runs via: uv run pytest tests/test_store_read_conn.py -v
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest

import hindsight_api.engine.memories as memories_mod
import hindsight_api.engine.memory_engine as engine_mod
from hindsight_api.engine.memory_engine import MemoryEngine

# What the faked pool hands out, so a test can tell "the real connection" from any other value.
POOLED_CONN = object()


class _Store:
    def __init__(self, owned: bool):
        self.owned = owned

    def store_owned_for(self, bank_id: str) -> bool:
        return self.owned


@dataclass
class _Yielded:
    """What one ``_store_read_conn`` entry produced, and what it cost the pool."""

    conn: Any = None
    # One entry per acquire_with_retry call — empty means the pool was never touched.
    acquired: list[object] = field(default_factory=list)


async def _conn_for(monkeypatch, *, owned: bool) -> _Yielded:
    result = _Yielded()

    @asynccontextmanager
    async def fake_acquire(backend):
        result.acquired.append(backend)
        yield POOLED_CONN

    monkeypatch.setattr(engine_mod, "acquire_with_retry", fake_acquire)
    monkeypatch.setattr(memories_mod, "get_memories", lambda: _Store(owned))
    engine = object.__new__(MemoryEngine)
    engine._initialized = True
    engine._backend = "pool"
    async with engine._store_read_conn("b") as conn:
        result.conn = conn
    return result


@pytest.mark.asyncio
async def test_store_owned_bank_takes_no_connection(monkeypatch):
    yielded = await _conn_for(monkeypatch, owned=True)
    assert yielded.conn is None
    assert yielded.acquired == [], "the pool must not be touched for a store-owned bank"


@pytest.mark.asyncio
async def test_sql_bank_still_gets_a_pooled_connection(monkeypatch):
    yielded = await _conn_for(monkeypatch, owned=False)
    assert yielded.conn is POOLED_CONN
    assert yielded.acquired == ["pool"]

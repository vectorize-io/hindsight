"""local(tars) 2026-07-16: recall access tracking (Vollaudit G-2).

``access_count`` existed since the initial schema but had no writer anywhere
(upstream's ``access_count_update`` task type is comment/test-only). These
tests cover the fire-and-forget bump on the recall path.

2026-07-23 (TARS-Review M5): the bump must go through the BACKEND abstraction
(``_get_backend()`` → ``DatabaseConnection.execute``), never the raw pool —
on Oracle the raw pool yields a raw oracledb connection whose ``execute``
skips the PG→Oracle rewrite pipeline, so ``ANY($1::uuid[])`` silently no-ops.
The tests below pin both the routing and the dialect translation itself.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine


def _bare_engine() -> MemoryEngine:
    eng = MemoryEngine.__new__(MemoryEngine)
    eng._access_bump_tasks = set()
    return eng


IDS = [
    "11111111-1111-1111-1111-111111111111",
    "22222222-2222-2222-2222-222222222222",
]


@pytest.mark.asyncio
async def test_bump_executes_single_update_via_backend(monkeypatch):
    """The bump acquires through the backend abstraction, not a raw pool."""
    eng = _bare_engine()
    conn = AsyncMock()
    fake_backend = object()

    async def fake_get_backend():
        return fake_backend

    @asynccontextmanager
    async def fake_acquire(target):
        assert target is fake_backend, "must pass the BACKEND to acquire_with_retry"
        yield conn

    eng._get_backend = fake_get_backend
    monkeypatch.setattr(
        "hindsight_api.engine.memory_engine.acquire_with_retry", fake_acquire
    )
    await eng._bump_access_counts(IDS)
    assert conn.execute.await_count == 1
    sql, passed_ids = conn.execute.await_args.args
    assert "access_count = access_count + 1" in sql
    assert passed_ids == IDS


@pytest.mark.asyncio
async def test_bump_never_touches_raw_pool(monkeypatch):
    """Regression (TARS M5): _get_pool() bypassed OracleConnection.execute."""
    eng = _bare_engine()
    conn = AsyncMock()

    async def fail_get_pool():  # pragma: no cover - failing is the assertion
        raise AssertionError("_bump_access_counts must not use the raw pool")

    async def fake_get_backend():
        return object()

    @asynccontextmanager
    async def fake_acquire(target):
        yield conn

    eng._get_pool = fail_get_pool
    eng._get_backend = fake_get_backend
    monkeypatch.setattr(
        "hindsight_api.engine.memory_engine.acquire_with_retry", fake_acquire
    )
    await eng._bump_access_counts(IDS)
    assert conn.execute.await_count == 1


def test_bump_sql_translates_for_oracle():
    """The exact bump SQL must survive the PG→Oracle rewrite pipeline.

    Mirrors the read-only rewriter smoke from the review: the ANY-cast form
    becomes an IN list with one named bind per element — the shape a raw
    oracledb connection would never produce.
    """
    from hindsight_api.engine.db.oracle import (
        OracleConnection,
        _rewrite_pg_to_oracle,
    )
    from hindsight_api.engine.memory_engine import fq_table

    sql = (
        f"UPDATE {fq_table('memory_units')} "
        "SET access_count = access_count + 1 "
        "WHERE id = ANY($1::uuid[])"
    )
    rewritten, _ignore_dup, _ret = _rewrite_pg_to_oracle(sql)
    assert "ANY(" not in rewritten.upper().replace(" ", ""), rewritten
    expanded, params = OracleConnection._expand_any_lists(rewritten, {"1": IDS})
    assert "IN (" in expanded, expanded
    bind_names = [k for k in params if k != "1"]
    assert len(bind_names) == len(IDS), (expanded, params)
    assert all(expanded.count(f":{name}") == 1 for name in bind_names)


@pytest.mark.asyncio
async def test_bump_failure_never_raises():
    eng = _bare_engine()

    async def broken_backend():
        raise RuntimeError("backend down")

    eng._get_backend = broken_backend
    await eng._bump_access_counts(IDS[:1])


@pytest.mark.asyncio
async def test_schedule_holds_reference_and_cleans_up():
    eng = _bare_engine()
    done = asyncio.Event()

    async def fake_bump(ids):
        done.set()

    eng._bump_access_counts = fake_bump
    eng._schedule_access_bump(IDS[:1])
    assert len(eng._access_bump_tasks) == 1
    await asyncio.wait_for(done.wait(), timeout=2)
    await asyncio.sleep(0)  # let done_callback run
    assert len(eng._access_bump_tasks) == 0


@pytest.mark.asyncio
async def test_schedule_noop_on_empty_ids():
    eng = _bare_engine()
    eng._schedule_access_bump([])
    assert eng._access_bump_tasks == set()

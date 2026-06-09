"""Unit tests for `BankStatsCache` — TTL, eviction, and concurrent coalescing.

These tests don't touch the database; they exercise the cache wrapper
directly so the semantics are checked in isolation from `MemoryEngine`.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from hindsight_api.engine.bank_stats_cache import BankStatsCache


def make_loader(return_value: dict[str, Any]) -> tuple[Any, list[int]]:
    """Returns (loader_fn, call_count_list). `call_count_list[0]` is the count."""
    calls = [0]

    async def loader() -> dict[str, Any]:
        calls[0] += 1
        return return_value

    return loader, calls


@pytest.mark.asyncio
async def test_cache_disabled_passes_through() -> None:
    cache = BankStatsCache(ttl_seconds=0, max_entries=100)
    loader, calls = make_loader({"v": 1})

    for _ in range(3):
        result = await cache.get_or_load("schema", "bank", loader)
        assert result == {"v": 1}
    assert calls[0] == 3


@pytest.mark.asyncio
async def test_cache_serves_hits_within_ttl() -> None:
    cache = BankStatsCache(ttl_seconds=60, max_entries=100)
    loader, calls = make_loader({"v": 1})

    first = await cache.get_or_load("schema", "bank", loader)
    second = await cache.get_or_load("schema", "bank", loader)
    assert first == second == {"v": 1}
    assert calls[0] == 1


@pytest.mark.asyncio
async def test_cache_reloads_after_ttl_expires(monkeypatch) -> None:
    cache = BankStatsCache(ttl_seconds=0.05, max_entries=100)
    loader, calls = make_loader({"v": 1})

    fake_time = [1000.0]
    monkeypatch.setattr(cache, "_now", lambda: fake_time[0])

    await cache.get_or_load("schema", "bank", loader)
    fake_time[0] += 0.1  # advance past TTL
    await cache.get_or_load("schema", "bank", loader)
    assert calls[0] == 2


@pytest.mark.asyncio
async def test_cache_isolates_by_schema_and_bank() -> None:
    cache = BankStatsCache(ttl_seconds=60, max_entries=100)
    loader, calls = make_loader({"v": 1})

    await cache.get_or_load("schema_a", "bank", loader)
    await cache.get_or_load("schema_b", "bank", loader)
    await cache.get_or_load("schema_a", "other", loader)
    # 3 distinct keys → 3 loader calls.
    assert calls[0] == 3


@pytest.mark.asyncio
async def test_concurrent_misses_are_coalesced() -> None:
    """6 concurrent callers on the same cold key must trigger exactly one loader."""
    cache = BankStatsCache(ttl_seconds=60, max_entries=100)
    calls = [0]
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_loader() -> dict[str, Any]:
        calls[0] += 1
        started.set()
        await release.wait()
        return {"v": calls[0]}

    tasks = [asyncio.create_task(cache.get_or_load("schema", "bank", slow_loader)) for _ in range(6)]
    await started.wait()
    # All other tasks should now be queued behind the in-flight loader.
    release.set()
    results = await asyncio.gather(*tasks)

    assert calls[0] == 1
    assert all(r == {"v": 1} for r in results)


@pytest.mark.asyncio
async def test_loader_exception_does_not_poison_cache() -> None:
    cache = BankStatsCache(ttl_seconds=60, max_entries=100)
    calls = [0]

    async def flaky_loader() -> dict[str, Any]:
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("boom")
        return {"v": calls[0]}

    with pytest.raises(RuntimeError, match="boom"):
        await cache.get_or_load("schema", "bank", flaky_loader)

    # Second call should still attempt the loader (cache wasn't populated).
    result = await cache.get_or_load("schema", "bank", flaky_loader)
    assert result == {"v": 2}
    assert calls[0] == 2


@pytest.mark.asyncio
async def test_concurrent_loader_exception_propagates_to_waiters() -> None:
    cache = BankStatsCache(ttl_seconds=60, max_entries=100)
    started = asyncio.Event()
    release = asyncio.Event()

    async def failing_loader() -> dict[str, Any]:
        started.set()
        await release.wait()
        raise RuntimeError("loader failed")

    tasks = [asyncio.create_task(cache.get_or_load("schema", "bank", failing_loader)) for _ in range(3)]
    await started.wait()
    release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(r, RuntimeError) for r in results)


@pytest.mark.asyncio
async def test_lru_eviction_respects_max_entries() -> None:
    cache = BankStatsCache(ttl_seconds=60, max_entries=2)

    async def loader_for(value: int):
        async def _loader() -> dict[str, Any]:
            return {"v": value}

        return _loader

    await cache.get_or_load("s", "a", await loader_for(1))
    await cache.get_or_load("s", "b", await loader_for(2))
    # Touch "a" so it's most-recently-used.
    await cache.get_or_load("s", "a", await loader_for(99))
    # Insert "c" — should evict "b" (the LRU), not "a".
    await cache.get_or_load("s", "c", await loader_for(3))

    # "a" is still cached (loader for "a" with value=99 must NOT be called again).
    miss_check_calls = [0]

    async def should_not_run() -> dict[str, Any]:
        miss_check_calls[0] += 1
        return {"v": -1}

    cached_a = await cache.get_or_load("s", "a", should_not_run)
    assert cached_a == {"v": 1}
    assert miss_check_calls[0] == 0

    # "b" was evicted; the loader must run on the next get.
    new_b_calls = [0]

    async def new_b() -> dict[str, Any]:
        new_b_calls[0] += 1
        return {"v": 200}

    fetched_b = await cache.get_or_load("s", "b", new_b)
    assert fetched_b == {"v": 200}
    assert new_b_calls[0] == 1


@pytest.mark.asyncio
async def test_invalidate_drops_entry() -> None:
    cache = BankStatsCache(ttl_seconds=60, max_entries=100)
    loader, calls = make_loader({"v": 1})

    await cache.get_or_load("schema", "bank", loader)
    await cache.invalidate("schema", "bank")
    await cache.get_or_load("schema", "bank", loader)
    assert calls[0] == 2


@pytest.mark.asyncio
async def test_clear_drops_all_entries() -> None:
    cache = BankStatsCache(ttl_seconds=60, max_entries=100)
    loader, calls = make_loader({"v": 1})

    await cache.get_or_load("s", "a", loader)
    await cache.get_or_load("s", "b", loader)
    assert calls[0] == 2

    await cache.clear()
    await cache.get_or_load("s", "a", loader)
    await cache.get_or_load("s", "b", loader)
    assert calls[0] == 4

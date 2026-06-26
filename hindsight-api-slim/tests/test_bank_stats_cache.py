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
async def test_invalidate_detaches_in_flight_loader() -> None:
    cache = BankStatsCache(ttl_seconds=60, max_entries=100)
    stale_started = asyncio.Event()
    release_stale = asyncio.Event()
    fresh_started = asyncio.Event()

    async def stale_loader() -> dict[str, Any]:
        stale_started.set()
        await release_stale.wait()
        return {"v": "stale"}

    async def fresh_loader() -> dict[str, Any]:
        fresh_started.set()
        return {"v": "fresh"}

    stale_task = asyncio.create_task(cache.get_or_load("schema", "bank", stale_loader))
    await stale_started.wait()
    await cache.invalidate("schema", "bank")

    # A request after invalidation must start a new load instead of joining the
    # pre-invalidation query, which may contain data from before a bank write.
    fresh_result = await asyncio.wait_for(cache.get_or_load("schema", "bank", fresh_loader), timeout=1)
    assert fresh_started.is_set()
    assert fresh_result == {"v": "fresh"}

    release_stale.set()
    assert await stale_task == {"v": "stale"}

    # The stale loader completed last, but must not overwrite the fresh value.
    async def should_not_run() -> dict[str, Any]:
        raise AssertionError("fresh value was not cached")

    cached = await cache.get_or_load("schema", "bank", should_not_run)
    assert cached == {"v": "fresh"}


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


@pytest.mark.asyncio
async def test_clear_detaches_in_flight_loaders() -> None:
    cache = BankStatsCache(ttl_seconds=60, max_entries=100)
    stale_started = asyncio.Event()
    release_stale = asyncio.Event()

    async def stale_loader() -> dict[str, Any]:
        stale_started.set()
        await release_stale.wait()
        return {"v": "stale"}

    async def fresh_loader() -> dict[str, Any]:
        return {"v": "fresh"}

    stale_task = asyncio.create_task(cache.get_or_load("schema", "bank", stale_loader))
    await stale_started.wait()
    await cache.clear()
    assert await cache.get_or_load("schema", "bank", fresh_loader) == {"v": "fresh"}

    release_stale.set()
    assert await stale_task == {"v": "stale"}


# ---------------------------------------------------------------------------
# key_suffix variant carve-out
# ---------------------------------------------------------------------------
# Callers can ask for the same (schema, bank_id) with different `key_suffix`
# values to carve out separate slots for variants that compute different
# results — e.g. a flag that toggles an optional expensive aggregation. The
# cache must not return one variant's payload to the other. `invalidate`,
# however, must clear every variant for a bank since writers don't know
# which suffixes the read path is using.


@pytest.mark.asyncio
async def test_key_suffix_separates_cache_slots() -> None:
    cache = BankStatsCache(ttl_seconds=60, max_entries=100)
    calls = {"with": 0, "without": 0}

    async def with_loader() -> dict[str, Any]:
        calls["with"] += 1
        return {"variant": "with"}

    async def without_loader() -> dict[str, Any]:
        calls["without"] += 1
        return {"variant": "without"}

    a = await cache.get_or_load("schema", "bank", with_loader, key_suffix=(True,))
    b = await cache.get_or_load("schema", "bank", without_loader, key_suffix=(False,))
    assert a == {"variant": "with"}
    assert b == {"variant": "without"}
    assert calls == {"with": 1, "without": 1}

    # Each variant has its own slot: a re-read returns its own cached value
    # and does not run the other loader.
    async def should_not_run() -> dict[str, Any]:
        raise AssertionError("variant cross-contamination")

    again_a = await cache.get_or_load("schema", "bank", should_not_run, key_suffix=(True,))
    again_b = await cache.get_or_load("schema", "bank", should_not_run, key_suffix=(False,))
    assert again_a == {"variant": "with"}
    assert again_b == {"variant": "without"}


@pytest.mark.asyncio
async def test_default_key_suffix_is_distinct_slot() -> None:
    """Omitting key_suffix uses the empty tuple; it must not collide with (True,)."""
    cache = BankStatsCache(ttl_seconds=60, max_entries=100)

    calls = {"none": 0, "true": 0}

    async def none_loader() -> dict[str, Any]:
        calls["none"] += 1
        return {"slot": "default"}

    async def true_loader() -> dict[str, Any]:
        calls["true"] += 1
        return {"slot": "true"}

    # Default key_suffix=()
    assert await cache.get_or_load("schema", "bank", none_loader) == {"slot": "default"}
    # Explicit key_suffix=(True,) — different slot, runs its own loader.
    assert await cache.get_or_load("schema", "bank", true_loader, key_suffix=(True,)) == {
        "slot": "true"
    }
    assert calls == {"none": 1, "true": 1}


@pytest.mark.asyncio
async def test_invalidate_clears_all_variants_for_bank() -> None:
    cache = BankStatsCache(ttl_seconds=60, max_entries=100)

    async def loader_with() -> dict[str, Any]:
        return {"variant": "with"}

    async def loader_without() -> dict[str, Any]:
        return {"variant": "without"}

    await cache.get_or_load("schema", "bank", loader_with, key_suffix=(True,))
    await cache.get_or_load("schema", "bank", loader_without, key_suffix=(False,))

    # Invalidate without knowing which suffixes exist; both variants must clear.
    await cache.invalidate("schema", "bank")

    fresh_calls = {"with": 0, "without": 0}

    async def fresh_with() -> dict[str, Any]:
        fresh_calls["with"] += 1
        return {"variant": "with-2"}

    async def fresh_without() -> dict[str, Any]:
        fresh_calls["without"] += 1
        return {"variant": "without-2"}

    a = await cache.get_or_load("schema", "bank", fresh_with, key_suffix=(True,))
    b = await cache.get_or_load("schema", "bank", fresh_without, key_suffix=(False,))
    assert a == {"variant": "with-2"}
    assert b == {"variant": "without-2"}
    assert fresh_calls == {"with": 1, "without": 1}


@pytest.mark.asyncio
async def test_invalidate_keeps_other_banks() -> None:
    """invalidate on (schema, bank_a) must not touch (schema, bank_b) variants."""
    cache = BankStatsCache(ttl_seconds=60, max_entries=100)

    async def loader_a_with() -> dict[str, Any]:
        return {"bank": "a", "variant": "with"}

    async def loader_b_with() -> dict[str, Any]:
        return {"bank": "b", "variant": "with"}

    async def loader_b_without() -> dict[str, Any]:
        return {"bank": "b", "variant": "without"}

    await cache.get_or_load("schema", "bank_a", loader_a_with, key_suffix=(True,))
    await cache.get_or_load("schema", "bank_b", loader_b_with, key_suffix=(True,))
    await cache.get_or_load("schema", "bank_b", loader_b_without, key_suffix=(False,))

    await cache.invalidate("schema", "bank_a")

    async def should_not_run() -> dict[str, Any]:
        raise AssertionError("bank_b cache was cleared by bank_a invalidate")

    # bank_b variants must still be served from cache.
    assert (await cache.get_or_load("schema", "bank_b", should_not_run, key_suffix=(True,))) == {
        "bank": "b",
        "variant": "with",
    }
    assert (await cache.get_or_load("schema", "bank_b", should_not_run, key_suffix=(False,))) == {
        "bank": "b",
        "variant": "without",
    }


@pytest.mark.asyncio
async def test_concurrent_misses_coalesce_per_variant() -> None:
    """Two callers on the same variant share one load; different variants don't."""
    cache = BankStatsCache(ttl_seconds=60, max_entries=100)

    started = {"with": asyncio.Event(), "without": asyncio.Event()}
    release = asyncio.Event()
    calls = {"with": 0, "without": 0}

    async def with_loader() -> dict[str, Any]:
        calls["with"] += 1
        started["with"].set()
        await release.wait()
        return {"variant": "with"}

    async def without_loader() -> dict[str, Any]:
        calls["without"] += 1
        started["without"].set()
        await release.wait()
        return {"variant": "without"}

    # Two concurrent callers on the True variant should coalesce, and one
    # concurrent caller on the False variant should run independently.
    t1 = asyncio.create_task(cache.get_or_load("schema", "bank", with_loader, key_suffix=(True,)))
    t2 = asyncio.create_task(cache.get_or_load("schema", "bank", with_loader, key_suffix=(True,)))
    t3 = asyncio.create_task(
        cache.get_or_load("schema", "bank", without_loader, key_suffix=(False,))
    )

    await started["with"].wait()
    await started["without"].wait()
    release.set()

    results = await asyncio.gather(t1, t2, t3)
    assert results[0] == results[1] == {"variant": "with"}
    assert results[2] == {"variant": "without"}
    assert calls == {"with": 1, "without": 1}

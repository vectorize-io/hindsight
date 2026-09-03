"""Process-wide state must work from more than one event loop.

Hindsight can be run with an event loop per thread in a single process (uvicorn on
free-threaded CPython, `--target api-only-freethreaded`). Anything shared between
those loops must therefore not be tied to one of them.

**These tests need no free-threaded build.** `asyncio.Lock`/`Semaphore`/`Future` bind
to the loop that first waits on them regardless of the GIL, so two loops in two
threads reproduce the whole failure class on 3.11 — which is the point: this file is
the cheap guard that runs in the normal suite, so the expensive free-threaded CI job
is only needed for what genuinely requires it (a C extension silently re-enabling
the GIL, and true-parallelism races).

Every case below corresponds to a bug that shipped and was found only by running the
server with eight loops. Each failed with either
``RuntimeError: ... is bound to a different event loop`` or
``RuntimeError: dictionary changed size during iteration``.
"""

import asyncio
import threading
from typing import Any, Callable

import pytest

LOOPS = 4
PER_LOOP = 6


def _across_loops(make_coros: Callable[[], list], loops: int = LOOPS) -> list[BaseException]:
    """Run `make_coros()` concurrently on `loops` independent event loops.

    Returns the exceptions raised, so a caller can assert on emptiness and print
    something useful when it is not.
    """
    errors: list[BaseException] = []
    lock = threading.Lock()

    def run() -> None:
        async def main() -> Any:
            return await asyncio.gather(*(fn() for fn in make_coros()))

        try:
            asyncio.run(main())
        except BaseException as exc:  # noqa: BLE001 - reported by the assertion
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(loops)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_bank_info_cache_is_usable_from_several_loops():
    """Regression: the cache held an asyncio.Lock and coalesced behind an asyncio.Future."""
    from hindsight_api.engine.bank_stats_cache import BankStatsCache

    cache = BankStatsCache(ttl_seconds=60, max_entries=128)
    loads = 0
    guard = threading.Lock()

    async def loader() -> dict[str, Any]:
        nonlocal loads
        with guard:
            loads += 1
        await asyncio.sleep(0.01)  # widen the window so callers really do coalesce
        return {"value": 1}

    async def read() -> dict[str, Any]:
        return await cache.get_or_load("schema", "bank", loader)

    errors = _across_loops(lambda: [read] * PER_LOOP)
    assert not errors, f"cache failed across loops: {errors[:1]}"
    # The cached data is shared, so far fewer loads than callers.
    assert loads < LOOPS * PER_LOOP


def test_connection_budget_manager_is_usable_from_several_loops():
    """Regression: ConnectionBudgetManager is a process-wide singleton with a lock."""
    from hindsight_api.engine.db_budget import ConnectionBudgetManager

    manager = ConnectionBudgetManager(default_budget=2)

    async def use() -> None:
        async with manager.operation(max_connections=2):
            await asyncio.sleep(0.005)

    errors = _across_loops(lambda: [use] * PER_LOOP)
    assert not errors, f"budget manager failed across loops: {errors[:1]}"


def test_llm_concurrency_permits_are_usable_from_several_loops():
    """Regression: the LLM caps were module-level asyncio.Semaphores built at import."""
    from hindsight_api.engine.llm_wrapper import _semaphores_for_scope

    async def hold() -> None:
        from contextlib import AsyncExitStack

        async with AsyncExitStack() as stack:
            for sem in _semaphores_for_scope("retain"):
                await stack.enter_async_context(sem)
            await asyncio.sleep(0.005)

    errors = _across_loops(lambda: [hold] * PER_LOOP)
    assert not errors, f"LLM permits failed across loops: {errors[:1]}"


def test_temporal_language_detection_is_usable_from_several_threads():
    """Regression: dateparser cleans a cached locale dict in place on first use.

    Two threads reaching a locale together raised "dictionary changed size during
    iteration", which killed the lifespan of whichever loops lost the race.
    """
    from dateparser.languages.loader import LocaleDataLoader

    from hindsight_api.engine.temporal_language_detection import best_language

    locales = list(LocaleDataLoader().get_locales(languages=None, locales=None, region=None))[:40]

    # Vary the text so different character sets select different candidate locales,
    # which is what left some dictionaries cold when only English was exercised.
    texts = [
        "what happened last friday",
        "cosa e successo ieri sera",
        "was ist gestern passiert",
        "que paso el martes pasado",
    ]

    async def detect() -> None:
        for text in texts:
            best_language(text, locales)

    errors = _across_loops(lambda: [detect] * 2, loops=6)
    assert not errors, f"language detection failed across threads: {errors[:1]}"


@pytest.mark.asyncio
async def test_asyncio_primitives_still_bind_to_a_loop():
    """Pins the premise. If this ever stops failing, the guards above can be revisited."""
    shared = asyncio.Semaphore(1)
    async with shared:
        pass

    errors = _across_loops(lambda: [lambda: _contend(shared)] * 3, loops=3)
    assert any(isinstance(e, RuntimeError) and "different event loop" in str(e) for e in errors), (
        "a module-level asyncio.Semaphore no longer breaks across loops"
    )


async def _contend(sem: asyncio.Semaphore) -> None:
    async with sem:
        await asyncio.sleep(0.02)

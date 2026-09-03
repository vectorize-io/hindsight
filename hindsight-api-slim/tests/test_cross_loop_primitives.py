"""Cross-loop primitives must work where asyncio's own do not.

`asyncio.Lock`/`Semaphore` bind to the event loop that first waits on them, so a
module-level one breaks the moment a second loop contends it. That is not a
free-threading-only problem — it reproduces on any build as soon as two loops exist
in one process — so these tests run everywhere.
"""

import asyncio
import functools
import threading
import time

import pytest

from hindsight_api._cross_loop import CrossLoopLock, CrossLoopSemaphore


def _run_in_own_loop(make_coros, results: list, errors: list) -> threading.Thread:
    """Run `make_coros()` concurrently on a fresh event loop in a new thread.

    `make_coros` returns coroutine *factories*, not coroutines: gather has to be
    built inside the loop, not before `asyncio.run` creates one.
    """

    def run():
        async def main():
            return await asyncio.gather(*(fn() for fn in make_coros()))

        try:
            results.append(asyncio.run(main()))
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
            errors.append(exc)

    return threading.Thread(target=run)


def test_asyncio_semaphore_is_the_thing_being_replaced():
    """Pins the failure mode, so the reason for this module cannot quietly stop applying."""
    shared = asyncio.Semaphore(1)
    errors: list[BaseException] = []
    results: list[object] = []

    async def contend():
        async with shared:
            await asyncio.sleep(0.05)
        return True

    threads = [_run_in_own_loop(lambda: [contend] * 4, results, errors) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert any(isinstance(e, RuntimeError) and "different event loop" in str(e) for e in errors), (
        "asyncio.Semaphore no longer fails across loops; CrossLoopSemaphore may be unnecessary"
    )


def test_cross_loop_semaphore_survives_several_loops():
    shared = CrossLoopSemaphore(2)
    errors: list[BaseException] = []
    results: list[object] = []

    async def contend():
        async with shared:
            await asyncio.sleep(0.02)
        return True

    threads = [_run_in_own_loop(lambda: [contend] * 5, results, errors) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"cross-loop use raised: {errors[:1]}"
    assert len(results) == 4


def test_cross_loop_semaphore_actually_caps_concurrency():
    """The cap is process-wide, not per loop — that is the whole point of it."""
    shared = CrossLoopSemaphore(3)
    active = 0
    peak = 0
    guard = threading.Lock()
    errors: list[BaseException] = []
    results: list[object] = []

    async def work():
        nonlocal active, peak
        async with shared:
            with guard:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.02)
            with guard:
                active -= 1
        return True

    threads = [_run_in_own_loop(lambda: [work] * 6, results, errors) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert peak <= 3, f"cap leaked: {peak} concurrent holders across loops"
    assert peak > 1, "never actually contended, so the cap was not exercised"


def test_cross_loop_lock_is_exclusive_across_loops():
    lock = CrossLoopLock()
    order: list[str] = []
    guard = threading.Lock()
    errors: list[BaseException] = []
    results: list[object] = []

    async def critical(tag: str):
        async with lock:
            with guard:
                order.append(f"enter-{tag}")
            await asyncio.sleep(0.01)  # held across an await, unlike threading.Lock
            with guard:
                order.append(f"exit-{tag}")
        return True

    threads = [
        _run_in_own_loop(lambda i=i: [functools.partial(critical, f"{i}-{j}") for j in range(3)], results, errors)
        for i in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # Exclusivity: every enter is immediately followed by its own exit.
    for a, b in zip(order[::2], order[1::2]):
        assert a.replace("enter-", "") == b.replace("exit-", ""), f"interleaved critical sections: {order}"


@pytest.mark.asyncio
async def test_uncontended_acquire_does_not_yield():
    """The fast path must not cost a scheduler round-trip on every LLM call."""
    sem = CrossLoopSemaphore(4)
    start = time.perf_counter()
    for _ in range(1000):
        async with sem:
            pass
    assert time.perf_counter() - start < 0.5

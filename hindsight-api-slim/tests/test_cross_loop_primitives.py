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


@pytest.mark.asyncio
async def test_holder_that_re_acquires_without_suspending_does_not_starve_a_waiter():
    """A permit released and retaken in the same step must still reach the queue first."""
    sem = CrossLoopSemaphore(1)
    hold = 0.01
    stop = asyncio.Event()
    sections = 0

    async def holder():
        nonlocal sections
        while not stop.is_set():
            async with sem:
                await asyncio.sleep(hold)
                sections += 1

    task = asyncio.create_task(holder())
    try:
        await asyncio.sleep(5 * hold)
        start = time.perf_counter()
        await asyncio.wait_for(sem.acquire(), timeout=2.0)
        waited = time.perf_counter() - start
        sem.release()
    finally:
        stop.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert waited < 20 * hold, f"waited {waited * 1000:.0f} ms behind {sections} critical sections"


@pytest.mark.asyncio
async def test_waiters_are_served_in_arrival_order():
    """Handoff order is the arrival order, not whichever waiter happens to poll first."""
    sem = CrossLoopSemaphore(1)
    served: list[str] = []
    await sem.acquire()

    async def waiter(tag: str):
        async with sem:
            served.append(tag)

    tasks = []
    for tag in "abcde":
        tasks.append(asyncio.create_task(waiter(tag)))
        # Stagger arrivals so each waiter sits at a different point on the backoff ladder.
        await asyncio.sleep(0.005)

    sem.release()
    await asyncio.gather(*tasks)

    assert served == list("abcde")


@pytest.mark.asyncio
async def test_cancelled_waiters_neither_strand_a_permit_nor_block_the_queue():
    """Cancellation must pass on a handed-over permit and drop a still-queued waiter."""
    sem = CrossLoopSemaphore(1)
    await sem.acquire()

    first = asyncio.create_task(sem.acquire())
    await asyncio.sleep(0.005)
    second = asyncio.create_task(sem.acquire())
    await asyncio.sleep(0.005)
    queued = asyncio.create_task(sem.acquire())
    await asyncio.sleep(0.005)

    # Cancelled behind a held permit, so it leaves the queue owning nothing.
    queued.cancel()
    await asyncio.gather(queued, return_exceptions=True)

    # Cancelled before it can wake, so it is holding the permit just handed to it.
    sem.release()
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)

    await asyncio.wait_for(second, timeout=2.0)
    sem.release()
    await asyncio.wait_for(sem.acquire(), timeout=0.5)


def test_handoff_reaches_a_waiter_on_another_loop():
    """The starvation fix must hold across loops, which is the only reason this class exists.

    The single-loop starvation test above cannot see this: there the holder's release and
    the waiter's wake-up run on the same scheduler. Here the permit is granted by a
    release on one loop to a ticket owned by a thread running a different one.
    """
    sem = CrossLoopSemaphore(1)
    stop = threading.Event()
    sections = 0
    guard = threading.Lock()
    errors: list[BaseException] = []
    results: list[object] = []
    waited: list[float] = []

    async def holder():
        # Re-acquires with no suspension point between release and the next acquire,
        # so a bare counter would hand the permit straight back to this task forever.
        nonlocal sections
        while not stop.is_set():
            async with sem:
                await asyncio.sleep(0.01)
                with guard:
                    sections += 1
        return True

    async def waiter():
        # Let the holder saturate the cap first, so this is a genuinely contended wait.
        await asyncio.sleep(0.05)
        start = time.perf_counter()
        async with sem:
            waited.append(time.perf_counter() - start)
        stop.set()
        return True

    threads = [
        _run_in_own_loop(lambda: [holder], results, errors),
        _run_in_own_loop(lambda: [waiter], results, errors),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    stop.set()
    assert not errors, f"cross-loop handoff raised: {errors[:1]}"
    assert waited, "waiter never acquired across the loop boundary"
    assert waited[0] < 0.2, f"waited {waited[0] * 1000:.0f} ms behind {sections} critical sections on another loop"

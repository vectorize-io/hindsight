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


@pytest.mark.asyncio
async def test_foreground_waiter_jumps_ahead_of_background_waiters():
    """A live request queued after background jobs still gets the next free permit."""
    sem = CrossLoopSemaphore(1)
    served: list[str] = []
    await sem.acquire()

    async def waiter(tag: str, background: bool):
        await sem.acquire(background=background)
        served.append(tag)
        sem.release()

    tasks = [asyncio.create_task(waiter("bg1", True))]
    await asyncio.sleep(0.005)
    tasks.append(asyncio.create_task(waiter("bg2", True)))
    await asyncio.sleep(0.005)
    tasks.append(asyncio.create_task(waiter("live", False)))
    await asyncio.sleep(0.005)
    assert sem.locked()

    sem.release()
    await asyncio.gather(*tasks)

    assert served == ["live", "bg1", "bg2"]


async def _serve_order(sem: CrossLoopSemaphore, arrivals: list[tuple[str, str, str, str | None]]) -> list[str]:
    """Queue `(tag, tenant, bank, document)` waiters behind a held permit, release it, return serve order."""
    served: list[str] = []
    await sem.acquire()

    async def waiter(tag: str, tenant: str, bank: str, document: str | None):
        await sem.acquire(tenant=tenant, bank=bank, document=document)
        served.append(tag)
        sem.release()

    tasks = []
    for arrival in arrivals:
        tasks.append(asyncio.create_task(waiter(*arrival)))
        await asyncio.sleep(0.002)  # fix arrival order
    sem.release()
    await asyncio.gather(*tasks)
    return served


@pytest.mark.asyncio
async def test_a_bank_that_queued_later_takes_turns_with_a_big_one():
    """Banks take turns, whatever their size: one bank's backlog does not hold up another."""
    big = [(f"big{i}", "t1", "a", "doc") for i in range(4)]
    served = await _serve_order(CrossLoopSemaphore(1), big + [("other", "t1", "b", "doc")])

    assert served == ["big0", "other", "big1", "big2", "big3"]


@pytest.mark.asyncio
async def test_tenants_take_turns_before_banks():
    """A tenant with many banks waiting gets no more turns than a tenant with one."""
    many = [(f"c{i}", "t1", f"bank{i}", None) for i in range(4)]
    one = [(f"d{i}", "t2", "only", None) for i in range(2)]
    served = await _serve_order(CrossLoopSemaphore(1), many + one)

    assert served == ["c0", "d0", "c1", "d1", "c2", "c3"]


@pytest.mark.asyncio
async def test_inside_a_bank_a_small_document_goes_ahead_of_a_big_one():
    """The document closest to done is served first, even if it queued later."""
    big = [(f"big{i}", "t1", "a", "big") for i in range(4)]
    served = await _serve_order(CrossLoopSemaphore(1), big + [("small", "t1", "a", "small")])

    assert served == ["small", "big0", "big1", "big2", "big3"]


@pytest.mark.asyncio
async def test_a_big_document_still_moves_while_small_ones_keep_arriving():
    """Every 4th turn of a bank goes to its longest-waiting document, so a big one is not starved."""
    big = [(f"big{i}", "t1", "a", "big") for i in range(4)]
    small = [(f"s{i}", "t1", "a", f"s{i}") for i in range(6)]
    served = await _serve_order(CrossLoopSemaphore(1), big + small)

    assert served[:8] == ["s0", "s1", "s2", "big0", "s3", "s4", "s5", "big1"]


@pytest.mark.asyncio
async def test_equal_documents_in_a_bank_finish_one_after_another():
    """Equal documents are not interleaved to the end: the one that arrived first finishes first."""
    arrivals = [(f"{d}{i}", "t1", "a", d) for d in "xy" for i in range(3)]
    served = await _serve_order(CrossLoopSemaphore(1), arrivals)

    assert served == ["x0", "x1", "x2", "y0", "y1", "y2"]


@pytest.mark.asyncio
async def test_a_lone_bank_still_gets_every_permit():
    """Turn-taking reserves nothing: one bank alone runs at the full cap."""
    sem = CrossLoopSemaphore(4)
    in_flight = peak = 0

    async def call():
        nonlocal in_flight, peak
        await sem.acquire(tenant="t1", bank="a", document="doc")
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        sem.release()

    await asyncio.gather(*(call() for _ in range(20)))

    assert peak == 4

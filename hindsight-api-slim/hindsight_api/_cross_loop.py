"""Concurrency primitives that several event loops in one process can share.

``asyncio.Lock``/``Semaphore`` bind to the event loop that first *waits* on them.
That is fine for state one loop owns, and wrong for anything process-wide: on a
free-threaded build (``python3.14t``) uvicorn can run an event loop per thread, and
the first contended acquire claims the primitive for its loop while every other loop
fails with ``RuntimeError: ... is bound to a different event loop``. The failure only
appears under contention, so it passes tests and breaks under load.

The counters here live in ``threading`` primitives, which are loop-agnostic, and the
wait is a short async backoff so a loop is never blocked while it queues.

**Why polling rather than a cross-loop future handoff.** Waking a waiter on another
loop means ``loop.call_soon_threadsafe`` and a registry of waiters per loop, which has
to stay correct when a loop dies mid-wait. Polling has none of that state: it only
runs while the cap is saturated, and it costs at most ``_MAX_DELAY`` of extra latency
on acquiring a slot. These caps gate LLM calls and llama.cpp server startup —
operations measured in hundreds of milliseconds to minutes — so that is not
measurable. Do not reach for this to guard something short and hot; scope the state
per loop instead.
"""

from __future__ import annotations

import asyncio
import threading

__all__ = ["CrossLoopSemaphore", "CrossLoopLock"]

#: Backoff bounds for a saturated wait. Starts tight so an uncontended-but-just-freed
#: slot is picked up almost immediately, and widens so a long queue does not spin.
_MIN_DELAY = 0.001
_MAX_DELAY = 0.02


class CrossLoopSemaphore:
    """A concurrency cap shared by every event loop in the process.

    Drop-in for ``asyncio.Semaphore`` as an async context manager. The cap stays
    process-wide, which is the contract the surrounding config already implies:
    running ``--workers N`` has always meant N independent caps, one per process.
    """

    def __init__(self, value: int) -> None:
        self._capacity = value
        self._sem = threading.Semaphore(value)

    @property
    def capacity(self) -> int:
        """The configured cap. Public so callers and tests need not read a private."""
        return self._capacity

    async def acquire(self) -> None:
        # Fast path: uncontended acquire never yields, so the common case costs one
        # atomic operation and no scheduler round-trip.
        if self._sem.acquire(blocking=False):
            return
        delay = _MIN_DELAY
        while not self._sem.acquire(blocking=False):
            await asyncio.sleep(delay)
            delay = min(delay * 2, _MAX_DELAY)

    def release(self) -> None:
        self._sem.release()

    async def __aenter__(self) -> "CrossLoopSemaphore":
        await self.acquire()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.release()


class CrossLoopLock(CrossLoopSemaphore):
    """Mutual exclusion across every event loop in the process.

    Unlike ``threading.Lock`` this is safe to hold across ``await`` — waiters yield
    instead of blocking their loop — so it suits sections that do real async work,
    such as starting or stopping a shared subprocess.
    """

    def __init__(self) -> None:
        super().__init__(1)

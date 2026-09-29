"""Concurrency primitives that several event loops in one process can share.

``asyncio.Lock``/``Semaphore`` bind to the event loop that first *waits* on them.
That is fine for state one loop owns, and wrong for anything process-wide: the first
contended acquire claims the primitive for its loop, and any other loop reaching the
same module-level or class-level object then fails with ``RuntimeError: ... is bound
to a different event loop``. The failure only appears under contention, so it passes
tests and breaks under load.

The state here lives under a ``threading.Lock``, which is loop-agnostic, and a waiter
polls its own ticket on a short async backoff so a loop is never blocked while it
queues.

**Why a queue rather than a bare counter.** A counter alone starves waiters: a holder
that releases and re-acquires with no suspension point in between takes its own permit
straight back, and the waiter — asleep on the backoff ladder for the whole moment the
permit was free — never sees it. So ``release()`` hands the permit to the
longest-waiting ticket instead of returning it to the pool.

**Why polling rather than a cross-loop future handoff.** The queue is a flat deque of
tickets; a granted waiter still learns about it by polling. Waking it directly would
mean ``loop.call_soon_threadsafe`` and a registry keyed *per loop*, which has to stay
correct when a loop dies mid-wait. Polling keeps none of that: it only runs while the
cap is saturated, and it costs at most ``_MAX_DELAY`` between the handoff and the
waiter resuming. These caps gate LLM calls and llama.cpp server startup — operations
measured in hundreds of milliseconds to minutes — so that is not measurable. Do not
reach for this to guard something short and hot; scope the state per loop instead.

**The cost of handing over.** A permit granted to a waiter is reserved for it, so a
waiter that never resumes takes the permit with it and the cap drops by one for the
life of the process. Cancellation is handled (a cancelled waiter passes on a permit it
was granted), so in practice this needs a loop closed with a pending ``acquire()`` on
it — ``asyncio.run`` cancels those, ``run_until_complete`` plus ``close`` does not.
"""

from __future__ import annotations

import asyncio
import collections
import threading

__all__ = ["CrossLoopSemaphore", "CrossLoopLock"]

#: Backoff bounds for a saturated wait. Starts tight so an uncontended-but-just-freed
#: slot is picked up almost immediately, and widens so a long queue does not spin.
_MIN_DELAY = 0.001
_MAX_DELAY = 0.02


class _Ticket:
    """One waiter's place in the queue, and where its permit is handed to it."""

    __slots__ = ("granted",)

    def __init__(self) -> None:
        self.granted = False


class CrossLoopSemaphore:
    """A concurrency cap shared by every event loop in the process.

    Drop-in for ``asyncio.Semaphore`` as an async context manager. The cap stays
    process-wide, which is the contract the surrounding config already implies:
    running ``--workers N`` has always meant N independent caps, one per process.

    Waiters are served in arrival order, as ``asyncio.Semaphore`` serves them.
    """

    def __init__(self, value: int) -> None:
        self._capacity = value
        self._lock = threading.Lock()
        self._free = value
        self._waiters: collections.deque[_Ticket] = collections.deque()

    @property
    def capacity(self) -> int:
        """The configured cap. Public so callers and tests need not read a private."""
        return self._capacity

    async def acquire(self) -> None:
        with self._lock:
            # Fast path: uncontended acquire never yields. Gated on an empty queue,
            # so an arriving task cannot take a permit an earlier waiter is owed.
            if not self._waiters and self._free:
                self._free -= 1
                return
            ticket = _Ticket()
            self._waiters.append(ticket)

        delay = _MIN_DELAY
        try:
            while True:
                with self._lock:
                    if ticket.granted:
                        return
                await asyncio.sleep(delay)
                delay = min(delay * 2, _MAX_DELAY)
        except BaseException:
            with self._lock:
                if ticket.granted:
                    self._hand_on()
                else:
                    self._waiters.remove(ticket)
            raise

    def release(self) -> None:
        with self._lock:
            self._hand_on()

    def _hand_on(self) -> None:
        """Give the permit to the longest-waiting task, or return it to the pool.

        Handing it over directly is what stops a holder that re-acquires without
        suspending from taking its own permit back before any waiter can see it free.

        The permit is granted here rather than claimed by the waiter on its next poll,
        which trades a little latency for burst behaviour: the grantee can sit on the
        permit for up to ``_MAX_DELAY`` before it wakes and nobody else may take it,
        but K permits freed at once reach K waiters at once. Claiming would hand out
        one permit per poll tick, draining a burst at ``_MAX_DELAY`` per waiter.
        """
        if self._waiters:
            self._waiters.popleft().granted = True
        else:
            self._free += 1

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

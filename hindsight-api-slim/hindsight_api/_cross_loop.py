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
permit was free — never sees it. So ``release()`` hands the permit to the next
waiting ticket (see ``_FairQueue`` for which one) instead of returning it to the pool.

**Why polling rather than a cross-loop future handoff.** The queue holds plain
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
import itertools
import threading

__all__ = ["CrossLoopSemaphore", "CrossLoopLock"]

#: Backoff bounds for a saturated wait. Starts tight so an uncontended-but-just-freed
#: slot is picked up almost immediately, and widens so a long queue does not spin.
_MIN_DELAY = 0.001
_MAX_DELAY = 0.02


class _Ticket:
    """One waiter's place in the queue, and where its permit is handed to it."""

    __slots__ = ("granted", "tenant", "bank", "document", "seq")

    def __init__(self, tenant: str | None, bank: str | None, document: str | None, seq: int) -> None:
        self.granted = False
        self.tenant = tenant
        self.bank = bank
        self.document = document
        self.seq = seq  # arrival order across the whole queue


#: Inside a bank, every Nth turn goes to the document that has waited longest instead
#: of the one closest to done, so a big document still moves while small ones arrive.
_OLDEST_EVERY = 4

_Lines = dict[str | None, collections.deque[_Ticket]]


class _FairQueue:
    """Waiters grouped by tenant, then bank, then document.

    Tenants take turns, and inside a tenant its banks take turns: each permit goes to
    the next one with a waiter, so a tenant or bank gets the same share whether it
    queued one call or a thousand, and one big document cannot make another bank
    wait for all of it.

    Inside one bank, the permit goes to the document with the fewest calls waiting:
    the one closest to done. A small document queued behind a big one finishes almost
    at once, and equal documents finish one after another instead of all together at
    the end. Every ``_OLDEST_EVERY``-th turn of the bank goes to the document that has
    waited longest instead, so a big one keeps at least that share while small ones
    keep arriving. Ties go to the document that arrived first; inside one document,
    arrival order. Calls with no document (consolidation, reflect) share one line.

    With nothing given this is a plain FIFO queue. Nothing is reserved: one waiter
    alone still gets every permit.

    History: this used to be one FIFO deque, so a retain that queued one call per
    chunk made every later call wait for all of its chunks (#6727). A fixed share per
    operation was rejected because a document alone would then leave most permits
    idle. Strict turns between documents was tried and made equal documents all
    finish at the end; picking the bank with the least queued was tried and favoured
    small banks over busy ones for no reason. Hence: turns between tenants and banks,
    closest-to-done only between documents of one bank.
    """

    def __init__(self) -> None:
        self._tenants: dict[str | None, dict[str | None, _Lines]] = {}
        self._turns: dict[tuple[str | None, str | None], int] = {}
        self._seq = itertools.count()

    def __bool__(self) -> bool:
        return bool(self._tenants)

    def make_ticket(self, tenant: str | None, bank: str | None, document: str | None) -> _Ticket:
        return _Ticket(tenant, bank, document, next(self._seq))

    def push(self, ticket: _Ticket) -> None:
        banks = self._tenants.setdefault(ticket.tenant, {})
        docs = banks.setdefault(ticket.bank, {})
        docs.setdefault(ticket.document, collections.deque()).append(ticket)

    def pop(self) -> _Ticket:
        tenant, banks = next(iter(self._tenants.items()))
        bank, docs = next(iter(banks.items()))
        turn = self._turns.get((tenant, bank), 0) + 1
        self._turns[(tenant, bank)] = turn
        # ponytail: linear scan over the bank's waiting documents per handoff; fine
        # for hundreds, keep a heap per bank if one bank queues thousands at once.
        if turn % _OLDEST_EVERY == 0:
            doc = min(docs, key=lambda d: docs[d][0].seq)
        else:
            doc = min(docs, key=lambda d: len(docs[d]))
        ticket = docs[doc][0]
        self.remove(ticket)
        # The bank and tenant just served go to the back of their turn order.
        if bank in banks:
            banks[bank] = banks.pop(bank)
        if tenant in self._tenants:
            self._tenants[tenant] = self._tenants.pop(tenant)
        return ticket

    def remove(self, ticket: _Ticket) -> None:
        banks = self._tenants[ticket.tenant]
        docs = banks[ticket.bank]
        line = docs[ticket.document]
        line.remove(ticket)
        if not line:
            del docs[ticket.document]
        if not docs:
            del banks[ticket.bank]
            self._turns.pop((ticket.tenant, ticket.bank), None)
        if not banks:
            del self._tenants[ticket.tenant]


class CrossLoopSemaphore:
    """A concurrency cap shared by every event loop in the process.

    Drop-in for ``asyncio.Semaphore`` as an async context manager. The cap stays
    process-wide, which is the contract the surrounding config already implies:
    running ``--workers N`` has always meant N independent caps, one per process.

    Waiters are served in arrival order, as ``asyncio.Semaphore`` serves them, except
    that ``acquire(background=True)`` waiters only get a permit when no foreground
    waiter is queued. That lets live requests jump ahead of background jobs. Waiters
    that pass ``tenant``/``bank``/``document`` take turns by tenant, then by bank, and
    inside a bank the document closest to done goes first (see ``_FairQueue``), so one
    big job cannot make everyone else wait for all of it.
    """

    def __init__(self, value: int) -> None:
        self._capacity = value
        self._lock = threading.Lock()
        self._free = value
        self._waiters = _FairQueue()
        self._background_waiters = _FairQueue()

    @property
    def capacity(self) -> int:
        """The configured cap. Public so callers and tests need not read a private."""
        return self._capacity

    def locked(self) -> bool:
        """True when an acquire would wait, as ``asyncio.Semaphore.locked`` reports it."""
        with self._lock:
            return not self._free or bool(self._waiters or self._background_waiters)

    async def acquire(
        self,
        *,
        background: bool = False,
        tenant: str | None = None,
        bank: str | None = None,
        document: str | None = None,
    ) -> None:
        """Wait for a permit. ``background`` waiters queue behind every foreground one.

        ``tenant``, ``bank`` and ``document`` pick the line the waiter joins (see
        ``_FairQueue`` for how lines take turns).
        """
        queue = self._background_waiters if background else self._waiters
        with self._lock:
            # Fast path: uncontended acquire never yields. Gated on empty queues,
            # so an arriving task cannot take a permit an earlier waiter is owed.
            if not self._waiters and not self._background_waiters and self._free:
                self._free -= 1
                return
            ticket = queue.make_ticket(tenant, bank, document)
            queue.push(ticket)

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
                    queue.remove(ticket)
            raise

    def release(self) -> None:
        with self._lock:
            self._hand_on()

    def _hand_on(self) -> None:
        """Give the permit to the next waiter in turn, or return it to the pool.

        Foreground waiters are served first; a background waiter gets the permit only
        when no foreground one is queued (#5926).

        Handing it over directly is what stops a holder that re-acquires without
        suspending from taking its own permit back before any waiter can see it free.

        The permit is granted here rather than claimed by the waiter on its next poll,
        which trades a little latency for burst behaviour: the grantee can sit on the
        permit for up to ``_MAX_DELAY`` before it wakes and nobody else may take it,
        but K permits freed at once reach K waiters at once. Claiming would hand out
        one permit per poll tick, draining a burst at ``_MAX_DELAY`` per waiter.
        """
        if self._waiters:
            self._waiters.pop().granted = True
        elif self._background_waiters:
            # ponytail: background can starve under constant foreground load; add an
            # every-N-handoffs fairness rule if that shows up in practice.
            self._background_waiters.pop().granted = True
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

"""Admission control: bound the wait, not just the concurrency.

The engine already limits concurrent work (``recall_max_concurrent`` and friends),
but a bare ``async with semaphore`` is *backpressure*, not admission control: it
caps how much runs at once and lets an unbounded queue form behind it. Measured on
a 2-vCPU container, 1024 concurrent recalls against a 32-permit semaphore produced
a 12.8s p50 — the latency did not go away, it moved from the event loop into the
semaphore queue, and the server spent CPU on responses whose clients had long
since given up.

What is missing is a bound on *waiting*. A lane here has two numbers:

* ``max_in_flight`` — how much of this operation runs concurrently;
* ``max_wait_seconds`` — how long a request may queue before it is refused.

A request that cannot be admitted within its deadline gets 503 with ``Retry-After``
immediately, which is a far better answer than a response that arrives after the
caller timed out. Capacity is unchanged; what changes is that the capacity stops
being spent on work nobody is waiting for.

**Where this runs.** As a FastAPI dependency on the same routes that carry
``precheck_for``, so a rejection happens *before the request body is deserialised*
— the cheapest possible point to say no, and the same place an extension already
rejects on quota.

**Why a lane per operation rather than one global limit.** Per-request cost spans
three orders of magnitude on this API (measured: ~0.1ms for ``/health/live``,
~0.5ms for bank stats, ~23ms for a recall). A single global request cap calibrated
for recall would throttle health checks that the server can serve 100x faster; one
calibrated for health would never engage for recall. The operations worth limiting
are exactly the ones :class:`PrecheckOperation` already enumerates.

**The limits are PER WORKER PROCESS.** Each ``--workers N`` process imports the app
and builds its own controller, so the process-wide budget is ``N x max_in_flight``.
Size a lane for one worker, not for the cluster.

A plain ``asyncio.Semaphore`` is correct here because one process runs one event
loop. (It would not be under ``--event-loops > 1`` on a free-threaded build, where
each loop would get its own semaphore and admit N times the limit — if multi-loop
serving ever comes back, this needs ``CrossLoopSemaphore`` instead.) The semaphore
is constructed before any loop is running, which is fine on 3.10+: it binds lazily
on first await, not at construction.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from ..cancellation import CancellationToken

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LaneConfig:
    """Limits for one operation class."""

    #: Concurrent requests of this operation. 0 disables the lane entirely (the
    #: resolved value -- see `admission_in_flight_for`, where a *negative* env var
    #: is what resolves to 0, because 0 there means "derive from the CPU budget").
    max_in_flight: int
    #: How long a request may queue for a permit before being refused. 0 means
    #: "never queue": either a permit is free now or the request is rejected.
    max_wait_seconds: float

    @property
    def enabled(self) -> bool:
        return self.max_in_flight > 0


class AdmissionRejected(Exception):
    """Raised when a request could not be admitted within its deadline."""

    def __init__(self, lane: str, waited_seconds: float, limit: int) -> None:
        self.lane = lane
        self.waited_seconds = waited_seconds
        self.limit = limit
        super().__init__(
            f"admission refused for {lane!r} after waiting {waited_seconds * 1000:.0f}ms (limit {limit} concurrent)"
        )

    @property
    def retry_after_seconds(self) -> int:
        """Conservative hint for the client, in whole seconds (``Retry-After``)."""
        return max(1, round(self.waited_seconds)) if self.waited_seconds else 1


@dataclass
class AdmissionAbandoned(Exception):
    """The client disconnected while its request was queued for a permit.

    Distinct from :class:`AdmissionRejected`: there is nobody left to send a 503 to,
    so the only useful thing to do is stop working on the request.
    """

    def __init__(self, lane: str) -> None:
        self.lane = lane
        super().__init__(f"client disconnected while queued for {lane!r}")


class _ClientGone(Exception):
    """Internal signal from the acquire race."""


async def _acquire_unless_abandoned(
    semaphore: asyncio.Semaphore, timeout: float, abandoned: CancellationToken | None
) -> None:
    """Acquire within ``timeout``, giving up early if the client disconnects.

    Raises :class:`_ClientGone` when the client vanished first, ``TimeoutError`` when
    the deadline passed.
    """
    if abandoned is None:
        await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
        return

    # Already gone before it even queued: there is no work worth starting, and
    # checking here also settles the race below, where a free permit and a cancelled
    # token would otherwise both be "done" and the permit would win.
    if abandoned.cancelled:
        raise _ClientGone()

    acquire = asyncio.ensure_future(semaphore.acquire())
    gone = asyncio.ensure_future(abandoned.wait())
    try:
        done, _ = await asyncio.wait({acquire, gone}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    finally:
        gone.cancel()

    if acquire in done:
        acquire.result()
        return

    # Not getting a permit. Cancelling a pending acquire is safe -- asyncio's
    # Semaphore wakes the next waiter when a granted-then-cancelled acquire unwinds
    # -- but a permit can land in the instant we give up, so if the task completed
    # anyway hand it straight back rather than leak it for the life of the process.
    acquire.cancel()
    try:
        await acquire
    except asyncio.CancelledError:
        pass
    else:
        semaphore.release()

    if gone in done:
        raise _ClientGone()
    raise TimeoutError()


@dataclass
class LaneStats:
    """Observable counters for one lane. Cheap to read; useful in an incident."""

    admitted: int = 0
    rejected: int = 0
    #: Clients that disconnected while queued -- neither served nor refused.
    abandoned: int = 0
    in_flight: int = 0
    queued: int = 0
    total_wait_seconds: float = 0.0

    @property
    def mean_wait_ms(self) -> float:
        total = self.admitted + self.rejected
        return (self.total_wait_seconds / total * 1000) if total else 0.0


class AdmissionController:
    """Per-operation concurrency limits with a bounded queue wait."""

    def __init__(self, lanes: dict[str, LaneConfig]) -> None:
        self._configs = lanes
        self._semaphores: dict[str, asyncio.Semaphore] = {
            name: asyncio.Semaphore(cfg.max_in_flight) for name, cfg in lanes.items() if cfg.enabled
        }
        self._stats: dict[str, LaneStats] = {name: LaneStats() for name in lanes}

    def lane_config(self, operation: str) -> LaneConfig | None:
        return self._configs.get(operation)

    def stats(self) -> dict[str, LaneStats]:
        return self._stats

    @asynccontextmanager
    async def admit(self, operation: str, abandoned: CancellationToken | None = None) -> AsyncIterator[None]:
        """Hold a permit for ``operation`` for the duration of the block.

        Falls through with no gating when the operation has no lane or the lane is
        disabled, so an unconfigured operation behaves exactly as it does today.

        ``abandoned`` is the request's client-disconnect token when one is available
        (recall and reflect carry one — see :mod:`hindsight_api.api.disconnect`). A
        queued request whose client has gone away gives up its place immediately
        instead of holding it for the full deadline. That is what makes a *longer*
        wait safe: the queue self-cleans, so patience costs nothing when nobody is
        still listening.
        """
        semaphore = self._semaphores.get(operation)
        if semaphore is None:
            yield
            return

        config = self._configs[operation]
        stats = self._stats[operation]
        started = time.monotonic()
        stats.queued += 1
        # "Never queue" mode still goes through acquire() to keep the permit
        # accounting honest; the deadline is just short enough to succeed only when a
        # permit is already free.
        timeout = 0.001 if config.max_wait_seconds <= 0 else config.max_wait_seconds
        try:
            await _acquire_unless_abandoned(semaphore, timeout, abandoned)
        except _ClientGone:
            waited = time.monotonic() - started
            stats.queued -= 1
            stats.abandoned += 1
            # Worth logging on its own: callers giving up while queued is the signal
            # that the deadline is longer than they are willing to wait.
            logger.info(
                "admission abandoned: lane=%s waited=%.3fs limit=%d queued=%d",
                operation,
                waited,
                config.max_in_flight,
                stats.queued,
            )
            raise AdmissionAbandoned(operation) from None
        except (TimeoutError, asyncio.TimeoutError):
            waited = time.monotonic() - started
            stats.rejected += 1
            stats.total_wait_seconds += waited
            logger.warning(
                "admission refused: lane=%s waited=%.3fs limit=%d in_flight=%d queued=%d",
                operation,
                waited,
                config.max_in_flight,
                stats.in_flight,
                stats.queued,
            )
            raise AdmissionRejected(operation, waited, config.max_in_flight) from None
        finally:
            stats.queued -= 1

        waited = time.monotonic() - started
        stats.admitted += 1
        stats.total_wait_seconds += waited
        stats.in_flight += 1
        try:
            yield
        finally:
            stats.in_flight -= 1
            semaphore.release()


def build_controller_from_config(config) -> AdmissionController:
    """Build the controller from ``HindsightConfig``.

    Only the three high-volume operations get a lane. `files_retain`,
    `dry_run_extract` and the mental-model routes are administrative and low-volume:
    gating them would add knobs nobody tunes without protecting anything that
    actually saturates a worker. They fall through ungated, exactly as before.

    A lane resolving to 0 in-flight is off (set its env var negative).
    """
    return AdmissionController(
        {
            "recall": LaneConfig(config.admission_recall_max_in_flight, config.admission_recall_max_wait_seconds),
            "reflect": LaneConfig(config.admission_reflect_max_in_flight, config.admission_reflect_max_wait_seconds),
            "retain": LaneConfig(config.admission_retain_max_in_flight, config.admission_retain_max_wait_seconds),
        }
    )

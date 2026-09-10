"""Event-loop lag probe: how long a ready coroutine waits before it runs.

Every per-phase timer in the recall path measures its own await, so a request that is *runnable*
but not *running* is invisible to all of them — the phases stay fast and the total inflates, which
reads as unaccounted time in some uninstrumented I/O that does not exist. Measured on this API,
the instrumented phases covered 10% of a recall's wall time and no candidate I/O accounted for the
other 90%.

This distinguishes the two cases directly. The probe sleeps for a known interval and reports how
much longer than that it actually took. That overshoot is loop lag: time the loop spent running
other callbacks (or blocked in a synchronous call) while this one was ready. If lag is ~0 while
requests are slow, the time is in a real await and the phases are missing one; if lag tracks
request latency, the loop is oversubscribed and no amount of I/O tuning helps.

Enabled by HINDSIGHT_API_LOOP_LAG_REPORT_SECONDS (seconds between reports); 0 means the task never
starts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

logger = logging.getLogger(__name__)

#: Short enough that a report describes a moment rather than an average over a whole run, long
#: enough that the probe itself is not a meaningful share of the loop's work.
_TICK_S = 0.05

#: Floor on the report interval, so a typo like `0.01` does not turn the probe into log spam.
_MIN_REPORT_S = 1.0

# The loop only keeps a weak reference to a task, so an unreferenced one can be garbage-collected
# mid-run and the probe would silently stop reporting.
_tasks: set[asyncio.Task[None]] = set()


def _percentile(sorted_lags: list[float], p: float) -> float:
    return sorted_lags[min(len(sorted_lags) - 1, int(len(sorted_lags) * p / 100))]


async def _run(report_every: float) -> None:
    pid = os.getpid()
    while True:
        lags: list[float] = []
        deadline = time.monotonic() + report_every
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            await asyncio.sleep(_TICK_S)
            lags.append((time.monotonic() - t0 - _TICK_S) * 1000.0)
        lags.sort()
        logger.info(
            "[loop-lag] pid=%d n=%d p50=%.1fms p90=%.1fms p99=%.1fms max=%.1fms",
            pid,
            len(lags),
            _percentile(lags, 50),
            _percentile(lags, 90),
            _percentile(lags, 99),
            lags[-1],
        )


def install(report_every: float) -> asyncio.Task[None] | None:
    """Start the probe on the running loop. No-op when `report_every` is 0 (the default)."""
    if report_every <= 0:
        return None
    report_every = max(_MIN_REPORT_S, report_every)
    task = asyncio.get_running_loop().create_task(_run(report_every))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    logger.info("[loop-lag] armed: reporting every %.0fs", report_every)
    return task

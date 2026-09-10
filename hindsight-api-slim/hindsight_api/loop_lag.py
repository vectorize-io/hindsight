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

Enabled by HINDSIGHT_API_LOOP_LAG (seconds between reports); unset means the task never starts.
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


def _interval() -> float | None:
    raw = os.getenv("HINDSIGHT_API_LOOP_LAG")
    if not raw:
        return None
    try:
        return max(1.0, float(raw))
    except ValueError:
        logger.warning("[loop-lag] ignoring unparseable HINDSIGHT_API_LOOP_LAG=%r", raw)
        return None


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
        n = len(lags)
        q = lambda p: lags[min(n - 1, int(n * p / 100))]  # noqa: E731
        logger.info(
            "[loop-lag] pid=%d n=%d p50=%.1fms p90=%.1fms p99=%.1fms max=%.1fms",
            pid,
            n,
            q(50),
            q(90),
            q(99),
            lags[-1],
        )


def install() -> bool:
    """Start the probe on the running loop. No-op unless the env var asks for it."""
    every = _interval()
    if every is None:
        return False
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # Called before the loop exists (import time); the caller retries from a startup hook.
        return False
    asyncio.ensure_future(_run(every))
    logger.info("[loop-lag] armed: reporting every %.0fs", every)
    return True

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

HINDSIGHT_API_LOOP_LAG_REPORT_SECONDS sets the seconds between log reports (0: no reports), and
HINDSIGHT_API_LOOP_LAG_METRIC records every sample in the ``hindsight.event_loop.lag`` histogram.
With neither set the task never starts.
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


async def _run(report_every: float, *, record_metric: bool) -> None:
    from hindsight_api.metrics import get_metrics_collector

    pid = os.getpid()
    # Without reports the window only bounds how long `lags` grows before it is dropped.
    window = report_every if report_every > 0 else 10.0
    while True:
        lags: list[float] = []
        deadline = time.monotonic() + window
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            await asyncio.sleep(_TICK_S)
            lag_s = max(0.0, time.monotonic() - t0 - _TICK_S)
            lags.append(lag_s * 1000.0)
            if record_metric:
                # Looked up per sample: the API lifespan installs the real collector after the probe
                # starts, and a test may swap it.
                get_metrics_collector().record_loop_lag(lag_s)
        if report_every <= 0:
            continue
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


def install(report_every: float, *, metric: bool = False) -> asyncio.Task[None] | None:
    """Start the probe on the running loop.

    No-op when neither log reports (`report_every` > 0) nor the histogram (`metric`) is enabled,
    which is the default.
    """
    if report_every <= 0 and not metric:
        return None
    if report_every > 0:
        report_every = max(_MIN_REPORT_S, report_every)
    task = asyncio.get_running_loop().create_task(_run(report_every, record_metric=metric))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    logger.info("[loop-lag] armed: reports every %.0fs, histogram %s", report_every, "on" if metric else "off")
    return task

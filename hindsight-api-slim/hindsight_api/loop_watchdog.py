"""Event-loop stall watchdog.

Hindsight's worker and API run the ``/health`` handler and all task work on one
asyncio event loop. If something does blocking (synchronous) work on that loop —
CPU-bound parsing, a mis-offloaded SDK call, a third-party library that signs a
request inline — the loop stops servicing coroutines, ``/health`` can't be
scheduled, and a Kubernetes liveness probe fails even though the process is "up".

This watchdog makes that condition self-diagnosing. It runs in a **separate OS
thread** (deliberately: a coroutine-based monitor would be frozen by the very
stall it's trying to observe), pings the loop, and when the loop fails to service
the ping within a threshold it logs the loop thread's current stack — naming the
exact frame that is blocking. It never raises and never touches the loop's work;
it only observes. Unlike monkeypatch-based blocking detectors it works with
uvloop, because it relies only on ``loop.call_soon_threadsafe`` and
``sys._current_frames()``.

It is the loop-side counterpart to the DB-pool acquire instrumentation
(``engine/db/pool_instrumentation.py``): together they let a stuck ``/health`` be
attributed to either a blocked loop or connection-pool exhaustion from the logs
alone.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import traceback
from collections.abc import Callable

logger = logging.getLogger("hindsight.loop_watchdog")


def start_loop_watchdog(loop) -> "LoopWatchdog | None":
    """Build and start a watchdog for ``loop`` from config, or return None if disabled.

    Call this once, from inside the running loop's process (worker CLI / API lifespan),
    and call ``.stop()`` on the returned handle at shutdown.
    """
    from .config import get_config

    config = get_config()
    if not config.loop_watchdog_enabled:
        return None
    watchdog = LoopWatchdog(
        loop,
        stall_threshold_s=config.loop_watchdog_stall_threshold_ms / 1000.0,
        poll_interval_s=config.loop_watchdog_poll_interval_ms / 1000.0,
    )
    watchdog.start()
    return watchdog


class LoopWatchdog:
    """Detects event-loop stalls from an off-loop thread and logs the culprit stack.

    Args:
        loop: the asyncio event loop to monitor.
        stall_threshold_s: log when the loop takes at least this long to service a ping.
        poll_interval_s: how often to ping the loop.
        on_stall: optional callback ``(blocked_for_s, stack_text)`` invoked on each
            detected stall instead of the default log+metric path. Used for testing.
    """

    def __init__(
        self,
        loop,
        *,
        stall_threshold_s: float = 1.0,
        poll_interval_s: float = 0.25,
        on_stall: Callable[[float, str], None] | None = None,
    ) -> None:
        self._loop = loop
        self._stall_threshold_s = stall_threshold_s
        self._poll_interval_s = poll_interval_s
        self._on_stall = on_stall
        self._stop = threading.Event()
        self._loop_thread_id: int | None = None
        self._thread = threading.Thread(target=self._run, name="loop-watchdog", daemon=True)
        self._started = False

    def start(self) -> None:
        """Start monitoring. Must not block the loop — the id is captured via pings.

        When called from the loop thread itself (the normal case: worker ``run()`` /
        API lifespan), ``threading.get_ident()`` is already the loop thread id, so we
        seed it here; each ping then re-affirms it authoritatively. We deliberately do
        NOT schedule-and-wait for a callback: that would deadlock, because the loop
        can't run the callback while ``start()`` is blocking it.
        """
        self._loop_thread_id = threading.get_ident()
        self._started = True
        self._thread.start()
        logger.info(
            "Loop watchdog started (stall_threshold=%.2fs, poll_interval=%.2fs)",
            self._stall_threshold_s,
            self._poll_interval_s,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._started and self._thread.is_alive():
            self._thread.join(timeout=self._poll_interval_s + self._stall_threshold_s + 1.0)

    def _run(self) -> None:
        while not self._stop.wait(self._poll_interval_s):
            serviced = threading.Event()
            sent_at = time.monotonic()

            def _ping() -> None:
                # Runs on the loop thread — capture its id authoritatively, then
                # signal that the loop serviced this ping.
                self._loop_thread_id = threading.get_ident()
                serviced.set()

            try:
                self._loop.call_soon_threadsafe(_ping)
            except RuntimeError:
                return  # loop closed — nothing left to watch
            if not serviced.wait(self._stall_threshold_s):
                self._report(sent_at)
                # Block until the loop finally services the ping so we emit one
                # report per stall, not one per poll while it stays blocked.
                serviced.wait()

    def _report(self, sent_at: float) -> None:
        frame = sys._current_frames().get(self._loop_thread_id or -1)
        stack = "".join(traceback.format_stack(frame)) if frame is not None else "<loop-thread frame unavailable>"
        blocked_for = time.monotonic() - sent_at
        if self._on_stall is not None:
            self._on_stall(blocked_for, stack)
            return
        logger.warning(
            "EVENT LOOP BLOCKED for >= %.2fs (%.2fs and counting). The loop is not "
            "servicing coroutines — /health cannot be scheduled. Blocking frame:\n%s",
            self._stall_threshold_s,
            blocked_for,
            stack,
        )
        try:
            from .metrics import get_metrics_collector

            get_metrics_collector().record_loop_stall(blocked_for)
        except Exception:
            pass

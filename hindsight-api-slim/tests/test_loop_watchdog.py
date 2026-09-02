"""Unit tests for the event-loop stall watchdog.

Deterministic (no LLM): we block the loop with a synchronous sleep and assert the
off-loop watchdog thread detects it and captures the culprit stack, and that
genuinely off-loop work does not trip it.
"""

import asyncio
import threading
import time

from hindsight_api.loop_watchdog import LoopWatchdog


async def test_watchdog_detects_on_loop_block():
    stalls: list[tuple[float, str]] = []
    wd = LoopWatchdog(
        asyncio.get_running_loop(),
        stall_threshold_s=0.1,
        poll_interval_s=0.02,
        on_stall=lambda dur, stack: stalls.append((dur, stack)),
    )
    wd.start()
    try:
        await asyncio.sleep(0.1)  # let the watchdog settle into steady polling
        time.sleep(0.6)  # BLOCK the event loop synchronously
        await asyncio.sleep(0.2)  # give the watchdog a chance to have reported
    finally:
        wd.stop()

    assert stalls, "watchdog did not detect the synchronous loop block"
    blocked_for, stack = stalls[0]
    assert blocked_for >= 0.1
    # The captured stack must name the frame that was blocking the loop.
    assert "test_watchdog_detects_on_loop_block" in stack


async def test_watchdog_ignores_offloop_work():
    stalls: list[tuple[float, str]] = []
    # 0.3s, not the 0.1s this used to use: the assertion is "no stall", so the threshold
    # is also the margin against ordinary scheduling jitter, and 0.1s is inside what a CI
    # runner hosting eight xdist workers hands out. It must still stay BELOW the 0.5s of
    # offloaded work below — at a threshold above it the test would pass even if that work
    # ran on the loop, which is the whole thing it exists to catch.
    wd = LoopWatchdog(
        asyncio.get_running_loop(),
        stall_threshold_s=0.3,
        poll_interval_s=0.02,
        on_stall=lambda dur, stack: stalls.append((dur, stack)),
    )
    wd.start()
    try:
        await asyncio.sleep(0.1)
        # Sync sleep offloaded to a thread — the loop stays free, exactly the
        # pattern litellm uses for boto3 credential resolution.
        await asyncio.get_running_loop().run_in_executor(None, time.sleep, 0.5)
        await asyncio.sleep(0.1)
    finally:
        wd.stop()

    assert not stalls, f"watchdog falsely reported a stall for off-loop work: {stalls}"


async def test_watchdog_quiet_when_loop_responsive():
    stalls: list[tuple[float, str]] = []
    # 1.0s for the same reason as above, and here nothing caps it: the loop below only
    # ever awaits, so any threshold far above one iteration proves the point. At 0.1s this
    # was asserting that the host never hiccups for 100ms, which a loaded CI runner does.
    wd = LoopWatchdog(
        asyncio.get_running_loop(),
        stall_threshold_s=1.0,
        poll_interval_s=0.02,
        on_stall=lambda dur, stack: stalls.append((dur, stack)),
    )
    wd.start()
    try:
        for _ in range(10):
            await asyncio.sleep(0.03)
    finally:
        wd.stop()

    assert not stalls, f"watchdog reported a stall on a responsive loop: {stalls}"


async def test_watchdog_stop_does_not_report_its_own_join_as_a_stall():
    """Regression: ``stop()`` must not be reported as the stall it causes.

    ``stop()`` runs on the loop thread and joins the watchdog thread, blocking the loop
    for up to ``poll_interval + stall_threshold + 1s``. A ping already in flight when
    that happens can never be serviced, so the watchdog used to report a stall whose
    named culprit was ``watchdog.stop()`` itself — a spurious "EVENT LOOP BLOCKED"
    warning at every clean shutdown, landing precisely when someone is reading the logs
    to diagnose one. It also made the two "no stall" tests above fail ~5% of runs
    (measured 11/200 locally; seen on CI shard 2).

    Reproduced deterministically rather than by retrying a responsive loop until the
    race lands. ``stop()`` is exactly two things — set the flag, then block the loop
    thread — so they are staged apart here: the sleep blocks the loop so the in-flight
    ping cannot be serviced, and the timer sets the flag partway through that block,
    inside the window between the ping going out and the threshold expiring. The
    watchdog therefore always reaches its report decision with the stop already
    requested, which is the state the assertion is about. Margins are wide (flag at
    ~0.02s, threshold 0.2s, block 0.5s) so the ordering does not depend on scheduling.
    """
    stalls: list[tuple[float, str]] = []
    wd = LoopWatchdog(
        asyncio.get_running_loop(),
        stall_threshold_s=0.2,
        poll_interval_s=0.01,
        on_stall=lambda dur, stack: stalls.append((dur, stack)),
    )
    wd.start()
    await asyncio.sleep(0.05)  # let the watchdog settle into steady polling
    threading.Timer(0.02, wd._stop.set).start()
    time.sleep(0.5)  # the loop thread is blocked, exactly as stop()'s join() blocks it
    wd.stop()

    assert not stalls, f"watchdog reported a stall caused by its own shutdown: {stalls}"

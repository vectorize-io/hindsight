"""Unit tests for the event-loop stall watchdog.

Deterministic (no LLM): we block the loop with a synchronous sleep and assert the
off-loop watchdog thread detects it and captures the culprit stack, and that
genuinely off-loop work does not trip it.
"""

import asyncio
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
    wd = LoopWatchdog(
        asyncio.get_running_loop(),
        stall_threshold_s=0.1,
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
    wd = LoopWatchdog(
        asyncio.get_running_loop(),
        stall_threshold_s=0.1,
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

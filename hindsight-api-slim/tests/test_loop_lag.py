"""The event-loop lag probe starts only when configured, and survives without a caller reference."""

import asyncio
import logging

import pytest

from hindsight_api import loop_lag


async def test_disabled_by_default() -> None:
    assert loop_lag.install(0) is None
    assert not loop_lag._tasks


async def test_reports_lag_percentiles(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(loop_lag, "_MIN_REPORT_S", 0.0)
    monkeypatch.setattr(loop_lag, "_TICK_S", 0.001)
    caplog.set_level(logging.INFO, logger=loop_lag.__name__)

    task = loop_lag.install(0.02)
    assert task is not None
    # Held by the module, so the loop's weak reference is not the only one keeping it alive.
    assert task in loop_lag._tasks
    try:
        for _ in range(100):
            if any("p99=" in r.getMessage() for r in caplog.records):
                break
            await asyncio.sleep(0.01)
        assert any("[loop-lag]" in r.getMessage() and "p99=" in r.getMessage() for r in caplog.records)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    await asyncio.sleep(0)  # done callbacks run on the next loop iteration
    assert task not in loop_lag._tasks

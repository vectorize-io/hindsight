"""Time queued for an LLM permit is kept out of the time limits that opt in (#6140)."""

import asyncio

import pytest

from hindsight_api.engine.permit_wait import excluding_permit_wait, waiting_for_permit


async def _queue(seconds: float) -> None:
    with waiting_for_permit():
        await asyncio.sleep(seconds)


@pytest.mark.asyncio
async def test_a_wait_longer_than_the_limit_is_not_charged():
    async with asyncio.timeout(0.3) as limit:
        with excluding_permit_wait(limit, max_wait=5):
            await _queue(0.5)
            await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_parallel_queued_calls_hold_the_clock_once():
    """Five calls queued side by side waited 0.3s of wall time, not 1.5s."""
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.2) as limit:
            with excluding_permit_wait(limit, max_wait=5):
                await asyncio.gather(*(_queue(0.3) for _ in range(5)))
                await asyncio.sleep(1.0)


@pytest.mark.asyncio
async def test_work_still_counts():
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.2) as limit:
            with excluding_permit_wait(limit, max_wait=5):
                await _queue(0.05)
                await asyncio.sleep(0.5)


@pytest.mark.asyncio
async def test_a_wait_past_max_wait_still_runs_the_clock_out():
    """A permit that never frees is a wedge, not a queue."""
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.1) as limit:
            with excluding_permit_wait(limit, max_wait=0.2):
                await _queue(5)


@pytest.mark.asyncio
async def test_a_wait_outside_any_limit_is_a_no_op():
    await _queue(0.01)

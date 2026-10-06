"""Shared fire-and-forget write bookkeeping (no DB)."""

import asyncio

import pytest

from hindsight_api.engine.background_writes import PendingWrites


@pytest.mark.asyncio
async def test_drain_waits_for_its_own_key_only():
    writes = PendingWrites("test write")
    done: list[str] = []
    other_started = asyncio.Event()

    async def write(name: str, delay: float) -> None:
        if name == "other":
            other_started.set()
        await asyncio.sleep(delay)
        done.append(name)

    writes.schedule(write("mine", 0), "trace-a")
    writes.schedule(write("other", 60), "trace-b")
    await other_started.wait()

    await asyncio.wait_for(writes.drain("trace-a"), timeout=1)

    assert done == ["mine"], "drain must not block on another key's write"

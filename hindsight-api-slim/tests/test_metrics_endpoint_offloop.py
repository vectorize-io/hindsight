"""Regression test: the Prometheus /metrics endpoint must render off the event loop.

``prometheus_client.generate_latest()`` (and the multi-worker ``WorkerMetrics.render``,
which additionally does file I/O) is synchronous, and its cost scales with the size of the
metric registry. On a large registry it can take seconds to serialize. If ``/metrics``
awaits that render inline, the asyncio event loop is frozen for the whole duration, so
``/health``, WebSocket handshakes, and every other request the worker is handling stall
until the scrape completes. The endpoint offloads the render with ``asyncio.to_thread`` to
keep the loop free; this test proves a concurrent coroutine keeps making progress while the
(deliberately slow) render is in flight -- independent of what the registry actually holds.
"""

import asyncio
import contextlib
import time
from unittest.mock import MagicMock

import httpx
import pytest

from hindsight_api.api import create_app

# Slow enough that an inline (loop-blocking) render is unambiguously distinguishable from an
# off-loop one, but short enough to keep the test fast.
_BLOCK_S = 0.3
_TICK_S = 0.01


@pytest.mark.asyncio
async def test_metrics_endpoint_does_not_block_event_loop(monkeypatch):
    # /metrics never touches the memory engine, so a mock keeps this test infra-free.
    app = create_app(MagicMock(), initialize_memory=False)

    def blocking_render(*args, **kwargs):
        # Stand in for an expensive, synchronous scrape render on a large registry.
        time.sleep(_BLOCK_S)
        return b"# HELP up 1\nup 1\n"

    # metrics_endpoint imports generate_latest from prometheus_client at call time, so patching
    # the module attribute is what the endpoint resolves. Single-worker mode (app.state.worker_metrics
    # unset) takes the generate_latest branch.
    monkeypatch.setattr("prometheus_client.generate_latest", blocking_render)

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(_TICK_S)
            ticks += 1

    ticker_task = asyncio.create_task(ticker())
    await asyncio.sleep(0)  # let the ticker get scheduled before the request

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")

    # Snapshot progress made *while the request was in flight*, before we cancel the ticker
    # (awaiting it to completion would let it finish counting regardless of the fix).
    ticks_during_request = ticks
    ticker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await ticker_task

    assert response.status_code == 200
    # Off-loop: the ~0.3s render yields to the loop, so the 10ms ticker advances ~30 times.
    # Inline (the bug): the loop is frozen for the whole render and the ticker cannot advance at
    # all. A threshold well above 0 but well below the ~30 expected off-loop ticks separates them.
    expected_ticks = int(_BLOCK_S / _TICK_S)
    assert ticks_during_request >= expected_ticks // 2, (
        f"event loop appears blocked during /metrics render: only {ticks_during_request} "
        f"ticks in flight (expected >= {expected_ticks // 2})"
    )

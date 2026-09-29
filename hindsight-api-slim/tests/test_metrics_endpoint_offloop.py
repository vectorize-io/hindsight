"""Regression test: the Prometheus /metrics endpoints must render off the event loop.

``prometheus_client.generate_latest()`` (and the multi-worker ``WorkerMetrics.render``,
which additionally does file I/O) is synchronous, and its cost scales with the size of the
metric registry. On a large registry it can take seconds to serialize. If ``/metrics``
awaits that render inline, the asyncio event loop is frozen for the whole duration, so
``/health``, WebSocket handshakes, and every other request the worker is handling stall
until the scrape completes.

Both apps offload the render with ``asyncio.to_thread``. The test asserts the render ran on
a thread other than the loop's -- exact, and with no wall-clock threshold to go flaky under
a loaded CI box. (A timing-based version was tried first: it measured ~19 ticks idle and
~15 under load against a threshold of 15, i.e. it sat on its own boundary.)
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from hindsight_api.api import create_app


def _record_thread(seen: list[int]):
    """Stand in for the real renderer, recording which thread it ran on."""

    def render():
        seen.append(threading.get_ident())
        return b"# HELP up 1\nup 1\n"

    return render


async def _get_metrics(app) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get("/metrics")


@pytest.mark.asyncio
async def test_api_metrics_renders_off_the_loop(monkeypatch):
    # /metrics never touches the memory engine, so a mock keeps this test infra-free.
    app = create_app(MagicMock(), initialize_memory=False)

    seen: list[int] = []
    # metrics_endpoint imports generate_latest from prometheus_client at call time, so patching the
    # module attribute is what the endpoint resolves. Single-worker mode (app.state.worker_metrics
    # unset) takes the generate_latest branch.
    monkeypatch.setattr("prometheus_client.generate_latest", _record_thread(seen))

    response = await _get_metrics(app)

    assert response.status_code == 200
    assert seen and seen[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_api_multiworker_render_runs_off_the_loop():
    # Multi-worker mode takes the WorkerMetrics.render branch, which additionally does blocking
    # file I/O (per-worker snapshot files) -- the path that most needs to stay off the loop.
    app = create_app(MagicMock(), initialize_memory=False)

    seen: list[int] = []
    app.state.worker_metrics = SimpleNamespace(render=_record_thread(seen))

    response = await _get_metrics(app)

    assert response.status_code == 200
    assert seen and seen[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_worker_metrics_renders_off_the_loop(monkeypatch):
    from hindsight_api.worker.main import create_worker_app

    seen: list[int] = []
    # create_worker_app imports generate_latest from prometheus_client when it runs, so patch the
    # module attribute before building the app.
    monkeypatch.setattr("prometheus_client.generate_latest", _record_thread(seen))

    response = await _get_metrics(create_worker_app(MagicMock(), MagicMock()))

    assert response.status_code == 200
    assert seen and seen[0] != threading.get_ident()

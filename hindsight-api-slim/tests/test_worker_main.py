"""Tests for hindsight_api.worker.main entry-point helpers."""

import asyncio
import os
import signal
import sys
from unittest.mock import MagicMock

import pytest

from hindsight_api.worker.main import _http_server, _install_shutdown_signal_handlers


def test_install_shutdown_signal_handlers_unix_path():
    """On platforms where asyncio supports signal handlers (Unix), both
    SIGINT and SIGTERM are registered and the helper reports success."""
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    handler = MagicMock()

    installed = _install_shutdown_signal_handlers(loop, handler)

    assert installed is True
    loop.add_signal_handler.assert_any_call(signal.SIGINT, handler)
    loop.add_signal_handler.assert_any_call(signal.SIGTERM, handler)
    assert loop.add_signal_handler.call_count == 2


def test_install_shutdown_signal_handlers_windows_path():
    """On Windows, asyncio's ProactorEventLoop raises NotImplementedError
    from add_signal_handler. The helper must swallow it and report failure
    so the worker keeps running with default Python signal behavior
    (regression test for issue #1411)."""
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    loop.add_signal_handler.side_effect = NotImplementedError
    handler = MagicMock()

    installed = _install_shutdown_signal_handlers(loop, handler)

    assert installed is False


def test_main_bootstraps_tracing_for_the_worker_process(monkeypatch):
    """The standalone worker must initialize tracing itself.

    initialize_tracing() used to be called only from the FastAPI lifespan, so a
    `hindsight-worker` process emitted no spans at all — silently, since both
    tracing chokepoints degrade to no-ops (issue #3614). It also identifies
    itself as "hindsight-worker" by default, matching the name it already
    reports for metrics.
    """
    import dataclasses
    import sys

    from hindsight_api import tracing
    from hindsight_api.config import _get_raw_config
    from hindsight_api.worker import main as worker_main

    config = dataclasses.replace(_get_raw_config(), worker_id="test-worker")
    monkeypatch.setattr(config, "configure_logging", lambda: None)
    monkeypatch.setattr(worker_main, "get_config", lambda: config)
    monkeypatch.setattr(worker_main, "load_dotenv_for_entrypoint", lambda: None)
    monkeypatch.setattr(sys, "argv", ["hindsight-worker"])

    bootstrap_calls = []

    def _record(cfg, **kwargs):
        bootstrap_calls.append(kwargs)
        return False

    monkeypatch.setattr(tracing, "initialize_tracing_from_config", _record)
    # Stop before the worker actually runs; we only care about the bootstrap.
    monkeypatch.setattr(worker_main.asyncio, "run", lambda coro: coro.close())

    worker_main.main()

    assert bootstrap_calls == [{"default_service_name": "hindsight-worker"}]


def test_main_warns_on_retired_flags_and_keeps_env_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--max-retries/--log-level never took effect; they still parse but only warn."""
    import dataclasses
    import sys

    from hindsight_api import tracing
    from hindsight_api.config import _get_raw_config
    from hindsight_api.worker import main as worker_main

    config = dataclasses.replace(_get_raw_config(), worker_id="test-worker", worker_max_retries=7)
    monkeypatch.setattr(config, "configure_logging", lambda: None)
    monkeypatch.setattr(worker_main, "get_config", lambda: config)
    monkeypatch.setattr(worker_main, "load_dotenv_for_entrypoint", lambda: None)
    monkeypatch.setattr(sys, "argv", ["hindsight-worker", "--max-retries", "2", "--log-level", "debug"])
    monkeypatch.setattr(tracing, "initialize_tracing_from_config", lambda *args, **kwargs: False)
    monkeypatch.setattr(worker_main.asyncio, "run", lambda coro: coro.close())

    worker_main.main()

    out = capsys.readouterr()
    assert "--max-retries 2 is ignored: set HINDSIGHT_API_WORKER_MAX_RETRIES instead." in out.err
    assert "--log-level debug is ignored: set HINDSIGHT_API_LOG_LEVEL instead." in out.err
    assert "Max retries: 7" in out.out
    assert config.worker_max_retries == 7


@pytest.mark.skipif(not hasattr(signal, "SIGTERM") or sys.platform == "win32", reason="Unix signals only")
async def test_one_sigterm_reaches_worker_handler_once():
    """One SIGTERM must call the worker's handler once. Uvicorn's own signal capture
    re-raised it after the server stopped, so the handler saw a "second signal" and
    force-exited before releasing its claimed tasks (#5611)."""
    loop = asyncio.get_running_loop()
    calls = []
    loop.add_signal_handler(signal.SIGTERM, lambda: calls.append(1))
    try:

        async def app(scope, receive, send):
            pass

        server = _http_server(app, "127.0.0.1", 0)
        serve_task = asyncio.create_task(server.serve())

        async def _started() -> None:
            while not server.started:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_started(), timeout=5)
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.sleep(0.2)
        server.should_exit = True  # the worker stops the server itself on shutdown
        await asyncio.wait_for(serve_task, timeout=5)
        await asyncio.sleep(0.2)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
    assert calls == [1]

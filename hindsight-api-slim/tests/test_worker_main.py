"""Tests for hindsight_api.worker.main entry-point helpers."""

import asyncio
import signal
from unittest.mock import MagicMock

from hindsight_api.worker.main import _install_shutdown_signal_handlers


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

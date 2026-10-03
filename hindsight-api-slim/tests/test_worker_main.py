"""Tests for hindsight_api.worker.main entry-point helpers."""

import asyncio
import dataclasses
import logging
import signal
import sys
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

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
    monkeypatch.setattr(worker_main, "_get_raw_config", lambda: config)
    monkeypatch.setattr(worker_main, "load_dotenv_for_entrypoint", lambda: None)
    monkeypatch.setattr(sys, "argv", ["hindsight-worker"])

    bootstrap_calls = []

    def _record(cfg, **kwargs):
        bootstrap_calls.append(kwargs)
        return False

    monkeypatch.setattr(tracing, "initialize_tracing_from_config", _record)
    # Stop before the worker actually runs; we only care about the bootstrap.
    import hindsight_api

    monkeypatch.setitem(hindsight_api.__dict__, "MemoryEngine", MagicMock())
    monkeypatch.setattr(worker_main.asyncio, "run", lambda coro: coro.close())

    worker_main.main()

    assert bootstrap_calls == [{"default_service_name": "hindsight-worker"}]


@pytest.mark.parametrize(
    "cli_args, expected_retries, expected_level",
    [
        ([], 7, logging.WARNING),
        (["--log-level", "debug"], 7, logging.DEBUG),
        (["--max-retries", "2", "--log-level", "debug"], 2, logging.DEBUG),
        (["--max-retries", "0", "--log-level", "error"], 0, logging.ERROR),
    ],
)
def test_main_applies_cli_overrides_to_poller_and_shared_config(
    monkeypatch: pytest.MonkeyPatch,
    cli_args: list[str],
    expected_retries: int,
    expected_level: int,
) -> None:
    """The displayed retry budget must also govern the poller and engine retries."""
    import hindsight_api
    import hindsight_api.extensions as extensions
    from hindsight_api import config as config_module
    from hindsight_api import tracing
    from hindsight_api.config import HindsightConfig, get_config
    from hindsight_api.worker import main as worker_main

    @dataclass
    class Backend:
        supports_worker_poller: bool = True

    @dataclass
    class Engine:
        _backend: Backend = field(default_factory=Backend)
        _pg0: None = None

        async def initialize(self) -> None:
            pass

        def _require_backend(self) -> Backend:
            return self._backend

        async def execute_task(self, task: Any) -> None:
            pass

        async def on_task_wall_timeout(self, task: Any, schema: str | None, message: str) -> None:
            pass

    class StartedPoller(Exception):
        pass

    engine = Engine()

    def create_engine(**kwargs: Any) -> Engine:
        return engine

    observed_retries: list[int] = []

    def create_poller(*, max_retries: int, **kwargs: Any) -> None:
        observed_retries.append(max_retries)
        # MemoryEngine.execute_task obtains its retry budget from this same cache.
        assert get_config().worker_max_retries == expected_retries
        raise StartedPoller

    config = dataclasses.replace(HindsightConfig.from_env(), worker_max_retries=7, log_level="warning")
    monkeypatch.setattr(config_module, "_config_cache", config)
    monkeypatch.setattr(worker_main, "load_dotenv_for_entrypoint", lambda: None)
    monkeypatch.setattr(sys, "argv", ["hindsight-worker", *cli_args])
    monkeypatch.setitem(hindsight_api.__dict__, "MemoryEngine", create_engine)
    monkeypatch.setattr(worker_main, "WorkerPoller", create_poller)
    monkeypatch.setattr(extensions, "load_extension", lambda *args: None)
    monkeypatch.setattr(tracing, "initialize_tracing_from_config", lambda *args, **kwargs: False)
    monkeypatch.setattr(worker_main.atexit, "register", lambda callback: None)

    root_logger = logging.getLogger()
    previous_handlers = root_logger.handlers[:]
    previous_level = root_logger.level
    try:
        with pytest.raises(StartedPoller):
            worker_main.main()
        assert observed_retries == [expected_retries]
        assert root_logger.level == expected_level
    finally:
        for handler in root_logger.handlers:
            if handler not in previous_handlers:
                handler.close()
        root_logger.handlers = previous_handlers
        root_logger.setLevel(previous_level)

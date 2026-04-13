"""Tests for lib/daemon.py startup coordination.

Regression coverage for the April-2026 incident where two Codex hooks raced
on `hindsight-embed daemon start`, one lost the port bind, and its shutdown
cleanup stopped the shared embedded Postgres — leaving the survivor stuck
on connection-refused errors.
"""

import threading
import time

import pytest

from lib import daemon, state


@pytest.mark.skipif(state.fcntl is None, reason="startup lock coordination uses fcntl on Unix")
def test_get_api_url_waits_for_inflight_daemon_start(monkeypatch, tmp_path):
    """A second hook should wait for the first daemon start instead of racing it.

    We simulate the race by grabbing the startup lock on a helper thread,
    then calling get_api_url() from the main thread. flock on Linux/macOS
    is scoped to the open-file-description, not the process — because the
    helper thread and the daemon code each call open() separately, each gets
    its own OFD and flock blocks across them even though they share a PID.
    """
    monkeypatch.setenv("HOME", str(tmp_path))

    config = {
        "apiPort": 9077,
        "llmProvider": "openai-codex",
        "llmModel": "gpt-5.2-codex",
    }

    health_state = {"ready": False}
    lock_path = state.state_file_path(daemon.DAEMON_START_LOCK_FILE)
    lock_ready = threading.Event()
    release_lock = threading.Event()

    def hold_startup_lock():
        with open(lock_path, "w") as lock_fd:
            state.fcntl.flock(lock_fd, state.fcntl.LOCK_EX)
            lock_ready.set()
            release_lock.wait(timeout=5)
            state.fcntl.flock(lock_fd, state.fcntl.LOCK_UN)

    holder = threading.Thread(target=hold_startup_lock, daemon=True)
    holder.start()
    assert lock_ready.wait(timeout=1), "test setup failed to acquire daemon lock"

    def check_health(_base_url, timeout=2):
        return health_state["ready"]

    monkeypatch.setattr(daemon, "_check_health", check_health)
    monkeypatch.setattr(daemon, "_is_embed_available", lambda _config: True)
    monkeypatch.setattr(
        daemon,
        "detect_llm_config",
        lambda _config: (_ for _ in ()).throw(AssertionError("should not detect llm")),
    )
    monkeypatch.setattr(
        daemon,
        "_run_embed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not start second daemon")),
    )
    monkeypatch.setattr(daemon, "write_state", lambda *args, **kwargs: None)

    def mark_ready():
        time.sleep(0.1)
        health_state["ready"] = True
        release_lock.set()

    releaser = threading.Thread(target=mark_ready, daemon=True)
    releaser.start()

    api_url = daemon.get_api_url(config, allow_daemon_start=True)

    holder.join(timeout=2)
    releaser.join(timeout=2)
    assert api_url == "http://127.0.0.1:9077"


def test_ensure_daemon_running_times_out_then_recovers_via_health(monkeypatch, tmp_path):
    """If lock acquisition times out but the daemon is now healthy, caller succeeds.

    Guards against a regression where a timeout-then-hard-fail would race
    a second daemon start after the holder's slow-but-successful startup.
    """
    monkeypatch.setenv("HOME", str(tmp_path))

    from contextlib import contextmanager

    @contextmanager
    def fake_lock(*_args, **_kwargs):
        raise TimeoutError("simulated slow holder")
        yield  # pragma: no cover

    monkeypatch.setattr(daemon, "acquire_state_lock", fake_lock)
    monkeypatch.setattr(daemon, "_check_health", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        daemon,
        "_run_embed",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not start daemon after timeout")),
    )

    daemon._ensure_daemon_running({"apiPort": 9077}, 9077)


def test_ensure_daemon_running_times_out_and_raises_when_unhealthy(monkeypatch, tmp_path):
    """If lock times out and daemon is still unhealthy, surface a clear error."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from contextlib import contextmanager

    @contextmanager
    def fake_lock(*_args, **_kwargs):
        raise TimeoutError("simulated slow holder")
        yield  # pragma: no cover

    monkeypatch.setattr(daemon, "acquire_state_lock", fake_lock)
    monkeypatch.setattr(daemon, "_check_health", lambda *_a, **_kw: False)

    with pytest.raises(RuntimeError, match="Timed out"):
        daemon._ensure_daemon_running({"apiPort": 9077}, 9077)

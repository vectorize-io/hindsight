"""Tests for lib/daemon.py startup coordination."""

import threading
import time

import pytest

from lib import daemon, state


@pytest.mark.skipif(
    state.fcntl is None, reason="startup lock coordination uses fcntl on Unix"
)
def test_get_api_url_waits_for_inflight_daemon_start(monkeypatch, tmp_path):
    """A second hook should wait for the first daemon start instead of racing it."""
    monkeypatch.setenv("HOME", str(tmp_path))

    config = {
        "apiPort": 9077,
        "llmProvider": "openai-codex",
        "llmModel": "gpt-5.2-codex",
    }

    health_state = {"ready": False}
    lock_path = state._state_file("daemon-start.lock")
    lock_ready = threading.Event()
    release_lock = threading.Event()

    def hold_startup_lock():
        with open(lock_path, "w") as lock_fd:
            state.fcntl.flock(lock_fd, state.fcntl.LOCK_EX)
            lock_ready.set()
            release_lock.wait(timeout=2)
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
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("should not start second daemon")
        ),
    )
    monkeypatch.setattr(daemon, "write_state", lambda *args, **kwargs: None)

    def mark_ready():
        time.sleep(0.1)
        health_state["ready"] = True
        release_lock.set()

    releaser = threading.Thread(target=mark_ready, daemon=True)
    releaser.start()

    api_url = daemon.get_api_url(config, allow_daemon_start=True)

    holder.join(timeout=1)
    releaser.join(timeout=1)
    assert api_url == "http://127.0.0.1:9077"

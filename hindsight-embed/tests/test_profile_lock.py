"""Tests for the cross-platform profile lock (regression coverage for #3100).

The Windows branch used to call `msvcrt.locking(..., LK_LOCK, 1)`, a blocking
lock that retries exactly 10 times internally and then raises. Two processes
starting the same profile concurrently made the loser fail with an opaque
OSError. Both platforms now drive the non-blocking primitive from one bounded
retry loop.
"""

import os
import threading
import time

import pytest

from hindsight_embed import profile_manager
from hindsight_embed.profile_manager import (
    ENV_LOCK_TIMEOUT,
    ProfileLockTimeout,
    lock_file,
    unlock_file,
)


def test_lock_is_exclusive_and_reacquirable(tmp_path):
    """A released lock can be taken by the next caller."""
    lock_path = tmp_path / "daemon.lock"

    with open(lock_path, "w") as first:
        lock_file(first, timeout=1)
        unlock_file(first)

    with open(lock_path, "w") as second:
        lock_file(second, timeout=1)
        unlock_file(second)


def test_contended_lock_times_out_instead_of_blocking(tmp_path):
    """The loser gets a bounded, explicit timeout — not an unbounded block."""
    lock_path = tmp_path / "daemon.lock"

    with open(lock_path, "w") as holder:
        lock_file(holder, timeout=1)
        try:
            started = time.monotonic()
            with open(lock_path, "w") as waiter:
                with pytest.raises(ProfileLockTimeout) as excinfo:
                    lock_file(waiter, timeout=0.2)
            elapsed = time.monotonic() - started
            assert elapsed < 5  # bounded wait, not a hang
            message = str(excinfo.value)
            assert str(lock_path) in message  # names the lock file
            assert f"PID {os.getpid()}" in message  # names the holder
            assert ENV_LOCK_TIMEOUT in message  # says how to wait longer
        finally:
            unlock_file(holder)


def test_waiter_acquires_once_the_holder_releases(tmp_path):
    """A concurrent start that waits its turn succeeds instead of erroring out."""
    lock_path = tmp_path / "daemon.lock"
    acquired = threading.Event()

    with open(lock_path, "w") as holder:
        lock_file(holder, timeout=1)

        def waiter():
            with open(lock_path, "w") as f:
                lock_file(f, timeout=10)
                acquired.set()
                unlock_file(f)

        thread = threading.Thread(target=waiter)
        thread.start()
        assert not acquired.wait(timeout=0.3)  # still blocked by the holder
        unlock_file(holder)
        thread.join(timeout=10)

    assert acquired.is_set()


def test_owner_sidecar_is_cleaned_up_on_release(tmp_path):
    """The holder receipt must not outlive the lock it describes."""
    lock_path = tmp_path / "daemon.lock"
    owner_path = tmp_path / "daemon.lock.owner"

    with open(lock_path, "w") as holder:
        lock_file(holder, timeout=1)
        assert owner_path.read_text() == str(os.getpid())
        unlock_file(holder)

    assert not owner_path.exists()


def test_invalid_timeout_env_falls_back_to_default(monkeypatch):
    """A malformed HINDSIGHT_EMBED_LOCK_TIMEOUT must not disable the bound."""
    for bad in ("not-a-number", "0", "-5", "inf"):
        monkeypatch.setenv(ENV_LOCK_TIMEOUT, bad)
        assert profile_manager._default_lock_timeout() == profile_manager.DEFAULT_LOCK_TIMEOUT

    monkeypatch.setenv(ENV_LOCK_TIMEOUT, "12.5")
    assert profile_manager._default_lock_timeout() == 12.5


def test_delete_profile_removes_the_lock_owner_sidecar(tmp_path, monkeypatch):
    """A crash while holding the lock leaves a sidecar; delete must not orphan it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    manager = profile_manager.ProfileManager()
    manager.create_profile("doomed", {"HINDSIGHT_API_LLM_PROVIDER": "openai"})

    profiles_dir = tmp_path / ".hindsight" / "profiles"
    lock_path = profiles_dir / "doomed.lock"
    owner_path = profiles_dir / "doomed.lock.owner"
    lock_path.write_text("")
    owner_path.write_text("4242")  # left behind by a crashed holder

    manager.delete_profile("doomed")

    assert not lock_path.exists()
    assert not owner_path.exists()

"""Tests for cross-process Codex state primitives."""

import stat
from concurrent.futures import ThreadPoolExecutor

import pytest
from lib import state


def test_pending_retain_names_do_not_collide_after_filename_sanitization(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    first = {"idempotency_key": "first", "bank_id": "bank", "request": {}}
    second = {"idempotency_key": "second", "bank_id": "bank", "request": {}}

    state.write_pending_retain("session/a", first)
    state.write_pending_retain("session:a", second)

    assert state.read_pending_retain("session/a") == first
    assert state.read_pending_retain("session:a") == second
    assert len(list((tmp_path / ".hindsight" / "codex" / "state").glob("retain-*.json"))) == 2


def test_pending_retain_queue_round_trips_in_order(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    first = {
        "idempotency_key": "first",
        "bank_id": "bank",
        "api_url": "http://one",
        "auth_context_id": "auth-one",
        "turn_count": 2,
        "request": {},
    }
    second = {
        "idempotency_key": "second",
        "bank_id": "bank",
        "api_url": "http://one",
        "auth_context_id": "auth-one",
        "turn_count": 4,
        "request": {},
    }

    state.write_pending_retains("session-queue", [first, second])

    assert state.read_pending_retains("session-queue") == [first, second]
    assert state.read_pending_retain("session-queue") == first


def test_deferred_retain_round_trips_and_clears(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    submission = {
        "idempotency_key": "overflow",
        "bank_id": "bank",
        "api_url": "http://one",
        "auth_context_id": "auth-one",
        "turn_count": 9,
        "request": {"content": "exact due window"},
    }

    state.write_deferred_retain("session-overflow", submission)
    assert state.read_deferred_retain("session-overflow") == submission

    state.clear_deferred_retain("session-overflow")
    assert state.read_deferred_retain("session-overflow") is None


def test_retain_cadence_round_trips_transcript_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    cadence = {"turn_count": 7, "source_fingerprint": "abc123"}

    state.write_retain_cadence("session-cadence", cadence)

    assert state.read_retain_cadence("session-cadence") == cadence


def test_retain_cadence_starts_from_legacy_turn_counter(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    state.set_turn_count("session-upgrade", 7)

    assert state.read_retain_cadence("session-upgrade") == {
        "turn_count": 7,
        "source_fingerprint": "",
    }


def test_windows_retain_lock_uses_msvcrt_boundary(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    calls = []

    class _FakeMsvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(fd, mode, length):
            calls.append((mode, length))

    monkeypatch.setattr(state, "fcntl", None)
    monkeypatch.setattr(state, "msvcrt", _FakeMsvcrt)

    with state.retain_submission_lock("windows/session"):
        assert calls == [(_FakeMsvcrt.LK_NBLCK, 1)]

    assert calls == [(_FakeMsvcrt.LK_NBLCK, 1), (_FakeMsvcrt.LK_UNLCK, 1)]


def test_windows_retain_lock_times_out_without_entering_critical_section(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    class _ContendedMsvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(fd, mode, length):
            raise OSError("lock held")

    times = iter([0.0, 31.0])
    monkeypatch.setattr(state, "fcntl", None)
    monkeypatch.setattr(state, "msvcrt", _ContendedMsvcrt)
    monkeypatch.setattr(state.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(state.time, "sleep", lambda _: None)

    with state.retain_submission_lock("windows/contended") as acquired:
        assert acquired is False


def test_pending_retain_state_is_private(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    submission = {"idempotency_key": "private", "bank_id": "bank", "request": {"content": "private transcript"}}

    state.write_pending_retain("session-private", submission)

    state_dir = tmp_path / ".hindsight" / "codex" / "state"
    pending = next(state_dir.glob("retain-*.json"))
    assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(pending.stat().st_mode) == 0o600


def test_private_state_write_works_without_fchmod(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delattr(state.os, "fchmod", raising=False)

    state.write_pending_retain(
        "session-no-fchmod",
        {"idempotency_key": "private", "bank_id": "bank", "request": {}},
    )

    pending = next((tmp_path / ".hindsight" / "codex" / "state").glob("retain-*.json"))
    assert stat.S_IMODE(pending.stat().st_mode) == 0o600


def test_pending_retain_write_failure_is_not_swallowed(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    def fail_replace(source, destination):
        raise OSError("disk full")

    monkeypatch.setattr(state.os, "replace", fail_replace)

    with pytest.raises(OSError, match="disk full"):
        state.write_pending_retain(
            "session-fail",
            {"idempotency_key": "fail", "bank_id": "bank", "request": {"content": "must not send"}},
        )


def test_pending_retain_corruption_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = state._state_file(state._retain_state_name("session-corrupt", "json"))
    with open(path, "w") as handle:
        handle.write("{not-json")

    with pytest.raises(state.json.JSONDecodeError):
        state.read_pending_retain("session-corrupt")


def test_pending_retain_write_fsyncs_file_and_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    real_fsync = state.os.fsync
    fsynced_modes = []

    def record_fsync(fd):
        fsynced_modes.append(stat.S_IFMT(state.os.fstat(fd).st_mode))
        real_fsync(fd)

    monkeypatch.setattr(state.os, "fsync", record_fsync)
    state.write_pending_retain(
        "session-durable",
        {"idempotency_key": "durable", "bank_id": "bank", "request": {}},
    )

    assert stat.S_IFREG in fsynced_modes
    if state.sys.platform != "win32":
        assert stat.S_IFDIR in fsynced_modes


def test_pending_retain_clear_fsyncs_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    state.write_pending_retain(
        "session-clear-durable",
        {"idempotency_key": "durable", "bank_id": "bank", "request": {}},
    )
    real_fsync = state.os.fsync
    fsynced_modes = []

    def record_fsync(fd):
        fsynced_modes.append(stat.S_IFMT(state.os.fstat(fd).st_mode))
        real_fsync(fd)

    monkeypatch.setattr(state.os, "fsync", record_fsync)
    state.clear_pending_retain("session-clear-durable")

    if state.sys.platform != "win32":
        assert fsynced_modes == [stat.S_IFDIR]


def test_global_turn_counter_preserves_parallel_session_updates(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    sessions = [f"session-{index}" for index in range(8)]

    with ThreadPoolExecutor(max_workers=len(sessions)) as pool:
        counts = list(pool.map(state.increment_turn_count, sessions))

    assert counts == [1] * len(sessions)
    assert {session: state.get_turn_count(session) for session in sessions} == {session: 1 for session in sessions}

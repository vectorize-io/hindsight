"""File-based state persistence.

Codex hooks are ephemeral processes — state must be persisted to files.
Uses ~/.hindsight/codex/state/ as the storage directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional, TypedDict

# fcntl is Unix-only; import conditionally so the module loads on Windows
if sys.platform != "win32":
    import fcntl

    msvcrt = None
else:
    import msvcrt

    fcntl = None

WINDOWS_LOCK_TIMEOUT_SECONDS = 30
WINDOWS_LOCK_RETRY_SECONDS = 0.1


class PendingRetain(TypedDict, total=False):
    idempotency_key: str
    bank_id: str
    api_url: str | None
    auth_context_id: str
    source_fingerprint: str
    turn_count: int
    request: dict[str, Any]
    post_attempted: bool
    operation_id: str


class RetainCadence(TypedDict):
    turn_count: int
    source_fingerprint: str


def _state_dir() -> str:
    """Get the state directory, creating it if needed."""
    state_dir = os.path.join(os.path.expanduser("~"), ".hindsight", "codex", "state")
    os.makedirs(state_dir, mode=0o700, exist_ok=True)
    os.chmod(state_dir, 0o700)
    return state_dir


def _safe_filename(name: str) -> str:
    """Sanitize a filename to prevent path traversal."""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name)
    name = name.replace("..", "_")
    name = name[:200]
    return name or "state"


def _state_file(name: str) -> str:
    """Get path for a state file. Name is sanitized to prevent traversal."""
    safe = _safe_filename(name)
    path = os.path.join(_state_dir(), safe)
    # Final guard: resolved path must be inside state_dir
    resolved = os.path.realpath(path)
    expected_dir = os.path.realpath(_state_dir())
    if not resolved.startswith(expected_dir + os.sep) and resolved != expected_dir:
        raise ValueError(f"State file path escapes state directory: {name!r}")
    return path


def _retain_state_name(session_id: str, suffix: str) -> str:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return f"retain-{digest}.{suffix}"


def _lock(lock_fd) -> None:
    if fcntl is not None:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
    elif msvcrt is not None:
        lock_fd.seek(0)
        if not lock_fd.read(1):
            lock_fd.write(b"\0")
            lock_fd.flush()
        deadline = time.monotonic() + WINDOWS_LOCK_TIMEOUT_SECONDS
        while True:
            lock_fd.seek(0)
            try:
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("retain state lock timed out")
                time.sleep(WINDOWS_LOCK_RETRY_SECONDS)


def _unlock(lock_fd) -> None:
    if fcntl is not None:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    elif msvcrt is not None:
        lock_fd.seek(0)
        msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)


def read_state(name: str, default=None):
    """Read a JSON state file. Returns default if not found."""
    path = _state_file(name)
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def write_state(name: str, data):
    """Write data to a JSON state file atomically."""
    path = _state_file(name)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def get_turn_count(session_id: str) -> int:
    """Get the current turn count for a session."""
    turns = read_state("turns.json", {})
    return turns.get(session_id, 0)


def increment_turn_count(session_id: str) -> int:
    """Increment and return the turn count for a session.

    Uses flock on Unix to prevent race conditions. On Windows, proceeds
    without a lock — minor races here are harmless.
    """
    lock_path = _state_file("turns.lock")
    if fcntl is not None:
        try:
            lock_fd = open(lock_path, "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                turns = read_state("turns.json", {})
                turns[session_id] = turns.get(session_id, 0) + 1
                if len(turns) > 10000:
                    sorted_keys = sorted(turns.keys())
                    for k in sorted_keys[: len(sorted_keys) // 2]:
                        del turns[k]
                write_state("turns.json", turns)
                return turns[session_id]
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
        except OSError:
            pass

    # Fallback: proceed without lock
    turns = read_state("turns.json", {})
    turns[session_id] = turns.get(session_id, 0) + 1
    if len(turns) > 10000:
        sorted_keys = sorted(turns.keys())
        for k in sorted_keys[: len(sorted_keys) // 2]:
            del turns[k]
    write_state("turns.json", turns)
    return turns[session_id]


def set_turn_count(session_id: str, turn_count: int) -> None:
    """Durably set one session's checkpoint without losing parallel updates."""
    lock_path = _state_file("turns.lock")
    lock_fd = os.fdopen(os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600), "a+b")
    _lock(lock_fd)
    try:
        turns = read_state("turns.json", {})
        turns[session_id] = turn_count
        if len(turns) > 10000:
            sorted_keys = sorted(turns.keys())
            for key in sorted_keys[: len(sorted_keys) // 2]:
                del turns[key]
        _write_private_json_atomic(_state_file("turns.json"), turns)
    finally:
        _unlock(lock_fd)
        lock_fd.close()


def read_retain_cadence(session_id: str) -> RetainCadence:
    """Read the last transcript identity counted by the retain hook."""
    path = _state_file(_retain_state_name(session_id, "cadence.json"))
    try:
        with open(path) as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return {
            "turn_count": get_turn_count(session_id),
            "source_fingerprint": "",
        }
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("turn_count"), int)
        or not isinstance(value.get("source_fingerprint"), str)
    ):
        raise ValueError("invalid_retain_cadence")
    return value


def write_retain_cadence(session_id: str, cadence: RetainCadence) -> None:
    """Durably record one counted transcript identity."""
    _write_private_json_atomic(
        _state_file(_retain_state_name(session_id, "cadence.json")),
        cadence,
    )


@contextmanager
def retain_submission_lock(session_id: str) -> Iterator[bool]:
    """Serialize checkpoint and pending-retain changes for one Codex session."""
    lock_path = _state_file(_retain_state_name(session_id, "lock"))
    lock_fd = os.fdopen(os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600), "a+b")
    os.chmod(lock_path, 0o600)
    try:
        _lock(lock_fd)
    except TimeoutError:
        lock_fd.close()
        yield False
        return
    try:
        yield True
    finally:
        _unlock(lock_fd)
        lock_fd.close()


def read_pending_retain(session_id: str) -> Optional[PendingRetain]:
    """Return the unacknowledged Retain submission for a session, if any."""
    pending = read_pending_retains(session_id)
    return pending[0] if pending else None


def read_pending_retains(session_id: str) -> list[PendingRetain]:
    """Return every unacknowledged submission in original turn order."""
    path = _state_file(_retain_state_name(session_id, "json"))
    try:
        with open(path) as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return []
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("invalid_pending_retain_queue")
    return value


def read_deferred_retain(session_id: str) -> Optional[PendingRetain]:
    """Return the single overflow submission held behind the bounded queue."""
    path = _state_file(_retain_state_name(session_id, "overflow.json"))
    try:
        with open(path) as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError("invalid_deferred_retain")
    return value


def _write_private_json_atomic(path: str, value: Any) -> None:
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", dir=_state_dir())
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        else:
            os.chmod(tmp_path, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        os.chmod(path, 0o600)
        if sys.platform != "win32":
            state_dir_fd = os.open(_state_dir(), os.O_RDONLY)
            try:
                os.fsync(state_dir_fd)
            finally:
                os.close(state_dir_fd)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_pending_retain(session_id: str, submission: PendingRetain) -> None:
    """Durably persist the exact Retain request before sending it.

    Unlike generic hook state, losing this file after the POST can create a
    duplicate retain. Fail closed and keep transcript-bearing files private.
    """
    _write_private_json_atomic(_state_file(_retain_state_name(session_id, "json")), submission)


def write_pending_retains(session_id: str, submissions: list[PendingRetain]) -> None:
    """Durably replace one session's ordered retry queue."""
    if not submissions:
        clear_pending_retain(session_id)
        return
    _write_private_json_atomic(_state_file(_retain_state_name(session_id, "json")), submissions)


def write_deferred_retain(session_id: str, submission: PendingRetain) -> None:
    """Persist one exact due submission that did not fit in the bounded queue."""
    _write_private_json_atomic(
        _state_file(_retain_state_name(session_id, "overflow.json")),
        submission,
    )


def _clear_retain_file(session_id: str, suffix: str) -> None:
    path = _state_file(_retain_state_name(session_id, suffix))
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    if sys.platform != "win32":
        state_dir_fd = os.open(_state_dir(), os.O_RDONLY)
        try:
            os.fsync(state_dir_fd)
        finally:
            os.close(state_dir_fd)


def clear_pending_retain(session_id: str) -> None:
    """Clear a Retain submission only after the server acknowledges it."""
    _clear_retain_file(session_id, "json")


def clear_deferred_retain(session_id: str) -> None:
    """Clear the overflow submission only after the server acknowledges it."""
    _clear_retain_file(session_id, "overflow.json")

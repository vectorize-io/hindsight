"""File-based state persistence.

Copilot CLI hooks are ephemeral processes — state must be persisted to files.
Uses ~/.hindsight/copilot-cli/state/ as the storage directory.
"""

import json
import os
import re
import sys

if sys.platform != "win32":
    import fcntl
else:
    fcntl = None


def _state_dir():
    """Get the state directory, creating it if needed."""
    state_dir = os.path.join(os.path.expanduser("~"), ".hindsight", "copilot-cli", "state")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def _safe_filename(name):
    """Sanitize a filename to prevent path traversal."""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name)
    name = name.replace("..", "_")
    name = name[:200]
    return name or "state"


def _state_file(name):
    """Get path for a state file. Name is sanitized to prevent traversal."""
    safe = _safe_filename(name)
    path = os.path.join(_state_dir(), safe)
    resolved = os.path.realpath(path)
    expected_dir = os.path.realpath(_state_dir())
    if not resolved.startswith(expected_dir + os.sep) and resolved != expected_dir:
        raise ValueError(f"State file path escapes state directory: {name!r}")
    return path


def read_state(name, default=None):
    """Read a JSON state file. Returns default if not found."""
    path = _state_file(name)
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def write_state(name, data):
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


def delete_state(name):
    """Delete a state file if present. Never raises."""
    path = _state_file(name)
    try:
        os.unlink(path)
    except OSError:
        pass


def _locked_update(lock_name, state_name, mutate_fn):
    """Read-modify-write a JSON state file under an exclusive flock (POSIX).

    ``mutate_fn(data) -> data`` receives the current state dict and returns
    the updated dict to persist. On Windows (no ``fcntl``), falls back to an
    unlocked read-modify-write — acceptable since Copilot CLI hooks are rarely
    run concurrently there and any race would only affect turn cadence, not
    correctness of retain/recall.
    """
    lock_path = _state_file(lock_name)
    if fcntl is not None:
        try:
            lock_fd = open(lock_path, "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                data = read_state(state_name, {})
                data = mutate_fn(data)
                write_state(state_name, data)
                return data
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
        except OSError:
            pass

    data = read_state(state_name, {})
    data = mutate_fn(data)
    write_state(state_name, data)
    return data


def get_turn_count(session_id):
    """Get the current turn count for a session."""
    turns = read_state("turns.json", {})
    return turns.get(session_id, 0)


def increment_turn_count(session_id):
    """Increment and return the turn count for a session.

    Uses flock on Unix to prevent race conditions. On Windows, proceeds
    without a lock — minor races here are harmless.
    """

    def _mutate(turns):
        turns[session_id] = turns.get(session_id, 0) + 1
        if len(turns) > 10000:
            sorted_keys = sorted(turns.keys())
            for k in sorted_keys[: len(sorted_keys) // 2]:
                del turns[k]
        return turns

    turns = _locked_update("turns.lock", "turns.json", _mutate)
    return turns[session_id]


def cache_session_transcript(session_id, transcript_path, cwd=None):
    """Cache the most recently seen transcript path for a session.

    Written by agent_stop.py on every turn so session_end.py — whose
    `sessionEnd` payload carries no `transcriptPath` — can still force a
    final retain using the last known transcript location.

    Uses the same flock-protected read-modify-write as ``increment_turn_count``
    since multiple concurrent Copilot CLI sessions share ``sessions.json``.
    """

    def _mutate(sessions):
        sessions[session_id] = {"transcript_path": transcript_path, "cwd": cwd}
        if len(sessions) > 10000:
            sorted_keys = sorted(sessions.keys())
            for k in sorted_keys[: len(sorted_keys) // 2]:
                del sessions[k]
        return sessions

    _locked_update("sessions.lock", "sessions.json", _mutate)


def get_cached_session_transcript(session_id):
    """Return the cached {"transcript_path", "cwd"} dict for a session, or None."""
    sessions = read_state("sessions.json", {})
    return sessions.get(session_id)


def clear_cached_session_transcript(session_id):
    """Remove a session's cached transcript info (called from session_end.py)."""

    def _mutate(sessions):
        sessions.pop(session_id, None)
        return sessions

    _locked_update("sessions.lock", "sessions.json", _mutate)

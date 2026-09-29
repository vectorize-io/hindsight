"""File-based state persistence for OMO hooks.

OMO hooks are ephemeral processes — state must be persisted to files.
Uses ~/.hindsight/omo/state/ as the storage directory.
"""

import json
import os
import re
import sys

if sys.platform != "win32":
    import fcntl
else:
    fcntl = None


def _state_dir() -> str:
    """Get the state directory, creating it if needed."""
    # Prefer OMO's plugin data dir if available, else use ~/.hindsight/omo/state/
    plugin_data = os.environ.get("PLUGIN_DATA", "")
    if plugin_data:
        state_dir = os.path.join(plugin_data, "state")
    else:
        state_dir = os.path.join(os.path.expanduser("~"), ".hindsight", "omo", "state")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def _safe_filename(name: str) -> str:
    """Sanitize a filename to prevent path traversal."""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name)
    name = name.replace("..", "_")
    name = name[:200]
    return name or "state"


def _state_file(name: str) -> str:
    """Get path for a state file."""
    safe = _safe_filename(name)
    path = os.path.join(_state_dir(), safe)
    resolved = os.path.realpath(path)
    expected_dir = os.path.realpath(_state_dir())
    if not resolved.startswith(expected_dir + os.sep) and resolved != expected_dir:
        raise ValueError(f"State file path escapes state directory: {name!r}")
    return path


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


def increment_turn_count(session_id: str) -> int:
    """Increment and return the turn count for a session."""
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

    turns = read_state("turns.json", {})
    turns[session_id] = turns.get(session_id, 0) + 1
    if len(turns) > 10000:
        sorted_keys = sorted(turns.keys())
        for k in sorted_keys[: len(sorted_keys) // 2]:
            del turns[k]
    write_state("turns.json", turns)
    return turns[session_id]


def track_retention(session_id: str, message_count: int) -> tuple:
    """Track retention state and detect compaction.

    Returns (chunk_index, compacted).
    """

    def _update(data):
        entry = data.get(session_id, {"message_count": 0, "chunk": 0})
        last_count = entry["message_count"]
        chunk = entry["chunk"]
        compacted = False

        if message_count < last_count:
            chunk += 1
            compacted = True

        entry["message_count"] = message_count
        entry["chunk"] = chunk
        data[session_id] = entry

        if len(data) > 10000:
            sorted_keys = sorted(data.keys())
            for k in sorted_keys[: len(sorted_keys) // 2]:
                del data[k]

        return data, (chunk, compacted)

    lock_path = _state_file("retention_tracking.lock")
    if fcntl is not None:
        try:
            lock_fd = open(lock_path, "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                data = read_state("retention_tracking.json", {})
                data, result = _update(data)
                write_state("retention_tracking.json", data)
                return result
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
        except OSError:
            pass

    data = read_state("retention_tracking.json", {})
    data, result = _update(data)
    write_state("retention_tracking.json", data)
    return result

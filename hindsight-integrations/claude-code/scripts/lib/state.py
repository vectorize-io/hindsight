"""File-based state persistence.

Claude Code hooks are ephemeral processes — state must be persisted to files.
Uses $CLAUDE_PLUGIN_DATA/state/ as the storage directory.
"""

import json
import os
import re
import sys
from dataclasses import dataclass

# fcntl is Unix-only; import conditionally so the module loads on Windows
if sys.platform != "win32":
    import fcntl
else:
    fcntl = None


@dataclass(frozen=True)
class RetentionProgress:
    """State transition for a full-session retain attempt."""

    chunk_index: int
    compacted: bool
    start_index: int


def _state_dir() -> str:
    """Get the state directory, creating it if needed."""
    plugin_data = os.environ.get("CLAUDE_PLUGIN_DATA", "")
    if not plugin_data:
        # Fallback to a temp location for testing
        plugin_data = os.path.join(os.path.expanduser("~"), ".claude", "plugins", "data", "hindsight-memory")
    state_dir = os.path.join(plugin_data, "state")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def _safe_filename(name: str) -> str:
    """Sanitize a filename to prevent path traversal.

    Strips path separators, .., and control characters. Mirrors Openclaw's
    sanitizeFilename().
    """
    # Replace path separators and dangerous patterns
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name)
    # Collapse .. to prevent traversal
    name = name.replace("..", "_")
    # Limit length
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
        # Best-effort cleanup
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

    Uses flock on Unix to prevent race conditions between concurrent hook
    processes (e.g. async Stop + new UserPromptSubmit). On Windows, flock is
    unavailable so we proceed without a lock — minor races here are harmless.
    """
    lock_path = _state_file("turns.lock")
    if fcntl is not None:
        try:
            lock_fd = open(lock_path, "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                turns = read_state("turns.json", {})
                turns[session_id] = turns.get(session_id, 0) + 1
                # Cap tracked sessions to prevent unbounded growth
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

    # Fallback: proceed without lock (Windows or lock acquisition failed)
    turns = read_state("turns.json", {})
    turns[session_id] = turns.get(session_id, 0) + 1
    # Cap tracked sessions to prevent unbounded growth
    if len(turns) > 10000:
        sorted_keys = sorted(turns.keys())
        for k in sorted_keys[: len(sorted_keys) // 2]:
            del turns[k]
    write_state("turns.json", turns)
    return turns[session_id]


def _locked_read_modify_write(state_name: str, lock_name: str, modify_fn):
    """Read-modify-write a state file under flock.

    modify_fn receives the current state dict and returns (updated_dict, result).
    Returns the result from modify_fn.
    """
    lock_path = _state_file(lock_name)
    if fcntl is not None:
        try:
            lock_fd = open(lock_path, "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                data = read_state(state_name, {})
                data, result = modify_fn(data)
                write_state(state_name, data)
                return result
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
        except OSError:
            pass

    # Fallback without lock
    data = read_state(state_name, {})
    data, result = modify_fn(data)
    write_state(state_name, data)
    return result


def plan_retention(session_id: str, message_count: int) -> RetentionProgress:
    """Plan retention state without writing the checkpoint.

    The caller must only commit the returned progress after the retain request
    succeeds; otherwise a transient API failure would skip unsent messages on the
    next hook run.
    """
    data = read_state("retention_tracking.json", {})
    entry = data.get(session_id, {"message_count": 0, "chunk": 0})
    last_count = entry["message_count"]
    chunk = entry["chunk"]
    compacted = False
    start_index = last_count

    if message_count < last_count:
        # Transcript shrank — compaction happened. The compacted transcript is
        # new canonical content, so send it as a fresh chunk rather than trying
        # to diff it against the pre-compaction transcript.
        chunk += 1
        compacted = True
        start_index = 0
    elif message_count > last_count and last_count > 0:
        # The transcript grew normally. Store only the new suffix in its own
        # document so repeated Stop hooks grow linearly instead of resending
        # the whole session under a replacement document_id.
        chunk += 1
    elif message_count == last_count:
        start_index = message_count

    return RetentionProgress(chunk_index=chunk, compacted=compacted, start_index=start_index)


def commit_retention(session_id: str, message_count: int, chunk_index: int) -> None:
    """Persist the checkpoint for a successful retain."""

    def _update(data):
        data[session_id] = {"message_count": message_count, "chunk": chunk_index}

        # Cap tracked sessions
        if len(data) > 10000:
            sorted_keys = sorted(data.keys())
            for k in sorted_keys[: len(sorted_keys) // 2]:
                del data[k]

        return data, None

    _locked_read_modify_write("retention_tracking.json", "retention_tracking.lock", _update)


def track_retention(session_id: str, message_count: int) -> RetentionProgress:
    """Track retention state, compaction, and the next unsent message offset.

    Full-session mode reads Claude Code's cumulative transcript, but each retain
    should only send messages added since the previous successful hook run. Reusing
    a document_id replaces that document on the server, so incremental payloads use
    a monotonically increasing chunk id. When the transcript shrinks, Claude Code
    compacted the session; the new compacted transcript starts a fresh chunk.

    Returns:
        RetentionProgress with the chunk_index for building document_id, whether
        compaction was detected, and the first message index to retain from the
        current transcript.
    """
    progress = plan_retention(session_id, message_count)
    commit_retention(session_id, message_count, progress.chunk_index)
    return progress

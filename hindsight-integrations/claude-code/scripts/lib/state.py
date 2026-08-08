"""File-based state persistence.

Claude Code hooks are ephemeral processes — state must be persisted to files.
Uses $CLAUDE_PLUGIN_DATA/state/ as the storage directory.
"""

import contextlib
import json
import os
import re
import sys
import time
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
    expected_dir = os.path.realpath(_state_dir())
    path = os.path.join(expected_dir, safe)

    # Guard 1 (always): the sanitized name must be a bare basename. _safe_filename
    # already strips separators and collapses "..", so this is a cheap assertion
    # that holds regardless of what exists on disk.
    if safe != os.path.basename(safe) or safe in (os.curdir, os.pardir):
        raise ValueError(f"State file path escapes state directory: {name!r}")

    # Guard 2 (only when the target exists): defend against a symlink planted at
    # the state path itself.
    #
    # This is deliberately conditional. realpath() on a path that does NOT exist —
    # or that is momentarily absent because another process is between its
    # tempfile write and its os.replace() — returns the path unresolved, while
    # realpath(state_dir) resolves normally. When the state dir sits behind a
    # symlink/junction the two then disagree and the prefix compare raises, even
    # though nothing is wrong. Measured on Windows: 1-2 spurious raises per
    # 150-240 concurrent calls, 0 sequentially. A raise here aborts the hook, so
    # the state write is silently skipped.
    if os.path.lexists(path):
        try:
            resolved = os.path.realpath(path)
        except OSError:
            # Vanished mid-check — the write path is what matters, and guard 1
            # has already established it is inside the state dir.
            return path
        if resolved != expected_dir and not resolved.startswith(expected_dir + os.sep):
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


@contextlib.contextmanager
def _exclusive_lock(lock_name: str, timeout: float = 5.0, stale_after: float = 60.0):
    """Hold an exclusive lock around a state read-modify-write.

    Unix keeps flock — unchanged. Where flock is unavailable (Windows) this used
    to proceed with NO lock, on the reasoning that "minor races here are
    harmless". They are not: write_state() is atomic via os.replace(), but the
    read-modify-write wrapped around it is not, so a concurrent writer rebuilds
    the whole dict from a stale read and drops the other writer's changes —
    including, because these files are dicts keyed by session, whole entries
    belonging to sessions that never raced at all.

    Measured on Windows against this file, isolated CLAUDE_PLUGIN_DATA, real
    increment_turn_count: 1 process x 50 increments loses 0; 6 concurrent
    processes x 40 increments return 239/240 successful calls but leave a total
    of 9 in the file, with only 3 of 6 session keys still present.

    Yields True if the lock was held and False if it could not be taken within
    `timeout` — callers proceed either way, so this is never worse than the
    previous behaviour.
    """
    lock_path = _state_file(lock_name)

    if fcntl is not None:
        lock_fd = None
        acquired = False
        try:
            lock_fd = open(lock_path, "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            acquired = True
        except OSError:
            acquired = False
        try:
            yield acquired
        finally:
            if lock_fd is not None:
                try:
                    if acquired:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    try:
                        lock_fd.close()
                    except OSError:
                        pass
        return

    # No flock available: an atomic create-exclusive lockfile. O_CREAT|O_EXCL is
    # a single atomic syscall on Windows too, so exactly one waiter wins.
    fd = None
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except (FileExistsError, PermissionError):
            # FileExistsError: another process holds the lock.
            # PermissionError: Windows pending-delete. The previous holder has
            #   called unlink() but its handle is not fully released, and an
            #   O_EXCL create against a pending-delete name raises EACCES rather
            #   than EEXIST. Treating it as fatal is what makes a lockfile look
            #   unreliable on Windows — it is transient and the correct response
            #   is to retry. (Measured: 4 spurious lock-acquisition failures per
            #   240 concurrent calls before this branch existed.)
            # Both mean "try again".
            try:
                # Reclaim a lock orphaned by a killed holder (hooks run under a
                # timeout and can be terminated mid-write).
                if time.time() - os.path.getmtime(lock_path) > stale_after:
                    os.unlink(lock_path)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        except OSError:
            break
    try:
        yield fd is not None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(lock_path)
            except OSError:
                pass


def increment_turn_count(session_id: str) -> int:
    """Increment and return the turn count for a session.

    Held under _exclusive_lock to prevent races between concurrent hook
    processes (e.g. async Stop + new UserPromptSubmit, or several agents sharing
    one CLAUDE_PLUGIN_DATA).
    """
    with _exclusive_lock("turns.lock"):
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
    with _exclusive_lock(lock_name):
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

"""File-based state persistence.

Devin CLI hooks are ephemeral processes, like Claude Code's — state must be
persisted to files. Unlike the Claude Code plugin, Devin CLI does not set a
per-plugin data-directory env var (no `CLAUDE_PLUGIN_DATA` equivalent), so
this uses a fixed, well-known location instead: `~/.hindsight/devin-cli/state/`.
"""

import contextlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass

# Each platform offers exactly one of these; import conditionally so the module
# loads on both. See _file_lock() for how they are used.
if sys.platform != "win32":
    import fcntl

    msvcrt = None
else:
    fcntl = None
    import msvcrt


@dataclass(frozen=True)
class RetentionProgress:
    """State transition for a full-session retain attempt."""

    chunk_index: int
    compacted: bool
    start_index: int
    # The checkpoint plan_retention() read — both halves of it. commit_retention()
    # writes only if the stored checkpoint still matches, so a slower concurrent
    # hook cannot roll a newer one back — see commit_retention().
    observed_message_count: int
    observed_chunk: int


def _state_dir() -> str:
    """Get the state directory, creating it if needed."""
    override = os.environ.get("HINDSIGHT_DEVIN_CLI_DATA_DIR")
    base = override or os.path.join(os.path.expanduser("~"), ".hindsight", "devin-cli")
    state_dir = os.path.join(base, "state")
    os.makedirs(state_dir, exist_ok=True)
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
    resolved = os.path.realpath(path)
    expected_dir = os.path.realpath(_state_dir())
    if not resolved.startswith(expected_dir + os.sep) and resolved != expected_dir:
        raise ValueError(f"State file path escapes state directory: {name!r}")
    return path


def read_state(name: str, default=None):
    """Read a JSON state file. Returns default if not found or unusable.

    A caller passing a dict default gets a dict back, always. Every state file
    here is a JSON object, and every caller immediately subscripts or `.get()`s
    the result — so a file holding valid-but-wrong JSON (`null`, `[]`, a bare
    string) would crash the hook, while a file holding *invalid* JSON already
    degraded to the default. Tolerating the corrupt case but not the merely
    wrong-shaped one is a distinction no caller can act on, and nothing in this
    module rewrites a bad file, so the crash would repeat on every hook run.
    """
    path = _state_file(name)
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            data = json.load(f)
    # UnicodeDecodeError is what a partially-written file looks like from text
    # mode, and it is neither a JSONDecodeError nor an OSError — so without it
    # here a single torn write crashes every subsequent hook run.
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return default
    if isinstance(default, dict) and not isinstance(data, dict):
        return default
    return data


def write_state(name: str, data) -> None:
    """Write data to a JSON state file atomically.

    Raises OSError if the write fails. Turn counters and retention checkpoints
    live here, and a caller that believes a checkpoint was saved when it was not
    will re-send messages that were already retained (or skip ones that were
    not), so an unwritable state directory has to be visible rather than silent.
    """
    path = _state_file(name)
    # A per-process staging file, not a fixed `<path>.tmp`. Hooks run
    # concurrently and several write the same state file, so a shared staging
    # name lets one writer os.replace() the file another is still filling —
    # yielding a torn read or an OSError on an otherwise healthy write.
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def get_turn_count(session_id: str) -> int:
    """Get the current turn count for a session."""
    turns = read_state("turns.json", {})
    return turns.get(session_id, 0)


def increment_turn_count(session_id: str) -> int:
    """Increment and return the turn count for a session.

    Locked against concurrent hook processes (e.g. an async Stop overlapping the
    next UserPromptSubmit). A lost increment is not harmless: it shifts every
    later turn number, so a `retainEveryNTurns` retain fires late or not at all.
    """

    def _bump(turns):
        # Count captured before capping, not read back after it.
        count = turns.get(session_id, 0) + 1
        _touch(turns, session_id, count)
        _cap_tracked_sessions(turns, protect=session_id)
        return turns, count

    return locked_read_modify_write("turns.json", "turns.lock", _bump)


def _stored_int(entry, key: str) -> int:
    """Read a non-negative int field out of a persisted session entry.

    Anything malformed — a missing key, a non-dict entry, a string where an int
    belongs — reads as 0, i.e. "no checkpoint yet". Both plan_retention and
    commit_retention go through here so their views of a broken entry agree;
    when they disagreed, the compare-and-swap between them could never match
    and the entry was never repaired.

    Negatives are rejected, not just non-ints. A stored message_count of -1
    would flow through as `start_index`, so the retain would slice from -1 —
    sending only the final message and then committing the real total as the
    checkpoint, permanently skipping everything before it. Reading it as 0
    re-retains the session instead, which is recoverable.

    bool is excluded explicitly because it is a subclass of int, so `True`
    would otherwise pass as the checkpoint 1.
    """
    if not isinstance(entry, dict):
        return 0
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value if value >= 0 else 0


_MAX_TRACKED_SESSIONS = 10000


def _touch(data: dict, session_id: str, value) -> None:
    """Write a session's entry and move it to the end of the dict.

    Assigning an existing key leaves it where it is, so without the pop a
    long-lived session keeps the insertion position it got on its very first
    turn — and _cap_tracked_sessions, which evicts from the front, would drop
    the busiest session while idle ones survive. Dict order round-trips
    through JSON, so this makes the file its own recency list at no cost.
    """
    data.pop(session_id, None)
    data[session_id] = value


def _cap_tracked_sessions(data: dict, protect: str = "") -> None:
    """Drop the least recently used half once the file grows past the cap.

    Session ids are never cleaned up by the CLI, so without this the file grows
    for the life of the install.

    `protect` is the session being written right now. It is exempt because the
    caller reads `data[session_id]` back immediately afterwards: evicting it
    here would raise KeyError out of the hook, and it is by definition the most
    recently used entry in the file.
    """
    if len(data) > _MAX_TRACKED_SESSIONS:
        # Front of the dict is the least recently touched — see _touch().
        for k in list(data)[: len(data) // 2]:
            if k != protect:
                del data[k]


@contextlib.contextmanager
def _file_lock(lock_path: str):
    """Hold an exclusive interprocess lock on `lock_path` for the block.

    Uses flock on Unix and msvcrt.locking on Windows. Never falls through
    unlocked — not for the platform, which is what previously left Windows
    racing, and not for a failed acquisition either.

    A failure to acquire raises. Yielding unlocked would let the caller's
    read-modify-write run with no synchronisation at all while looking like it
    had the lock, which is the corruption the lock exists to prevent: two hooks
    read the same counter and one overwrites the other. Raising cannot be
    dismissed as "the state directory is unwritable anyway, so the write would
    have failed regardless" — a `turns.lock` left behind with the wrong owner
    or mode blocks the lock while `turns.json` stays perfectly writable.

    Callers treat the OSError as a failed state operation, which degrades to a
    recomputed checkpoint or a skipped registration rather than silent loss.
    """
    lock_fd = None
    try:
        lock_fd = open(lock_path, "w")
        if fcntl is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        else:
            # msvcrt locks a byte range rather than the whole file, and the file
            # is ours alone, so locking one byte is a whole-file lock in effect.
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_LOCK, 1)
    except OSError:
        if lock_fd is not None:
            lock_fd.close()
        raise

    try:
        yield
    finally:
        if lock_fd is not None:
            with contextlib.suppress(OSError):
                if fcntl is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                else:
                    lock_fd.seek(0)
                    msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
            lock_fd.close()


def locked_read_modify_write(state_name: str, lock_name: str, modify_fn):
    """Read-modify-write a state file under an interprocess lock.

    modify_fn receives the current state dict and returns (updated_dict, result).
    Returns the result from modify_fn.

    The read-modify-write itself is never retried. Earlier this sat inside the
    `try` that guards lock acquisition, so a failed write fell through to the
    unlocked path and applied the same mutation a second time.
    """
    with _file_lock(_state_file(lock_name)):
        data = read_state(state_name, {})
        data, result = modify_fn(data)
        write_state(state_name, data)
        return result


def plan_retention(session_id: str, message_count: int) -> RetentionProgress:
    """Plan retention state without writing the checkpoint.

    The caller must only commit the returned progress after the retain request
    succeeds; otherwise a transient API failure would skip unsent messages on
    the next hook run.
    """
    data = read_state("retention_tracking.json", {})
    # A per-session entry is read defensively for the same reason the file as a
    # whole is (see read_state): a half-written or hand-edited entry missing
    # either key would otherwise raise KeyError out of every retain hook, and
    # nothing repairs it. Treating it as "no checkpoint yet" re-retains at worst
    # a few turns, which is recoverable; crashing on every run is not.
    last_count = _stored_int(data.get(session_id), "message_count")
    observed_chunk = _stored_int(data.get(session_id), "chunk")
    chunk = observed_chunk
    compacted = False
    start_index = last_count

    if message_count < last_count:
        chunk += 1
        compacted = True
        start_index = 0
    elif message_count > last_count > 0:
        chunk += 1
    elif message_count == last_count:
        start_index = message_count

    return RetentionProgress(
        chunk_index=chunk,
        compacted=compacted,
        start_index=start_index,
        observed_message_count=last_count,
        observed_chunk=observed_chunk,
    )


def commit_retention(session_id: str, message_count: int, progress: RetentionProgress) -> bool:
    """Persist the checkpoint for a successful retain.

    Compare-and-swap against the checkpoint `plan_retention` observed. Retain
    hooks can overlap, and they plan from independent snapshots: a hook that saw
    20 messages may finish after one that saw 30 and already checkpointed them.
    Writing its stale count would roll the checkpoint backwards and re-send
    messages 21-30 on the next run. Returns False when the write was skipped for
    that reason.
    """

    def _update(data):
        # Read through the same coercion plan_retention used. Comparing the raw
        # stored value instead would make the CAS unsatisfiable for a malformed
        # entry — plan_retention reports 0, the stored "4" never equals it, the
        # write is skipped forever and the session re-retains from 0 on every
        # run. Matching coercions are what let the bad entry be overwritten.
        # Both halves of the checkpoint, not just the count. A compaction can
        # bring the count back to a value this hook already observed while the
        # chunk has moved on: hook X plans at 20, a faster hook checkpoints 30,
        # then a compaction checkpoints 20 again under a new chunk. Comparing
        # the count alone lets X's write through, rolling the chunk backwards
        # and advancing the count past a transcript that no longer has those
        # messages — which are then never retained.
        if (
            _stored_int(data.get(session_id), "message_count") != progress.observed_message_count
            or _stored_int(data.get(session_id), "chunk") != progress.observed_chunk
        ):
            return data, False
        _touch(data, session_id, {"message_count": message_count, "chunk": progress.chunk_index})
        _cap_tracked_sessions(data, protect=session_id)
        return data, True

    return locked_read_modify_write("retention_tracking.json", "retention_tracking.lock", _update)


def track_retention(session_id: str, message_count: int) -> RetentionProgress:
    """Track retention state, compaction, and the next unsent message offset."""
    progress = plan_retention(session_id, message_count)
    commit_retention(session_id, message_count, progress)
    return progress

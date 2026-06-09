"""File-based state persistence.

Cline hooks are ephemeral processes, and Cline does not hand hooks a
conversation transcript — so we persist the per-task transcript (and
bank-mission bookkeeping) to files under ~/.hindsight/cline/state/ and read
it back at task end to retain.
"""

import json
import os
import re


def _state_dir() -> str:
    """Return the state directory, creating it if needed."""
    state_dir = os.path.join(os.path.expanduser("~"), ".hindsight", "cline", "state")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


def _safe_filename(name: str) -> str:
    """Sanitize a filename to prevent path traversal."""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name)
    name = name.replace("..", "_")
    name = name[:200]
    return name or "state"


def _state_file(name: str) -> str:
    """Path for a state file. Name is sanitized to prevent traversal."""
    safe = _safe_filename(name)
    path = os.path.join(_state_dir(), safe)
    resolved = os.path.realpath(path)
    expected_dir = os.path.realpath(_state_dir())
    if not resolved.startswith(expected_dir + os.sep) and resolved != expected_dir:
        raise ValueError(f"State file path escapes state directory: {name!r}")
    return path


def read_state(name: str, default=None):
    """Read a JSON state file. Returns default if not found or unreadable."""
    path = _state_file(name)
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def write_state(name: str, data) -> None:
    """Write data to a JSON state file atomically (tmp + rename)."""
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


def clear_state(name: str) -> None:
    """Delete a state file if it exists."""
    try:
        os.unlink(_state_file(name))
    except OSError:
        pass

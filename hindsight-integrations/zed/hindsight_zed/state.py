"""Persistent daemon state: which threads have been retained, and at what revision.

A thread is retained again only when its ``updated_at`` advances past what we
last stored, so a long-running conversation is re-captured as it grows without
duplicating unchanged threads.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_STATE_FILE = Path.home() / ".hindsight" / "zed-state.json"


@dataclass
class DaemonState:
    """Tracks the last-retained ``updated_at`` per thread id."""

    retained: dict = field(default_factory=dict)  # thread_id -> updated_at string
    path: Path = DEFAULT_STATE_FILE

    @classmethod
    def load(cls, path: Path = DEFAULT_STATE_FILE) -> "DaemonState":
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                retained = data.get("retained", {})
                if isinstance(retained, dict):
                    return cls(retained={str(k): str(v) for k, v in retained.items()}, path=path)
            except (json.JSONDecodeError, OSError):
                pass
        return cls(path=path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"retained": self.retained}, indent=2) + "\n", encoding="utf-8")

    def needs_retain(self, thread_id: str, updated_at: str) -> bool:
        """True if this thread is new or has advanced since the last retain."""
        return self.retained.get(thread_id) != updated_at

    def mark_retained(self, thread_id: str, updated_at: str) -> None:
        self.retained[thread_id] = updated_at

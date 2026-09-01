"""Stage breadcrumbs for in-flight worker tasks.

The worker poller binds a `StageHolder` to each task's contextvar scope.
Engine code calls `set_stage("retain.facts.llm")` at phase boundaries; the
poller reads the holder periodically to surface what each in-flight task is
currently doing in `WORKER_STATS` / `WORKER_TASK` log lines.

Outside a worker context the contextvar is unset and `set_stage` is a no-op,
so engine code is safe to call from sync HTTP requests, tests, or the CLI
without any setup.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class StageHolder:
    """Mutable container for the current task's stage label."""

    stage: str = "init"
    updated_at: float = field(default_factory=time.monotonic)
    #: Optional callback fired on every stage change, used by the poller to push
    #: back an *idle* wall-clock ceiling (see ``_WallCeiling.extends_on_progress``).
    #: Push, not polling: a stage change is exactly the progress signal, so there
    #: is no interval to tune and no window in which a task looks stalled because
    #: nobody has looked at it yet. Must not raise — it runs on the engine's hot
    #: path, and a stage breadcrumb is never worth failing a task over.
    on_progress: Callable[[], None] | None = None


_current_holder: ContextVar[StageHolder | None] = ContextVar("hindsight_stage_holder", default=None)


def bind_holder(holder: StageHolder):
    """Bind a holder to the current async context.

    Must be called from inside the task coroutine itself (not from the
    spawning code) so the binding lives in the task's own contextvar scope.

    Returns the token that can be passed to `_current_holder.reset()` if
    the binding ever needs to be unwound.
    """
    return _current_holder.set(holder)


def set_stage(name: str) -> None:
    """Update the current task's stage label.

    No-op when called outside a worker task context (e.g. from a sync HTTP
    request, a test, or the CLI). Cheap enough to call per-phase.
    """
    holder = _current_holder.get()
    if holder is None:
        return
    holder.stage = name
    holder.updated_at = time.monotonic()
    if holder.on_progress is not None:
        holder.on_progress()


def get_stage() -> str | None:
    """Return the current stage label, or None if no holder is bound."""
    holder = _current_holder.get()
    return holder.stage if holder is not None else None

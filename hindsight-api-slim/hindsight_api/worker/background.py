"""Marks code that runs inside a background worker task.

The poller calls `mark_background_task()` at the start of every task it executes.
The mark lives in that task's contextvar scope, so everything the task awaits or
spawns sees it, and live HTTP requests never do. The LLM semaphore reads it to let
live requests take a free slot ahead of background work (#5926).

Kept separate from `stage.StageHolder` on purpose: the stage holder is a logging
breadcrumb the poller may skip binding, so "has a stage" is not a reliable
"is background" signal.
"""

from contextvars import ContextVar

_in_background_task: ContextVar[bool] = ContextVar("hindsight_in_background_task", default=False)


def mark_background_task() -> None:
    """Mark the current task context as background work.

    Must be called from inside the task coroutine itself, so the mark lands in the
    task's own contextvar scope and not in the code that spawned it.
    """
    _in_background_task.set(True)


def in_background_task() -> bool:
    """True when called from inside a background worker task."""
    return _in_background_task.get()

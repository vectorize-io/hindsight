"""Keep time queued for an LLM permit out of a time limit.

A time limit is meant to catch work that will never finish. Time a call spends
queued behind other operations' calls is not this operation's work, and charging
it cancelled healthy mental-model refreshes whenever a burst of them shared a
small per-operation LLM cap (#6140).

A limit opts in by running under :func:`excluding_permit_wait`. The LLM wrapper
wraps every permit acquire in :func:`waiting_for_permit`. While at least one of
the block's calls is queued, the clock is held: the deadline moves out by
``max_wait`` when the queueing starts and back by whatever of it went unused when
it ends, so the limit is charged nothing for the queue. A wait longer than
``max_wait`` still runs the clock out, so a permit that never frees is still
caught. Overlapping waits of parallel calls hold the clock once.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class _HeldLimit:
    """One time limit and the calls currently queued under it."""

    __slots__ = ("limit", "max_wait", "queued_calls", "queued_since")

    def __init__(self, limit: asyncio.Timeout, max_wait: float) -> None:
        self.limit = limit
        self.max_wait = max_wait
        self.queued_calls = 0
        self.queued_since = 0.0

    def shift(self, seconds: float) -> None:
        when = self.limit.when()
        if when is None:
            return
        try:
            self.limit.reschedule(when + seconds)
        except RuntimeError:
            # The limit already fired or its block exited; nothing left to move.
            pass


_held_limits: ContextVar[tuple[_HeldLimit, ...]] = ContextVar("hindsight_permit_wait_limits", default=())


@contextmanager
def excluding_permit_wait(limit: asyncio.Timeout, max_wait: float) -> Iterator[None]:
    """Hold ``limit``'s clock while the block's LLM calls are queued, up to ``max_wait`` per wait."""
    token = _held_limits.set((*_held_limits.get(), _HeldLimit(limit, max_wait)))
    try:
        yield
    finally:
        _held_limits.reset(token)


@contextmanager
def waiting_for_permit() -> Iterator[None]:
    """Mark the block as time spent queued for an LLM permit."""
    held = _held_limits.get()
    for h in held:
        if h.queued_calls == 0:
            h.queued_since = time.monotonic()
            h.shift(h.max_wait)
        h.queued_calls += 1
    try:
        yield
    finally:
        for h in held:
            h.queued_calls -= 1
            if h.queued_calls == 0:
                waited = time.monotonic() - h.queued_since
                h.shift(-max(0.0, h.max_wait - waited))

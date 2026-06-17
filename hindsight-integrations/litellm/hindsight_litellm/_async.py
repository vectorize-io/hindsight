"""Sync-to-async bridge backed by a single owned event loop per thread.

The integration exposes synchronous APIs (``recall``, ``retain``, ``reflect``,
``enable()``, ``wrap_openai`` …) that drive the async ``hindsight-client`` under
the hood. Bridging sync → async needs an event loop, and the historical pattern
used across this package was::

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)

That pattern has two problems on modern Python:

1. ``asyncio.get_event_loop()`` with no current loop emits a
   ``DeprecationWarning`` on 3.12+ and is slated for removal — it raised in
   early 3.14 betas.
2. Loops created this way are never closed, so the interpreter warns about
   "unclosed event loop" at exit, and the ``hindsight-client``'s cached aiohttp
   session is left bound to a loop nobody owns.

A fresh loop per call (``asyncio.run``) is *not* an option: the client caches
its aiohttp session on the loop that first used it, so a throwaway loop breaks
session reuse across subsequent sync calls.

This module keeps **one loop per thread**, created explicitly (so no
``get_event_loop`` deprecation) and set as the thread's current loop (so the
client's own internal ``get_event_loop()`` reuses it instead of spawning yet
another). The loop is reused across calls — keeping cached sessions valid — for
the lifetime of the thread. :func:`close_loop` can tear it down explicitly, but
it is deliberately NOT called by ``cleanup()`` (see its docstring): closing a
shared loop out from under a still-live client raises "Event loop is closed".
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine, TypeVar

_T = TypeVar("_T")

# Per-thread so concurrent callers never share a loop (asyncio loops are not
# thread-safe). Each thread that bridges sync→async gets its own owned loop.
_local = threading.local()


def ensure_loop() -> asyncio.AbstractEventLoop | None:
    """Make sure this thread has an owned, current event loop, and return it.

    No-ops (returns ``None``) when called from inside a running loop — in that
    async context the caller should be using the ``a``-prefixed async APIs, and
    we must not touch the running loop. Otherwise creates the thread's loop on
    first use and registers it via ``set_event_loop`` so nested
    ``get_event_loop()`` calls (including the client's own sync bridge) reuse it
    rather than emitting the "no current event loop" deprecation.
    """
    try:
        asyncio.get_running_loop()
        return None  # Inside an async context — leave the running loop alone.
    except RuntimeError:
        pass

    loop = getattr(_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _local.loop = loop
    return loop


def run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coro`` to completion on this thread's owned loop.

    Must be called from synchronous code (no running loop). The loop is reused
    across calls so the client's cached aiohttp session stays bound to a live
    loop.
    """
    loop = ensure_loop()
    if loop is None:
        raise RuntimeError("run_sync() called from within a running event loop; use the async API instead.")
    return loop.run_until_complete(coro)


def close_loop() -> None:
    """Close this thread's owned loop if one exists. Idempotent.

    WARNING: This invalidates *every* Hindsight client whose aiohttp session
    is bound to this thread's loop — including clients the caller still holds.
    It is therefore NOT called automatically by ``cleanup()``: a shared loop
    closed out from under a live client raises "Event loop is closed" on the
    next call. Only call this at true thread/process shutdown when no client
    will be used again. A subsequent sync call transparently creates a fresh
    loop.
    """
    loop = getattr(_local, "loop", None)
    _local.loop = None
    if loop is not None and not loop.is_closed():
        loop.close()
    # Never leave a closed loop as the thread's current loop — a later
    # get_event_loop() (ours or the client's) would otherwise hand back a dead
    # loop. Clearing it lets the next caller create a fresh one.
    try:
        asyncio.set_event_loop(None)
    except Exception:
        pass

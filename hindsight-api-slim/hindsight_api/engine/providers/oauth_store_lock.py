"""
Single-flight lock for an on-disk OAuth credential store.

The OAuth-backed providers (Codex, Nous, xai-oauth) refresh a rotating token and
write it back to a JSON store that other processes read too. The refresh awaits
the network, so whatever serialises it is held across ``await`` — which rules out
the path-keyed ``threading.Lock`` these managers used while the refresh was sync
(held across ``await``, it would block the loop instead of yielding). Two layers
replace it:

* **Within one event loop:** an ``asyncio.Lock`` per (running loop, store path), so
  concurrent coroutines queue on it rather than all polling the file lock. It is
  keyed by the running loop because an asyncio lock binds to the first loop that
  waits on it, and one manager can be reached from several loops.
* **Across loops, threads and processes:** an ``fcntl.flock`` on ``<store>.lock``,
  taken non-blocking and retried with ``asyncio.sleep`` so a waiter yields its
  loop. A flock conflicts between separate open file descriptions even inside one
  process, so it also serialises two loops of the same process — the job the old
  in-process lock did.

Where ``fcntl`` is unavailable (Windows) only the per-loop lock applies.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path

from ..aiohttp_session import LoopLocal

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Interval between non-blocking flock attempts while another holder has the store.
_POLL_INTERVAL_SECONDS = 0.05

# One map of store path -> lock per event loop; LoopLocal prunes closed loops (a weak
# map would not: a lock that has been waited on holds a reference to its loop).
_LOOP_LOCKS: LoopLocal[dict[Path, asyncio.Lock]] = LoopLocal(dict)


def _loop_lock(key: Path) -> asyncio.Lock:
    """Return the running loop's lock for one store, creating it on first use."""
    # setdefault on a map only the running loop touches: no await, no other thread.
    return _LOOP_LOCKS.get().setdefault(key, asyncio.Lock())


@contextlib.asynccontextmanager
async def oauth_store_lock(store: Path, *, timeout_seconds: float, label: str) -> AsyncIterator[None]:
    """Hold the refresh lock for the credential store at ``store``.

    ``label`` names the store in the timeout error and the no-``fcntl`` debug line.
    Only waiting for the file lock is bounded by ``timeout_seconds``; the per-loop
    lock is released as soon as its holder finishes, which is itself bounded.
    """
    key = store.expanduser().resolve(strict=False)
    async with _loop_lock(key):
        if fcntl is None:  # pragma: no cover - Windows
            logger.debug(f"fcntl unavailable; {label} refresh proceeds without a cross-process lock.")
            yield
            return

        lock_path = store.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+") as lock_file:
            deadline = time.monotonic() + max(1.0, timeout_seconds)
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out waiting for the {label} lock") from None
                    await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            try:
                yield
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

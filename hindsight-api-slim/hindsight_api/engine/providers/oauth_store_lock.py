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
import errno
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

# Errors that mean the store cannot be written, so no lock file can live beside
# it. The lock degrades to per-loop-only for these; every other OSError from
# creating the lock file propagates.
_UNWRITABLE_STORE_ERRNOS = frozenset({errno.EROFS, errno.EACCES, errno.EPERM})

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
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_file = open(lock_path, "a+")
        except OSError as e:
            # Only a store that cannot be written gets here, and that is the
            # normal shape on Kubernetes: a Secret volume is mounted read-only
            # whatever `volumeMount.readOnly` says, so a store fed by an
            # external secret manager (ESO, a sidecar, a ConfigMap projection)
            # raises EROFS. Nothing can be written back to such a store, so the
            # lock file does not exist and the caller's read path — which is how
            # a credential published by another writer arrives — must still run.
            # Failing here would instead take the whole refresh with it.
            #
            # Everything else must keep propagating. The same two calls fail on
            # a perfectly WRITABLE store — ENOSPC when the volume is full,
            # EMFILE/ENFILE when descriptors run out — and those arrive exactly
            # when the box is under pressure. Degrading there would silently
            # drop the cross-process lock and let two processes into the refresh
            # body together, which is the race this lock exists to prevent. For
            # these providers that means two concurrent rotations of a rotating
            # token, where one of them is necessarily lost.
            if e.errno not in _UNWRITABLE_STORE_ERRNOS:
                raise
            # Degrade to the per-loop lock alone, exactly as the no-`fcntl`
            # branch above does. Note the file lock never protected against a
            # store owned by another writer: such a writer does not take it.
            logger.debug(
                f"{label} store is not writable ({type(e).__name__}: {e}); refresh proceeds without a cross-process lock."
            )
            yield
            return

        with lock_file:
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

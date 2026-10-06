"""Shared bookkeeping for fire-and-forget background DB writes.

Used by the audit log and the LLM trace: both schedule best-effort writes with
``asyncio.create_task`` and both need the resulting tasks kept somewhere.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class PendingWrites:
    """In-flight fire-and-forget write tasks, held strongly and drainable.

    Two reasons the tasks cannot just be dropped on the floor: the event loop
    only keeps weak references to them, so a write nothing else references can
    be garbage collected mid-flight, and a caller tearing down the pool (or
    about to issue an UPDATE over the rows) has to let the writes land first.

    Tasks are bucketed by key so one operation's writes can be drained without
    blocking on unrelated ones.
    """

    def __init__(self, what: str) -> None:
        self._what = what
        self._pending: dict[str | None, set[asyncio.Task[None]]] = {}

    def schedule(self, coro: Coroutine[Any, Any, None], key: str | None = None) -> bool:
        """Start ``coro`` as a tracked task. False (and no task) if there is no loop."""
        try:
            task = asyncio.create_task(coro)
        except RuntimeError:
            # No running event loop (e.g. during shutdown)
            logger.debug(f"Cannot schedule {self._what}: no running event loop")
            coro.close()
            return False
        self._pending.setdefault(key, set()).add(task)
        task.add_done_callback(lambda t, k=key: self._discard(k, t))
        return True

    def _discard(self, key: str | None, task: asyncio.Task[None]) -> None:
        bucket = self._pending.get(key)
        if bucket is not None:
            bucket.discard(task)
            if not bucket:
                self._pending.pop(key, None)

    async def drain(self, key: str | None = None) -> None:
        """Await one key's in-flight writes. Unbounded: callers are mid-operation."""
        tasks = [t for t in self._pending.get(key, ()) if not t.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def drain_all(self, timeout: float) -> None:
        """Await every in-flight write, bounded so a stuck database cannot hang shutdown."""
        tasks = [t for bucket in self._pending.values() for t in bucket]
        if not tasks:
            return
        _, abandoned = await asyncio.wait(tasks, timeout=timeout)
        if abandoned:
            logger.warning(f"{len(abandoned)} {self._what}(s) still in flight after {timeout}s")

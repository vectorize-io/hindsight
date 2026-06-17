"""TTL + coalescing cache for `get_bank_stats`.

`get_bank_stats` aggregates over `memory_links` (and joins to `memory_units`),
which can be a multi-second parallel sequential scan on banks with millions of
rows. The result is intentionally approximate (it powers a UI widget and a
freshness hint inside `reflect`), so caching it for a few tens of seconds is
safe and dramatically reduces planner-driven thrash from clients that poll.

The cache also coalesces concurrent misses on the same key onto a single
in-flight task so that N concurrent callers produce one query rather than N.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable


class BankStatsCache:
    """Per-process TTL cache keyed on (schema, bank_id).

    `ttl_seconds <= 0` disables caching: each call passes straight through to
    the loader. `max_entries` bounds memory in environments with many banks.
    """

    def __init__(self, *, ttl_seconds: float, max_entries: int) -> None:
        self._ttl = float(ttl_seconds)
        self._max_entries = int(max_entries) if max_entries and max_entries > 0 else 0
        self._entries: OrderedDict[tuple[str, str], tuple[float, dict[str, Any]]] = OrderedDict()
        self._in_flight: dict[tuple[str, str], asyncio.Future[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    def _now(self) -> float:
        return time.monotonic()

    def _get_fresh_unlocked(self, key: tuple[str, str]) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at <= self._now():
            # Expired — drop so the loader runs again.
            self._entries.pop(key, None)
            return None
        # Mark as recently used for LRU eviction.
        self._entries.move_to_end(key)
        return value

    def _store_unlocked(self, key: tuple[str, str], value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._entries[key] = (self._now() + self._ttl, value)
        self._entries.move_to_end(key)
        if self._max_entries:
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    async def get_or_load(
        self,
        schema: str,
        bank_id: str,
        loader: Callable[[], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Return cached stats for `(schema, bank_id)` or call `loader()`.

        Concurrent misses on the same key are coalesced onto a single
        in-flight loader.
        """
        if not self.enabled:
            return await loader()

        key = (schema, bank_id)

        async with self._lock:
            cached = self._get_fresh_unlocked(key)
            if cached is not None:
                return cached
            in_flight = self._in_flight.get(key)
            if in_flight is None:
                in_flight = asyncio.get_running_loop().create_future()
                self._in_flight[key] = in_flight
                is_owner = True
            else:
                is_owner = False

        if not is_owner:
            return await asyncio.shield(in_flight)

        try:
            value = await loader()
        except BaseException as exc:
            async with self._lock:
                self._in_flight.pop(key, None)
            if not in_flight.done():
                in_flight.set_exception(exc)
            # Suppress "Future exception was never retrieved" when no other
            # caller was waiting on this loader — we re-raise to the owner
            # immediately and the future is a no-op in that case.
            in_flight.exception()
            raise

        async with self._lock:
            self._store_unlocked(key, value)
            self._in_flight.pop(key, None)
        if not in_flight.done():
            in_flight.set_result(value)
        return value

    async def invalidate(self, schema: str, bank_id: str) -> None:
        """Drop any cached stats for `(schema, bank_id)`."""
        async with self._lock:
            self._entries.pop((schema, bank_id), None)

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()

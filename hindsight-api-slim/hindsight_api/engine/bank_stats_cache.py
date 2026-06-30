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
    """Per-process TTL cache keyed on (schema, bank_id, *key_suffix).

    `ttl_seconds <= 0` disables caching: each call passes straight through to
    the loader. `max_entries` bounds memory in environments with many banks.

    Callers may extend the key with a `key_suffix` so semantically distinct
    variants of the same `(schema, bank_id)` (e.g. with and without an
    optional expensive aggregation) get separate cache slots. `invalidate`
    clears every variant for a `(schema, bank_id)` so writers don't have to
    know which suffixes the read path is using.
    """

    def __init__(self, *, ttl_seconds: float, max_entries: int) -> None:
        self._ttl = float(ttl_seconds)
        self._max_entries = int(max_entries) if max_entries and max_entries > 0 else 0
        self._entries: OrderedDict[tuple[Any, ...], tuple[float, dict[str, Any]]] = OrderedDict()
        self._in_flight: dict[tuple[Any, ...], asyncio.Future[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    def _now(self) -> float:
        return time.monotonic()

    def _get_fresh_unlocked(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
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

    def _store_unlocked(self, key: tuple[Any, ...], value: dict[str, Any]) -> None:
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
        *,
        key_suffix: tuple[Any, ...] = (),
    ) -> dict[str, Any]:
        """Return cached stats for `(schema, bank_id, *key_suffix)` or call `loader()`.

        Concurrent misses on the same key are coalesced onto a single
        in-flight loader. `key_suffix` lets a caller carve out separate
        slots for variants that compute different results (e.g. a flag that
        toggles an optional expensive aggregation) without confusing each
        other's responses.
        """
        if not self.enabled:
            return await loader()

        key = (schema, bank_id, *key_suffix)

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
                # Invalidation may have detached this loader and allowed a new
                # one to claim the key. Never remove that newer loader's slot.
                if self._in_flight.get(key) is in_flight:
                    self._in_flight.pop(key, None)
            if not in_flight.done():
                in_flight.set_exception(exc)
            # Suppress "Future exception was never retrieved" when no other
            # caller was waiting on this loader — we re-raise to the owner
            # immediately and the future is a no-op in that case.
            in_flight.exception()
            raise

        async with self._lock:
            # Only the loader that still owns the key may populate the cache.
            # An invalidated loader can finish for its original callers, but its
            # pre-invalidation result must not overwrite a newer load.
            if self._in_flight.get(key) is in_flight:
                self._store_unlocked(key, value)
                self._in_flight.pop(key, None)
        if not in_flight.done():
            in_flight.set_result(value)
        return value

    async def invalidate(self, schema: str, bank_id: str) -> None:
        """Drop every cached stats variant for `(schema, bank_id)`.

        Clears all entries whose key starts with `(schema, bank_id)`, so
        writers that don't know which `key_suffix` values the read path is
        using still wipe the bank cleanly. Detaches in-flight loaders
        rather than cancelling them — existing callers may finish with
        their pre-invalidation snapshot while post-invalidation callers
        reload.
        """
        async with self._lock:
            prefix = (schema, bank_id)
            for key in [k for k in self._entries if k[:2] == prefix]:
                self._entries.pop(key, None)
            for key in [k for k in self._in_flight if k[:2] == prefix]:
                self._in_flight.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()
            self._in_flight.clear()

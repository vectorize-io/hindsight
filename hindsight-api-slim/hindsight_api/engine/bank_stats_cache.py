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
import json
import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .db_utils import acquire_with_retry

if TYPE_CHECKING:
    from .db.base import DatabaseBackend

logger = logging.getLogger(__name__)


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
        """Drop any cached stats for `(schema, bank_id)`."""
        async with self._lock:
            key = (schema, bank_id)
            self._entries.pop(key, None)
            # Detach rather than cancel: existing callers may finish with the
            # snapshot they requested, while post-invalidation callers reload.
            self._in_flight.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()
            self._in_flight.clear()


class DistributedBankStatsCache:
    """Table-backed (cross-process) TTL cache for `get_bank_stats`.

    Same ``get_or_load`` / ``invalidate`` / ``clear`` contract as
    :class:`BankStatsCache`, but the store is the per-schema ``bank_stats_cache``
    table instead of a per-process dict — so one worker's computation is shared
    with every other worker, and no caller recomputes while a fresh row exists.

    On a hit, a call is a single primary-key ``SELECT`` (sub-millisecond); only a
    miss runs the (expensive) ``loader`` and writes the row back. Concurrent
    misses are *not* coalesced across processes (that would need a lock): they
    each compute and ``UPSERT``, last write wins — all results are correct, at the
    cost of a brief redundant compute at expiry.

    Every DB touch is best-effort: if the cache table is unreachable or missing
    (e.g. a schema mid-migration), the call degrades to computing without caching
    rather than failing ``get_bank_stats``. PostgreSQL only — the engine keeps the
    in-process :class:`BankStatsCache` for Oracle.
    """

    def __init__(self, *, backend: "DatabaseBackend", ttl_seconds: float) -> None:
        self._backend = backend
        self._ttl = float(ttl_seconds)

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    @staticmethod
    def _qualified(schema: str) -> str:
        return f'"{schema}".bank_stats_cache' if schema else "bank_stats_cache"

    async def get_or_load(
        self,
        schema: str,
        bank_id: str,
        loader: Callable[[], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        if not self.enabled:
            return await loader()

        table = self._qualified(schema)

        # 1. Fresh row? Single PK lookup; ``payload::text`` sidesteps any
        #    jsonb->object codec so we always decode the same way.
        try:
            async with acquire_with_retry(self._backend) as conn:
                row = await conn.fetchrow(
                    f"SELECT payload::text AS payload FROM {table} "
                    f"WHERE bank_id = $1 AND computed_at > now() - make_interval(secs => $2::double precision)",
                    bank_id,
                    self._ttl,
                )
            if row is not None:
                return json.loads(row["payload"])
        except Exception as exc:  # noqa: BLE001 — cache read must never break the endpoint
            logger.debug("bank_stats_cache read failed for %s.%s (%s); computing uncached", schema, bank_id, exc)
            return await loader()

        # 2. Miss — compute, then write the row back (best-effort).
        value = await loader()
        try:
            async with acquire_with_retry(self._backend) as conn:
                await conn.execute(
                    f"INSERT INTO {table} (bank_id, payload, computed_at) VALUES ($1, $2::jsonb, now()) "
                    f"ON CONFLICT (bank_id) DO UPDATE SET payload = EXCLUDED.payload, computed_at = now()",
                    bank_id,
                    json.dumps(value),
                )
        except Exception as exc:  # noqa: BLE001 — a failed write just means no caching this round
            logger.warning("bank_stats_cache write failed for %s.%s (%s)", schema, bank_id, exc)
        return value

    async def invalidate(self, schema: str, bank_id: str) -> None:
        """Drop the cached row so the next read recomputes."""
        if not self.enabled:
            return
        try:
            async with acquire_with_retry(self._backend) as conn:
                await conn.execute(f"DELETE FROM {self._qualified(schema)} WHERE bank_id = $1", bank_id)
        except Exception as exc:  # noqa: BLE001 — invalidation must never break the write path
            logger.debug("bank_stats_cache invalidate failed for %s.%s (%s)", schema, bank_id, exc)

    async def clear(self) -> None:
        """Drop all cached rows in the current schema (best-effort)."""
        if not self.enabled:
            return
        from .memory_engine import get_current_schema

        try:
            async with acquire_with_retry(self._backend) as conn:
                await conn.execute(f"DELETE FROM {self._qualified(get_current_schema())}")
        except Exception as exc:  # noqa: BLE001
            logger.debug("bank_stats_cache clear failed (%s)", exc)

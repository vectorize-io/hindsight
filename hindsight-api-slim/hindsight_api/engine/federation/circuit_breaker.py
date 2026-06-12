"""Minimal async circuit breaker for the Graphiti outbound client.

Trips after ``failure_threshold`` consecutive failures; while open, every
call short-circuits to ``CircuitOpenError`` until ``reset_timeout_seconds``
have elapsed, then transitions to half-open (one probe request decides
whether to close or re-open).

No clock injection, no metrics plumbing — the same shape other Hindsight
breakers use (see ``llm_wrapper._build_per_op_semaphores`` for the
sibling per-op rate-limiter pattern). Concurrency-safe via a single
asyncio.Lock around the state transitions; the lock is held briefly so it
does not serialize the hot path.
"""

import asyncio
import time
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when the breaker is open and the call is short-circuited."""


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_seconds: float = 30.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout_seconds
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def _maybe_half_open(self) -> None:
        if self._state is CircuitState.OPEN and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self._reset_timeout:
                self._state = CircuitState.HALF_OPEN

    async def call(self, coro_factory):
        """Run ``coro_factory()`` under the breaker.

        ``coro_factory`` is a zero-arg callable that returns a fresh awaitable
        on each call (so a half-open probe can be re-issued cleanly).
        """
        async with self._lock:
            await self._maybe_half_open()
            if self._state is CircuitState.OPEN:
                raise CircuitOpenError("graphiti circuit breaker is open")
        try:
            result = await coro_factory()
        except Exception:
            await self._record_failure()
            raise
        await self._record_success()
        return result

    async def _record_failure(self) -> None:
        async with self._lock:
            self._consecutive_failures += 1
            if (
                self._state is CircuitState.CLOSED and self._consecutive_failures >= self._failure_threshold
            ) or self._state is CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()

    async def _record_success(self) -> None:
        async with self._lock:
            self._consecutive_failures = 0
            if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
                self._state = CircuitState.CLOSED
            self._opened_at = None

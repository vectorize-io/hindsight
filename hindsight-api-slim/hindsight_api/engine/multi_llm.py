"""Multi-LLM routing: failover and (weighted) round-robin across N providers.

``MultiLLMProvider`` wraps an ordered list of :class:`LLMProvider` members and a
:class:`~hindsight_api.config.LLMStrategyConfig`, exposing the same public surface
as a single ``LLMProvider`` so it drops into every existing call path (including
``with_config()`` / ``ConfiguredLLMProvider``).

Member 0 is the **primary** (the operation's unindexed/base LLM); members 1..N are
the indexed extras (``HINDSIGHT_API_<OP>LLM_<n>_*``). Each member keeps its own
internal retry budget, so we only advance to the next member after a member has
exhausted its retries and raised.

Strategies:
- ``failover``: try members in declared order ``[0..N]``.
- ``round-robin``: rotate the starting member per request (optionally weighted),
  then fall through the remaining members on error.

Batch retain and any direct ``_provider_impl`` access operate on the **primary
member only** (via attribute passthrough) — failover/round-robin apply to the
interactive ``call`` / ``call_with_tools`` paths.
"""

import logging
import threading
import uuid
from typing import TYPE_CHECKING, Any

from ..config import LLM_STRATEGY_FAILOVER, LLMStrategyConfig
from .llm_wrapper import LLMProvider, OutputTooLongError

if TYPE_CHECKING:
    from .llm_wrapper import ConfiguredLLMProvider, LLMToolCallResult

logger = logging.getLogger(__name__)


def _should_failover(exc: BaseException) -> bool:
    """Whether ``exc`` from one member should trigger a try on the next member.

    Generic ``Exception`` instances (network errors, provider 5xx, timeouts after
    a member's own retries) fail over. ``OutputTooLongError`` is propagated — a
    different provider won't fit an over-length output either. ``CancelledError``,
    ``KeyboardInterrupt`` and ``SystemExit`` are ``BaseException`` (not
    ``Exception``) and therefore propagate unchanged.
    """
    if isinstance(exc, OutputTooLongError):
        return False
    return isinstance(exc, Exception)


class _WeightedRoundRobin:
    """Smooth weighted round-robin scheduler (nginx SWRR).

    Produces a starting member index per request such that, over time, member
    ``i`` is chosen in proportion to ``weights[i]`` while keeping selections
    interleaved rather than bursty. Uniform weights degrade to plain round-robin.
    The tiny selection critical section is mutex-guarded so concurrent callers
    don't corrupt the running totals (they may still interleave, which only
    affects distribution, never correctness).
    """

    def __init__(self, weights: list[int]) -> None:
        self._weights = list(weights)
        self._current = [0] * len(weights)
        self._total = sum(weights)
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            best = 0
            for i, w in enumerate(self._weights):
                self._current[i] += w
                if self._current[i] > self._current[best]:
                    best = i
            self._current[best] -= self._total
            return best


class MultiLLMProvider:
    """Route LLM calls across multiple members per a failover / round-robin strategy."""

    def __init__(self, members: list[LLMProvider], strategy: LLMStrategyConfig) -> None:
        if not members:
            raise ValueError("MultiLLMProvider requires at least one member")
        self._members = members
        self._strategy = strategy

        weights = strategy.weights or [1] * len(members)
        if len(weights) != len(members):
            raise ValueError(
                f"LLM strategy 'weights' has {len(weights)} entries but the chain has "
                f"{len(members)} members (primary + indexed); they must match."
            )
        self._scheduler = _WeightedRoundRobin(weights)

    # ── routing ────────────────────────────────────────────────────────────────

    def _member_order(self) -> list[int]:
        """Indices to try, in order, for one request."""
        n = len(self._members)
        if self._strategy.mode == LLM_STRATEGY_FAILOVER:
            return list(range(n))
        start = self._scheduler.next()
        return [(start + i) % n for i in range(n)]

    async def _dispatch(self, method_name: str, **kwargs: Any) -> Any:
        last_exc: BaseException | None = None
        order = self._member_order()
        for position, idx in enumerate(order):
            member = self._members[idx]
            try:
                return await getattr(member, method_name)(**kwargs)
            except BaseException as e:  # noqa: BLE001 - re-raised unless it should fail over
                if not _should_failover(e):
                    raise
                last_exc = e
                remaining = len(order) - position - 1
                logger.warning(
                    "LLM member %d (%s/%s) failed on %s: %s%s",
                    idx,
                    member.provider,
                    member.model,
                    method_name,
                    e,
                    f"; trying next member ({remaining} left)" if remaining else "; no members left",
                )
        # All members failed; surface the last error (loop ran at least once).
        assert last_exc is not None
        raise last_exc

    async def call(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return await self._dispatch("call", messages=messages, **kwargs)

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> "LLMToolCallResult":
        return await self._dispatch("call_with_tools", messages=messages, tools=tools, **kwargs)

    # ── lifecycle ────────────────────────────────────────────────────────────────

    async def verify_connection(self) -> None:
        """Strictly verify the primary; soft-verify the rest (warn, don't fail).

        A failover member being unreachable at startup must not block the server —
        it may come back before it's needed. The primary is the steady-state path,
        so its failure is still surfaced (the caller already wraps this in a
        warn-only try/except at startup).
        """
        await self._members[0].verify_connection()
        for member in self._members[1:]:
            try:
                await member.verify_connection()
            except Exception as e:  # noqa: BLE001 - soft verification
                logger.warning(
                    "Failover LLM member %s/%s failed connection verification: %s. "
                    "It will be tried at request time if the primary fails.",
                    member.provider,
                    member.model,
                    e,
                )

    async def cleanup(self) -> None:
        for member in self._members:
            await member.cleanup()

    def with_config(
        self,
        config: Any,
        *,
        bank_id: str | None = None,
        operation: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ConfiguredLLMProvider":
        """Mirror ``LLMProvider.with_config`` so the strategy runs inside the
        per-operation configured wrapper (gemini-safety + trace contextvars wrap
        every member call)."""
        from .llm_trace import LLMTraceContext
        from .llm_wrapper import ConfiguredLLMProvider

        trace_ctx = None
        if bank_id is not None or operation is not None or metadata:
            trace_ctx = LLMTraceContext(
                bank_id=bank_id,
                operation=operation,
                metadata=dict(metadata or {}),
                trace_id=str(uuid.uuid4()),
                operation_span_id=str(uuid.uuid4()),
            )
        return ConfiguredLLMProvider(self, config.llm_gemini_safety_settings, trace_ctx)

    # ── attribute passthrough ────────────────────────────────────────────────────

    @property
    def members(self) -> list[LLMProvider]:
        return self._members

    def __getattr__(self, name: str) -> Any:
        # Anything not defined here (provider, model, api_key, base_url,
        # _provider_impl, mock helpers, batch helpers, ...) delegates to the
        # primary member so existing call sites keep working unchanged.
        return getattr(object.__getattribute__(self, "_members")[0], name)

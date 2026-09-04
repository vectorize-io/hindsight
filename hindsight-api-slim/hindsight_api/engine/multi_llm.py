"""Multi-LLM routing across N providers.

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
- ``metadata``: bind an operation to the member selected by its ephemeral
  metadata, defaulting to member 0. Matches to different members are rejected;
  a selected member is strict and never falls across lanes on error.

For failover/round-robin, batch retain runs on the **first batch-capable member**
in declared order (see ``batch_provider_impl``), which need not be the primary.
Metadata routing pins the member before batch submission. Once selected, the
whole batch lifecycle stays on that member and does not fail over. Every other
direct ``_provider_impl`` access still resolves to the primary via attribute
passthrough.
"""

import logging
import threading
import uuid
from typing import TYPE_CHECKING, Any

from ..config import LLM_STRATEGY_FAILOVER, LLM_STRATEGY_METADATA, LLMStrategyConfig
from .llm_wrapper import LLMProvider, OutputTooLongError

if TYPE_CHECKING:
    from .llm_interface import LLMInterface
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

        if strategy.mode == LLM_STRATEGY_METADATA:
            if not strategy.routes:
                raise ValueError("Metadata LLM routing requires at least one route")
            for route in strategy.routes:
                if route.member >= len(members):
                    raise ValueError(
                        f"LLM metadata route {route.key}={route.value!r} selects member {route.member}, "
                        f"but the chain has members 0..{len(members) - 1}."
                    )
            tag_members = {route.member for route in strategy.routes if route.key == "tags"}
            if len(tag_members) > 1:
                raise ValueError(
                    "Metadata LLM routes with key 'tags' must all select the same member; "
                    "reflect and consolidation include every configured tag route to stay fail-closed."
                )

    # ── routing ────────────────────────────────────────────────────────────────

    def _member_order(self) -> list[int]:
        """Indices to try, in order, for one request."""
        n = len(self._members)
        if self._strategy.mode == LLM_STRATEGY_METADATA:
            # Metadata is supplied at operation binding time (``with_config``).
            # A direct call has no routing context, so it stays on the primary.
            return [0]
        if self._strategy.mode == LLM_STRATEGY_FAILOVER:
            return list(range(n))
        start = self._scheduler.next()
        return [(start + i) % n for i in range(n)]

    def _member_for_metadata(self, metadata: dict[str, Any] | None) -> LLMProvider:
        """Return the unambiguous metadata route match, or the primary.

        One LLM call cannot be split across classification lanes. If its combined
        metadata matches routes to different members, choosing either member
        could disclose the other lane's data. Refuse the operation instead.
        """
        matched_members: list[int] = []
        for route in self._strategy.routes or []:
            actual = (metadata or {}).get(route.key)
            matches = actual == route.value or (
                isinstance(actual, (list, tuple, set, frozenset)) and route.value in actual
            )
            if matches:
                matched_members.append(route.member)

        distinct_members = set(matched_members)
        if len(distinct_members) > 1:
            raise ValueError(
                "LLM metadata routes for this operation select multiple members; "
                "split the input or route all matching classifications to one member."
            )
        return self._members[matched_members[0]] if matched_members else self._members[0]

    def validate_routing_metadata(self, metadata: dict[str, Any] | None) -> None:
        """Validate operation-wide metadata before work is split or queued."""
        if self._strategy.mode == LLM_STRATEGY_METADATA:
            self._member_for_metadata(metadata)

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

        A secondary being unreachable at startup must not block the server — it
        may come back before its routing strategy selects it. The primary is the
        steady-state path, so its failure is still surfaced (the caller already
        wraps this in a warn-only try/except at startup).
        """
        await self._members[0].verify_connection()
        for member in self._members[1:]:
            try:
                await member.verify_connection()
            except Exception as e:  # noqa: BLE001 - soft verification
                logger.warning(
                    "Secondary LLM member %s/%s failed connection verification: %s. "
                    "It will be used if the routing strategy selects it at request time.",
                    member.provider,
                    member.model,
                    e,
                )

    # ── batch routing ───────────────────────────────────────────────────────────

    async def supports_batch_api(self) -> bool:
        """Whether the configured strategy can safely use the batch API.

        Failover and round-robin need any one capable member. Metadata routing
        needs every selectable member because the request-specific member is not
        known during startup validation.
        """
        if self._strategy.mode == LLM_STRATEGY_METADATA:
            routed_indices = {0, *(route.member for route in self._strategy.routes or [])}
            for index in routed_indices:
                if not await self._members[index].supports_batch_api():
                    return False
            return True
        return (await self.batch_provider_impl()) is not None

    async def batch_provider_impl(self, account_key: str | None = None) -> "LLMInterface | None":
        """The implementation serving batch, or ``None`` when no member can.

        Selection is deterministic by declared member order (primary first), so a
        fresh batch goes to the first batch-capable member — the whole batch
        lifecycle (submit → poll → retrieve) must target a single provider
        account, and it does not fail over: a batch already submitted to one
        account cannot be polled from another.

        Declared order is *not* enough to resume one, though. The chain can be
        reordered or extended across a restart, and two members of the same
        provider on different accounts look identical by provider name, so
        "first capable member" can resolve to an account that never saw the
        batch (#3671). A resume therefore passes the ``account_key`` recorded at
        submit time and gets back the member that owns the batch, or ``None`` —
        never a lookalike.
        """
        members = self._members[:1] if self._strategy.mode == LLM_STRATEGY_METADATA else self._members
        for member in members:
            impl = await member.batch_provider_impl(account_key)
            if impl is not None:
                return impl
        return None

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
        routing_metadata: dict[str, Any] | None = None,
    ) -> "ConfiguredLLMProvider":
        """Mirror ``LLMProvider.with_config`` so the strategy runs inside the
        per-operation configured wrapper (gemini-safety + trace contextvars wrap
        every member call)."""
        if self._strategy.mode == LLM_STRATEGY_METADATA:
            # Pin the whole operation before any interactive or batch call. A
            # sensitive route is intentionally strict: if its selected provider
            # fails, its content must not spill into a differently classified lane.
            return self._member_for_metadata(routing_metadata).with_config(
                config,
                bank_id=bank_id,
                operation=operation,
                metadata=metadata,
            )

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

    @property
    def strategy(self) -> LLMStrategyConfig:
        return self._strategy

    def __getattr__(self, name: str) -> Any:
        # Anything not defined here (provider, model, api_key, base_url,
        # _provider_impl, mock helpers, ...) delegates to the primary member so
        # existing call sites keep working unchanged. The batch helpers above are
        # defined precisely because the primary is the wrong answer for them.
        return getattr(object.__getattribute__(self, "_members")[0], name)

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

Batch retain runs on the **first batch-capable member** in declared order (see
``batch_provider_impl``), which need not be the primary; once selected, the whole
batch lifecycle stays on that member and does not fail over. Every other direct
``_provider_impl`` access still resolves to the primary via attribute passthrough
— failover/round-robin apply to the interactive ``call`` / ``call_with_tools``
paths.
"""

import logging
import math
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from time import monotonic
from typing import TYPE_CHECKING, Any

from ..config import LLM_STRATEGY_FAILOVER, LLMStrategyConfig
from .llm_interface import (
    LLMCooldownFailure,
    LLMTerminalFailure,
    ProviderRateLimitResetError,
    ProviderReauthenticationRequiredError,
)
from .llm_wrapper import LLMProvider, OutputTooLongError

if TYPE_CHECKING:
    from .llm_interface import LLMInterface
    from .llm_wrapper import ConfiguredLLMProvider, LLMToolCallResult

logger = logging.getLogger(__name__)

_DEFAULT_COOLDOWN_SECONDS = 60.0


@dataclass
class _MemberState:
    cooldown_until: float | None = None
    probing: bool = False
    generation: int = 0


def _should_failover(exc: BaseException) -> bool:
    """Whether ``exc`` from one member should trigger a try on the next member.

    Generic ``Exception`` instances (network errors, provider 5xx, timeouts after
    a member's own retries) fail over. ``OutputTooLongError`` is propagated — a
    different provider won't fit an over-length output either. ``CancelledError``,
    ``KeyboardInterrupt`` and ``SystemExit`` are ``BaseException`` (not
    ``Exception``) and therefore propagate unchanged.
    """
    if isinstance(exc, (OutputTooLongError, ProviderReauthenticationRequiredError)):
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
        # Ownership is per router instance, not per credential or process-wide.
        self._states = [_MemberState() for _ in members]
        self._state_lock = threading.Lock()

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
            label = member.member_label or ("primary" if idx == 0 else f"member-{idx}")
            with self._state_lock:
                state = self._states[idx]
                if state.probing:
                    logger.debug(
                        "LLM member %d (%s/%s, label=%s) skipped: state=%s",
                        idx,
                        member.provider,
                        member.model,
                        label,
                        "probing",
                    )
                    continue
                if state.cooldown_until is not None:
                    remaining_seconds = state.cooldown_until - monotonic()
                    if remaining_seconds > 0:
                        logger.debug(
                            "LLM member %d (%s/%s, label=%s) skipped: state=cooldown remaining=%.3fs",
                            idx,
                            member.provider,
                            member.model,
                            label,
                            remaining_seconds,
                        )
                        continue
                probing = state.cooldown_until is not None
                generation = state.generation
                if probing:
                    state.probing = True
            if probing:
                logger.info("LLM member %d (%s/%s, label=%s) state=probing", idx, member.provider, member.model, label)
            try:
                result = await getattr(member, method_name)(**kwargs)
                with self._state_lock:
                    # A success started before a newer quota failure says nothing
                    # about that cooldown; only its matching probe can reopen it.
                    if probing and state.generation == generation:
                        state.cooldown_until = None
                        logger.info(
                            "LLM member %d (%s/%s, label=%s) state=eligible after successful probe",
                            idx,
                            member.provider,
                            member.model,
                            label,
                        )
                return result
            except BaseException as e:  # noqa: BLE001 - re-raised unless it should fail over
                if not isinstance(e, Exception):
                    raise
                # The router knows the member's position; a standalone wrapper
                # cannot correctly name an unlabelled secondary member.
                failure = (
                    LLMTerminalFailure()
                    if isinstance(e, ProviderReauthenticationRequiredError)
                    else member.classify_failure(e)
                )
                if isinstance(failure, LLMTerminalFailure):
                    logger.warning(
                        "LLM member %d (%s/%s, label=%s) category=reauthentication_required; stopping operation",
                        idx,
                        member.provider,
                        member.model,
                        label,
                    )
                    raise ProviderReauthenticationRequiredError(
                        f"LLM member {idx} ({label}) requires reauthentication. "
                        "Refresh its configured credentials before retrying."
                    ) from None
                if isinstance(failure, LLMCooldownFailure) or probing:
                    delay = failure.retry_after_seconds if isinstance(failure, LLMCooldownFailure) else None
                    cooldown_source = "provider_retry_after"
                    if delay is None or not math.isfinite(delay) or delay < 0:
                        delay = _DEFAULT_COOLDOWN_SECONDS
                        cooldown_source = "default"
                    with self._state_lock:
                        state.cooldown_until = max(state.cooldown_until or 0.0, monotonic() + delay)
                        state.generation += 1
                    logger.warning(
                        "LLM member %d (%s/%s, label=%s) state=cooldown category=%s "
                        "cooldown_source=%s retry_after=%.3fs",
                        idx,
                        member.provider,
                        member.model,
                        label,
                        failure.category.value if failure is not None else "probe_failed",
                        cooldown_source,
                        delay,
                    )
                if not _should_failover(e):
                    raise
                last_exc = e
                remaining = len(order) - position - 1
                logger.warning(
                    "LLM member %d (%s/%s, label=%s) failed on %s: %s%s",
                    idx,
                    member.provider,
                    member.model,
                    label,
                    method_name,
                    failure.category.value if failure is not None else e,
                    f"; trying next member ({remaining} left)" if remaining else "; no members left",
                )
            finally:
                if probing:
                    with self._state_lock:
                        state.probing = False
        # No provider request is made while every member is cooling/probing.
        # Reuse the worker's existing quota defer signal, not a router wait loop.
        with self._state_lock:
            if last_exc is None or all(state.cooldown_until is not None for state in self._states):
                now = monotonic()
                delay = min(
                    (
                        1.0 if state.probing else max(0.0, state.cooldown_until - now)
                        for state in self._states
                        if state.cooldown_until is not None
                    ),
                    # Another thread may have finished its probe since this
                    # dispatch skipped it. Defer without inventing a new replay.
                    default=1.0,
                )
                wall_now = datetime.now(timezone.utc)
                # Extremely large finite Retry-After values must not overflow
                # datetime. Saturate only the external wakeup, not eligibility.
                max_delay = (datetime.max.replace(tzinfo=timezone.utc) - wall_now).total_seconds() - 1.0
                retry_at = wall_now + timedelta(seconds=min(delay, max_delay))
                raise ProviderRateLimitResetError(
                    retry_at=retry_at,
                    message="All LLM members are cooling down or probing; retry after the reset time.",
                ) from None
        # Otherwise preserve the existing final-error behavior.
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

    def supports_vision(self) -> bool | None:
        """Whether EVERY member can accept images — the opposite of batch routing.

        Batch capacity may live on one member because the batch path picks that
        member deliberately. Vision cannot: any call may fail over to any member,
        so a chain is only safe for images if none of its members would drop
        them. One ``False`` makes the chain False; otherwise an unknown member
        makes the whole chain unknown.
        """
        answers = [member.supports_vision() for member in self._members]
        if any(answer is False for answer in answers):
            return False
        if any(answer is None for answer in answers):
            return None
        return True

    # ── batch routing ───────────────────────────────────────────────────────────

    async def supports_batch_api(self) -> bool:
        """Whether ANY member supports the batch API.

        The single-provider path delegates to the primary, but in a multi-LLM
        chain batch capacity may live on a secondary member (e.g. an ``openai`` /
        ``groq`` fallback behind a non-batch primary). Mirroring the failover
        semantics, the batch path can proceed as long as one member can serve it.
        """
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
        for member in self._members:
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
        # _provider_impl, mock helpers, ...) delegates to the primary member so
        # existing call sites keep working unchanged. The batch helpers above are
        # defined precisely because the primary is the wrong answer for them.
        return getattr(object.__getattribute__(self, "_members")[0], name)

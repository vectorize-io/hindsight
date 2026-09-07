"""Multi-LLM routing: failover, (weighted) round-robin and metadata across N providers.

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
- ``metadata``: retain only. Each retained item picks its member from its own
  ``metadata`` (see ``member_for_metadata``); an item matching no route uses the
  primary. Selection happens per item at fact-extraction time, so nothing about
  it is stored and no other operation is affected — see ``config.py`` and the
  configuration docs for what this does and does not promise.

Batch retain runs on the **first batch-capable member** in declared order (see
``batch_provider_impl``), which need not be the primary; once selected, the whole
batch lifecycle stays on that member and does not fail over. Every other direct
``_provider_impl`` access still resolves to the primary via attribute passthrough
— failover/round-robin apply to the interactive ``call`` / ``call_with_tools``
paths.
"""

import asyncio
import logging
import math
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import TYPE_CHECKING, Any, cast

from ..config import LLM_STRATEGY_FAILOVER, LLM_STRATEGY_METADATA, LLMStrategyConfig
from .llm_interface import LLMCooldownFailure, LLMTerminalFailure, ProviderRateLimitResetError
from .llm_wrapper import LLMProvider, OutputTooLongError

if TYPE_CHECKING:
    from .llm_interface import LLMInterface
    from .llm_wrapper import ConfiguredLLMProvider, LLMToolCallResult

logger = logging.getLogger(__name__)

_DEFAULT_COOLDOWN_SECONDS = 60.0
_PROBE_POLL_SECONDS = 0.05


@dataclass
class _MemberState:
    cooldown_until: float | None = None
    cooldown_exception: BaseException | None = None
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
    if isinstance(exc, OutputTooLongError):
        return False
    return isinstance(exc, Exception)


def _metadata_matches(actual: Any, expected: str) -> bool:
    """Whether a retain item's metadata value matches a route's value.

    Retain metadata is free-form JSON while a route value is always a string, so
    compare on the string form. A list/tuple/set value matches when any of its
    entries does, which is what makes ``{"labels": ["pii", "eu"]}`` routable.
    """
    if isinstance(actual, (list, tuple, set, frozenset)):
        return any(_metadata_matches(entry, expected) for entry in actual)
    if actual is None or isinstance(actual, dict):
        return False
    if isinstance(actual, bool):
        # str(True) is "True"; JSON booleans should match "true"/"false".
        return str(actual).lower() == expected.lower()
    return str(actual) == expected


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
    """Route LLM calls across multiple members per the configured strategy."""

    def __init__(self, members: list[LLMProvider], strategy: LLMStrategyConfig) -> None:
        if not members:
            raise ValueError("MultiLLMProvider requires at least one member")
        self._members = members
        self._strategy = strategy
        # State belongs to this router instance. A second operation/router using
        # the same provider objects gets an independent cooldown history.
        self._states = [_MemberState() for _ in members]
        self._state_lock = threading.Lock()

        weights = strategy.weights or [1] * len(members)
        if len(weights) != len(members):
            raise ValueError(
                f"LLM strategy 'weights' has {len(weights)} entries but the chain has "
                f"{len(members)} members (primary + indexed); they must match."
            )
        self._scheduler = _WeightedRoundRobin(weights)

        if strategy.mode == LLM_STRATEGY_METADATA:
            for route in strategy.routes or []:
                if route.member >= len(members):
                    raise ValueError(
                        f"LLM metadata route {route.key}={route.value!r} selects member {route.member}, "
                        f"but the chain has members 0..{len(members) - 1}."
                    )

    # ── routing ────────────────────────────────────────────────────────────────

    def _member_order(self) -> list[int]:
        """Indices to try, in order, for one request."""
        n = len(self._members)
        if self._strategy.mode == LLM_STRATEGY_METADATA:
            # A metadata member is chosen per item by the retain path, which then
            # calls that member directly. Anything reaching the chain itself has
            # no item to route on, so it stays on the primary and does not fail
            # over into another member's lane.
            return [0]
        if self._strategy.mode == LLM_STRATEGY_FAILOVER:
            return list(range(n))
        start = self._scheduler.next()
        return [(start + i) % n for i in range(n)]

    def member_for_metadata(self, metadata: dict[str, Any] | None) -> LLMProvider | None:
        """The member selected by a retained item's metadata, or ``None``.

        ``None`` means "nothing to re-bind": either the chain is not in metadata
        mode, or no route matched and the caller's existing primary binding is
        already the right one. The first matching route in declared order wins,
        so overlapping routes are resolved by configuration order rather than
        rejected — one item is one prompt, so there is never more than one item
        to satisfy.
        """
        if self._strategy.mode != LLM_STRATEGY_METADATA or not metadata:
            return None
        for route in self._strategy.routes or []:
            if _metadata_matches(metadata.get(route.key), route.value):
                return self._members[route.member]
        return None

    async def _dispatch(self, method_name: str, **kwargs: Any) -> Any:
        last_exc: BaseException | None = None
        # Keep the request's original ordering across the one bounded inline
        # wait. Re-running _member_order() would rotate round-robin mid-request.
        order = self._member_order()
        inline_retry_used = False
        request_saved_failure: BaseException | None = None

        while True:
            attempted_member = False
            skipped_failure: BaseException | None = None
            for position, idx in enumerate(order):
                member = self._members[idx]
                label = getattr(member, "member_label", None) or ("primary" if idx == 0 else f"member-{idx}")
                with self._state_lock:
                    state = self._states[idx]
                    if state.probing:
                        if skipped_failure is None and state.cooldown_exception is not None:
                            skipped_failure = state.cooldown_exception
                        logger.debug(
                            "LLM member %d (%s/%s, label=%s) skipped: state=probing",
                            idx,
                            member.provider,
                            member.model,
                            label,
                        )
                        continue
                    if state.cooldown_until is not None:
                        remaining_seconds = state.cooldown_until - monotonic()
                        if remaining_seconds > 0:
                            if skipped_failure is None and state.cooldown_exception is not None:
                                skipped_failure = state.cooldown_exception
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

                attempted_member = True
                if probing:
                    logger.info(
                        "LLM member %d (%s/%s, label=%s) state=probing",
                        idx,
                        member.provider,
                        member.model,
                        label,
                    )
                try:
                    result = await getattr(member, method_name)(**kwargs)
                    with self._state_lock:
                        # A success that predates a newer quota observation must
                        # not erase it; only the matching generation may reopen.
                        if probing and state.generation == generation:
                            state.cooldown_until = None
                            state.cooldown_exception = None
                            logger.info(
                                "LLM member %d (%s/%s, label=%s) state=eligible after successful probe",
                                idx,
                                member.provider,
                                member.model,
                                label,
                            )
                    return result
                except BaseException as exc:  # noqa: BLE001 - exact object may be re-raised
                    if not isinstance(exc, Exception):
                        raise
                    classify_failure = getattr(member, "classify_failure", None)
                    failure = classify_failure(exc) if classify_failure is not None else None
                    if isinstance(failure, LLMTerminalFailure):
                        logger.warning(
                            "LLM member %d (%s/%s, label=%s) category=reauthentication_required; stopping operation",
                            idx,
                            member.provider,
                            member.model,
                            label,
                        )
                        # Identity, traceback, and provider remediation text are
                        # part of the existing exception contract.
                        raise

                    if isinstance(failure, LLMCooldownFailure) or probing:
                        delay = failure.retry_after_seconds if isinstance(failure, LLMCooldownFailure) else None
                        cooldown_source = "provider_retry_after"
                        if delay is None or not math.isfinite(delay) or delay < 0:
                            delay = _DEFAULT_COOLDOWN_SECONDS
                            cooldown_source = "default"
                        with self._state_lock:
                            state.cooldown_until = max(state.cooldown_until or 0.0, monotonic() + delay)
                            # Followers which find every member unavailable need
                            # the member's own nonterminal cause.  Keep the
                            # latest one with the lease it created so they can
                            # fail promptly after their one bounded wait.
                            state.cooldown_exception = exc
                            state.generation += 1
                        if probing:
                            # A half-open call is already this member's one
                            # recovery attempt.  If it fails, do not sleep and
                            # acquire another probe lease in the same request.
                            inline_retry_used = True
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

                    if not _should_failover(exc):
                        raise
                    last_exc = exc
                    remaining = len(order) - position - 1
                    logger.warning(
                        "LLM member %d (%s/%s, label=%s) failed on %s: %s%s",
                        idx,
                        member.provider,
                        member.model,
                        label,
                        method_name,
                        failure.category.value if failure is not None else exc,
                        f"; trying next member ({remaining} left)" if remaining else "; no members left",
                    )
                finally:
                    if probing:
                        with self._state_lock:
                            state.probing = False

            if skipped_failure is not None:
                request_saved_failure = skipped_failure
            with self._state_lock:
                now = monotonic()
                unavailable = all(state.probing or state.cooldown_until is not None for state in self._states)
                waits = [
                    (
                        _PROBE_POLL_SECONDS if state.probing else max(0.0, (state.cooldown_until or now) - now),
                        idx,
                    )
                    for idx, state in enumerate(self._states)
                    if state.probing or state.cooldown_until is not None
                ]

            if unavailable and waits:
                wait_seconds, earliest_idx = min(waits)
                earliest_member = self._members[earliest_idx]
                explicit_max_backoff = kwargs.get("max_backoff")
                configured_max_backoff = getattr(earliest_member, "max_backoff", None)
                effective_max_backoff = (
                    explicit_max_backoff
                    if explicit_max_backoff is not None
                    else configured_max_backoff
                    if configured_max_backoff is not None
                    else 30.0
                    if method_name == "call_with_tools"
                    else 60.0
                )

                if wait_seconds > effective_max_backoff:
                    wall_now = datetime.now(UTC)
                    # Saturate only the external timestamp; the monotonic
                    # eligibility deadline retains the provider's full delay.
                    max_delay = (datetime.max.replace(tzinfo=UTC) - wall_now).total_seconds() - 1.0
                    retry_at = wall_now + timedelta(seconds=min(wait_seconds, max_delay))
                    raise ProviderRateLimitResetError(
                        retry_at=retry_at,
                        message=f"All LLM members are cooling down; retry at {retry_at.isoformat()}.",
                    ) from None

                if not inline_retry_used:
                    inline_retry_used = True
                    await asyncio.sleep(wait_seconds)
                    continue

                # A concurrent half-open probe owns the only request lease. Do
                # not poll it: after the single bounded wait, fail with this
                # request's last error, or the nearest member's saved cooldown
                # cause.  A later request may probe again once it is eligible.
                if last_exc is not None:
                    raise last_exc
                if request_saved_failure is not None:
                    raise request_saved_failure

            # A probe owner may recover after this request observed and skipped
            # it but before the post-loop snapshot above.  With no member
            # attempted, spend the one inline budget on an immediate reroute;
            # after that budget is spent, surface the native cause captured at
            # the skip rather than polling or synthesizing an exhaustion error.
            if not attempted_member:
                if not inline_retry_used:
                    inline_retry_used = True
                    continue
                if request_saved_failure is not None:
                    raise request_saved_failure

            # One inline retry is the bound: a provider that immediately reports
            # another short reset does not cause an unbounded router retry loop.
            if last_exc is not None:
                raise last_exc
            if request_saved_failure is not None:
                raise request_saved_failure
            raise RuntimeError("MultiLLMProvider exhausted member routing without a result")

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
        return ConfiguredLLMProvider(cast(LLMProvider, self), config.llm_gemini_safety_settings, trace_ctx)

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

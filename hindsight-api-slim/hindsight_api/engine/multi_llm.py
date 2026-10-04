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

Prompt-prefix caching is member-aware too (see ``get_or_create_cached_prefix``):
each member that supports it creates its own cache handle, and ``_dispatch``
hands every member only its own handle, so a failover or round-robin member is
never sent another provider's cache name (#5123).
"""

import logging
import threading
import uuid
from typing import TYPE_CHECKING, Any

from ..config import LLM_STRATEGY_FAILOVER, LLM_STRATEGY_METADATA, LLMStrategyConfig
from .llm_wrapper import ConfiguredLLMProvider, LLMProvider, OutputTooLongError

if TYPE_CHECKING:
    from .llm_interface import LLMInterface
    from .llm_wrapper import LLMToolCallResult

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


class MemberCachedPrefixes:
    """Prompt-cache handles for a multi-LLM chain, one slot per member.

    A cache handle is provider-specific (a Gemini ``CachedContent`` name means
    nothing to any other member, or even to another Gemini account), so a chain
    cannot share one handle across its members. ``_dispatch`` resolves this to
    the handle of the member actually serving the request, or to no handle when
    that member has none.
    """

    __slots__ = ("_handles",)

    def __init__(self, handles: list[str | None]) -> None:
        self._handles = tuple(handles)

    def for_member(self, index: int) -> str | None:
        return self._handles[index] if index < len(self._handles) else None

    def __repr__(self) -> str:
        return f"MemberCachedPrefixes({list(self._handles)!r})"


async def get_or_create_cached_prefix(
    llm_config: Any,
    *,
    system_instruction: str,
    response_schema: Any | None = None,
) -> "str | MemberCachedPrefixes | None":
    """Cache handle for ``llm_config``'s stable prompt prefix, or ``None``.

    A single provider answers from its own ``_provider_impl``. A multi-LLM chain
    (bare or inside ``ConfiguredLLMProvider``) answers per member, because
    ``_provider_impl`` on a chain is only the primary's: asking the primary alone
    would leave a caching-capable failover / round-robin member uncached, and
    would send a caching primary's handle to members that cannot use it.
    """
    provider = llm_config
    if isinstance(provider, ConfiguredLLMProvider):
        provider = object.__getattribute__(provider, "_provider")
    if isinstance(provider, MultiLLMProvider):
        return await provider.get_or_create_cached_prefix(
            system_instruction=system_instruction,
            response_schema=response_schema,
        )
    provider_impl = getattr(llm_config, "_provider_impl", None)
    if provider_impl is None or not provider_impl.supports_prompt_caching():
        return None
    return await provider_impl.get_or_create_cached_prefix(
        system_instruction=system_instruction,
        response_schema=response_schema,
    )


class MultiLLMProvider:
    """Route LLM calls across multiple members per the configured strategy."""

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
        order = self._member_order()
        for position, idx in enumerate(order):
            member = self._members[idx]
            try:
                return await getattr(member, method_name)(**self._member_kwargs(idx, kwargs))
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

    @staticmethod
    def _member_kwargs(idx: int, kwargs: dict[str, Any]) -> dict[str, Any]:
        """``kwargs`` with a chain cache handle narrowed to member ``idx``'s own."""
        cached_prefix = kwargs.get("cached_prefix")
        if not isinstance(cached_prefix, MemberCachedPrefixes):
            return kwargs
        member_kwargs = dict(kwargs)
        handle = cached_prefix.for_member(idx)
        if handle is None:
            # This member has no cache: send the full, uncached prompt.
            member_kwargs.pop("cached_prefix")
            member_kwargs.pop("cached_prefix_message_count", None)
        else:
            member_kwargs["cached_prefix"] = handle
        return member_kwargs

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

    # ── prompt caching ─────────────────────────────────────────────────────────────

    def _cache_member_indices(self) -> list[int]:
        # Metadata mode keeps chain-level calls on the primary (routed items call
        # their member directly), so only the primary can use a chain handle.
        if self._strategy.mode == LLM_STRATEGY_METADATA:
            return [0]
        return list(range(len(self._members)))

    def supports_prompt_caching(self) -> bool:
        """Whether ANY member that can serve a chain call supports prefix caching."""
        return any(self._members[idx]._provider_impl.supports_prompt_caching() for idx in self._cache_member_indices())

    async def get_or_create_cached_prefix(
        self,
        *,
        system_instruction: str,
        response_schema: Any | None = None,
    ) -> MemberCachedPrefixes | None:
        """Per-member cache handles, or ``None`` when no member returned one.

        Every member that may serve a request and supports caching creates its
        own cache. A member whose cache lookup fails is logged and left uncached
        rather than failing the lookup for the whole chain.
        """
        handles: list[str | None] = [None] * len(self._members)
        for idx in self._cache_member_indices():
            impl = self._members[idx]._provider_impl
            if not impl.supports_prompt_caching():
                continue
            try:
                handles[idx] = await impl.get_or_create_cached_prefix(
                    system_instruction=system_instruction,
                    response_schema=response_schema,
                )
            except Exception:
                logger.exception("Cache prefix lookup failed for LLM member %d; it will run uncached", idx)
        if all(handle is None for handle in handles):
            return None
        return MemberCachedPrefixes(handles)

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

    @property
    def strategy(self) -> LLMStrategyConfig:
        return self._strategy

    def __getattr__(self, name: str) -> Any:
        # Anything not defined here (provider, model, api_key, base_url,
        # _provider_impl, mock helpers, ...) delegates to the primary member so
        # existing call sites keep working unchanged. The batch helpers above are
        # defined precisely because the primary is the wrong answer for them.
        return getattr(object.__getattribute__(self, "_members")[0], name)

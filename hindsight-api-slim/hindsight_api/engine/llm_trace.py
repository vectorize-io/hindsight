"""Per-bank model call tracing.

Fire-and-forget recording of every model call Hindsight makes (both successes
and failures) into the ``llm_requests`` table, per bank: LLM calls, embeddings,
reranker calls and decision-model calls. The kind of a call is read off its
scope (see ``call_type_clause``), so there is no separate column. Each row
captures the input, the output, token usage (input / visible output / cached /
thoughts / visible total), finish reason, and caller metadata. Controlled by
``HINDSIGHT_API_LLM_TRACE_ENABLED``; ``HINDSIGHT_API_LLM_TRACE_SCOPES`` and
``HINDSIGHT_API_LLM_TRACE_EXCLUDE_SCOPES`` pick which scopes are recorded.

Rows are buffered and written in batches (one ``executemany`` per tenant schema)
every ``FLUSH_INTERVAL_SECONDS`` or ``FLUSH_MAX_ROWS`` rows, so tracing the much
more frequent embedding and rerank calls costs one INSERT per batch rather than
one per call. A crash loses the rows still in the buffer.

This plugs into the OpenTelemetry **GenAI** recording pattern: providers already
call ``tracing.get_span_recorder().record_llm_call(...)`` on success, so the DB
tracer is registered as one of those recorders (alongside the OTLP span
exporter) rather than hooking the call path with custom code. Failures, which
providers don't report to the recorder, are forwarded from the LLM wrapper.

Bank/operation attribution is carried via a ContextVar, bound for a whole
operation run (``operation_run`` / ``operation_trace_context``) so the embedding,
rerank and decision calls inside it join the run's trace. Outside any run
``bank_id`` falls back to the request's current bank, else NULL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from ..schema_context import get_current_bank_id
from ..tracing import get_span_recorder
from .background_writes import PendingWrites
from .db_utils import acquire_with_retry

logger = logging.getLogger(__name__)


def _llm_requests_persistable() -> bool:
    """Whether the ``llm_requests`` table exists on the active backend.

    ``llm_requests`` is PostgreSQL-only: its migration is ``run_for_dialect(pg=...)``
    with the Oracle slot intentionally absent, and MaintenanceLoop skips its
    retention sweep on Oracle for the same reason. On Oracle the table does not
    exist, so best-effort trace writes must be skipped rather than attempted —
    otherwise every LLM call fires an INSERT that fails with ORA-00903 and spams
    the error log. Mirrors the ``_is_oracle()`` gate in MaintenanceLoop.start.
    """
    from .schema import _is_oracle

    return not _is_oracle()


# ── bank/operation attribution (carried across the async call chain) ──────────


@dataclass
class LLMTraceContext:
    """Attribution for in-flight LLM calls, bound by ``ConfiguredLLMProvider``.

    ``trace_id`` and ``operation_span_id`` are generated once per operation
    invocation (one ``with_config`` call), so every LLM call of a single
    reflect/retain/consolidation run shares them — reproducing the OTel
    parent (operation span) → children (LLM calls) hierarchy in the DB.
    """

    bank_id: str | None = None
    operation: str | None = None  # "retain" | "reflect" | "consolidation" | ...
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    operation_span_id: str | None = None
    # Memory_units this operation produced/consumed, accumulated at the DB-write
    # sites and flushed onto every row of the trace at operation end (see
    # LLMTraceRecorder.attach_memory_ids). Lets a retain/consolidation trace map
    # to the memories it created (outputs) and consumed (source inputs).
    created_memory_ids: list[str] = field(default_factory=list)
    source_memory_ids: list[str] = field(default_factory=list)


_trace_ctx: ContextVar[LLMTraceContext | None] = ContextVar("hindsight_llm_trace_ctx", default=None)

# Per-call requested parameters (max_completion_tokens, temperature, response
# schema, tool_choice). Set by ``LLMProvider.call`` around the provider
# delegation so the recorder can attach them even though success is reported by
# the provider. Only includes values the caller actually set — never nulls.
_request_ctx: ContextVar[dict[str, Any] | None] = ContextVar("hindsight_llm_request_ctx", default=None)

# Per-call caller metadata (e.g. document_id for retain extraction). Set by
# engine code around a specific LLM call; merged into the row's metadata on top
# of the operation-level LLMTraceContext.metadata.
_call_metadata_ctx: ContextVar[dict[str, Any] | None] = ContextVar("hindsight_llm_call_metadata_ctx", default=None)


@dataclass
class LLMResponseUsage:
    """Provider-reported token usage for the in-flight LLM call.

    Stashed by provider implementations as soon as a response is received —
    *before* local JSON parsing / schema validation, which may still fail. The
    wrapper reads it to attach real token counts to an error trace when the
    provider call itself succeeded but the structured output couldn't be parsed
    or validated (providers charge for those tokens regardless). See #2387.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    thoughts_tokens: int = 0


# Per-call provider usage, set by providers right after a response is received.
_response_usage_ctx: ContextVar[LLMResponseUsage | None] = ContextVar("hindsight_llm_response_usage_ctx", default=None)


def set_response_usage(usage: LLMResponseUsage | None) -> Token:
    """Bind provider-reported usage for the current call. Returns a reset token."""
    return _response_usage_ctx.set(usage)


def stash_response_usage(usage: LLMResponseUsage | None) -> None:
    """Record provider-reported usage so an error trace can attach it later.

    Called by provider implementations once a response (with usage) is in hand,
    before parsing/validation that may raise. Overwrites any prior value from an
    earlier retry attempt so the last attempt's usage wins.
    """
    _response_usage_ctx.set(usage)


def reset_response_usage(token: Token) -> None:
    """Unwind a binding made by :func:`set_response_usage`."""
    _response_usage_ctx.reset(token)


def current_response_usage() -> LLMResponseUsage | None:
    """Return the active call's provider-reported usage, or None."""
    return _response_usage_ctx.get()


@dataclass
class ProviderInputTokens:
    """Input tokens the provider reported for the embedding or rerank call in flight.

    A mutable sink rather than a value: a backend may fan one call out over several
    tasks, and each copies the context — they all add to this one object.
    """

    count: int = 0


_input_tokens_ctx: ContextVar[ProviderInputTokens | None] = ContextVar("hindsight_input_tokens_ctx", default=None)


def add_provider_input_tokens(count: int | None) -> None:
    """Report billed input tokens for the traced call in flight. No-op outside one.

    ``None`` (a provider that did not report the count) adds nothing: a trace must
    never fail the call it traces (#5941).
    """
    sink = _input_tokens_ctx.get()
    if sink is not None and count:
        sink.count += count


_T = TypeVar("_T")


async def trace_model_call(
    owner: Any,
    scope: str,
    input_count: int,
    call: Callable[[], Awaitable[_T]],
    *,
    request: Any = None,
    output_of: Callable[[_T], Any] | None = None,
) -> _T:
    """Run an embedding or rerank ``call`` and trace it next to the LLM calls.

    ``owner`` is the backend (its ``provider_name`` and ``model`` / ``model_name``
    label the row). ``input_count`` is how many texts or documents went in, which is
    what a provider that does not bill by token bills by. ``request`` is what the row
    shows as its input and ``output_of`` picks what it shows as output from the result
    (rerank scores; an embedding's vectors would only be noise, so it shows none).
    Both are serialized when the row is written, off the caller's path, and cut to
    ``HINDSIGHT_API_LLM_TRACE_MAX_CHARS`` like any other row.
    """
    result: Any = None
    tokens = ProviderInputTokens()
    reset = _input_tokens_ctx.set(tokens)
    started = time.monotonic()
    error: BaseException | None = None
    try:
        result = await call()
        return result
    except BaseException as e:
        error = e
        raise
    finally:
        _input_tokens_ctx.reset(reset)
        model = getattr(owner, "model", None)
        if not isinstance(model, str):
            model = getattr(owner, "model_name", None)
        get_span_recorder().record_llm_call(
            # getattr: callers pass any duck-typed backend, and a trace must never fail a call.
            provider=str(getattr(owner, "provider_name", None) or type(owner).__name__),
            model=model if isinstance(model, str) else "",
            scope=scope,
            messages=[],
            structured_input=request,
            response_content=output_of(result) if output_of is not None and error is None else None,
            input_tokens=tokens.count,
            output_tokens=0,
            duration=time.monotonic() - started,
            error=error,
            input_count=input_count,
        )


@dataclass
class LLMQueueWait:
    """Accumulator for time a call spent waiting on LLM concurrency permits."""

    seconds: float = 0.0


# Optional per-call sink for permit wait time, bound by a caller that wants to
# report it. Unset by default, which makes record_queue_wait a no-op.
_queue_wait_ctx: ContextVar[LLMQueueWait | None] = ContextVar("hindsight_llm_queue_wait_ctx", default=None)


def set_queue_wait_sink(sink: LLMQueueWait | None) -> Token:
    """Collect permit wait time for calls made under this context. Returns a reset token.

    Without a sink, ``duration`` for an LLM call conflates two very different things:
    time queued behind a concurrency permit and time the request was actually in
    flight. That ambiguity sent issue #3881 chasing the provider. The worker path has
    the ``.queued`` stage breadcrumb for this, but ``set_stage`` is a no-op outside a
    worker task, so a synchronous HTTP request had no signal at all.
    """
    return _queue_wait_ctx.set(sink)


def reset_queue_wait_sink(token: Token) -> None:
    """Unwind a binding made by :func:`set_queue_wait_sink`."""
    _queue_wait_ctx.reset(token)


def record_queue_wait(seconds: float) -> None:
    """Add permit wait time to the active sink. No-op when no caller bound one.

    Accumulates rather than overwrites: a provider that owns its retry loop
    re-acquires permits per attempt, and the caller wants the total.
    """
    sink = _queue_wait_ctx.get()
    if sink is not None:
        sink.seconds += seconds


def set_trace_context(ctx: LLMTraceContext | None) -> Token:
    """Bind trace attribution to the current context. Returns a reset token."""
    return _trace_ctx.set(ctx)


def reset_trace_context(token: Token) -> None:
    """Unwind a binding made by :func:`set_trace_context`."""
    _trace_ctx.reset(token)


def set_request_context(params: dict[str, Any] | None) -> Token:
    """Bind the current LLM call's requested parameters. Returns a reset token."""
    return _request_ctx.set(params)


def reset_request_context(token: Token) -> None:
    """Unwind a binding made by :func:`set_request_context`."""
    _request_ctx.reset(token)


def current_request_context() -> dict[str, Any] | None:
    """Return the active call's requested parameters, or None."""
    return _request_ctx.get()


def set_call_metadata(metadata: dict[str, Any] | None) -> Token:
    """Bind per-call caller metadata (e.g. ``{"document_id": ...}``)."""
    return _call_metadata_ctx.set(metadata)


def reset_call_metadata(token: Token) -> None:
    """Unwind a binding made by :func:`set_call_metadata`."""
    _call_metadata_ctx.reset(token)


def current_call_metadata() -> dict[str, Any] | None:
    """Return the active call's caller metadata, or None."""
    return _call_metadata_ctx.get()


def current_trace_context() -> LLMTraceContext | None:
    """Return the active trace attribution, or None outside a traced context."""
    return _trace_ctx.get()


def new_trace_context(
    bank_id: str | None, operation: str | None, metadata: dict[str, Any] | None = None
) -> LLMTraceContext:
    """A fresh operation run: its own trace id and operation span."""
    return LLMTraceContext(
        bank_id=bank_id,
        operation=operation,
        metadata=dict(metadata or {}),
        trace_id=str(uuid.uuid4()),
        operation_span_id=str(uuid.uuid4()),
    )


@contextmanager
def operation_run(bank_id: str | None, operation: str) -> Iterator[None]:
    """Bind a fresh run for ``operation`` unless one is already bound (then join it)."""
    token = _trace_ctx.set(new_trace_context(bank_id, operation)) if _trace_ctx.get() is None else None
    try:
        yield
    finally:
        if token is not None:
            _trace_ctx.reset(token)


def operation_trace_context(
    bank_id: str | None, operation: str | None, metadata: dict[str, Any] | None
) -> LLMTraceContext | None:
    """The trace a ``with_config`` wrapper attributes its LLM calls to.

    One trace + operation span per ``with_config()`` call — i.e. per operation
    invocation — so a reflect/retain/consolidation run groups its calls as parent
    (operation) → children (LLM calls). Except when the same operation already bound
    a trace for its whole run (a mental-model refresh wraps its reflect pass AND the
    page write after it): the wrapper joins that run instead of opening a second one,
    or the page's embedding would land in a run of its own.
    """
    if bank_id is None and operation is None and not metadata:
        return None
    bound = _trace_ctx.get()
    if bound is not None and bound.operation == operation and bound.bank_id == bank_id and not metadata:
        return bound
    return new_trace_context(bank_id, operation, metadata)


def trace_context_of(llm_config: Any) -> LLMTraceContext | None:
    """Return a configured provider's operation trace context, or None.

    Real providers expose ``trace_context()`` (``ConfiguredLLMProvider``); test
    or mock substitutes may not, so this degrades gracefully rather than raising
    — tracing is best-effort and must never break an operation.
    """
    getter = getattr(llm_config, "trace_context", None)
    return getter() if callable(getter) else None


def record_created_memory_ids(ids: Iterable[str]) -> None:
    """Accumulate output memory_units onto the active operation trace.

    No-op outside a traced operation context (e.g. tracing disabled). Child
    asyncio tasks inherit the same ``LLMTraceContext`` object, so appends from
    parallel consolidation batches land on one shared list.
    """
    ctx = _trace_ctx.get()
    if ctx is not None:
        ctx.created_memory_ids.extend(str(i) for i in ids)


def record_source_memory_ids(ids: Iterable[str]) -> None:
    """Accumulate consumed/source memory_units onto the active operation trace.

    No-op outside a traced operation context.
    """
    ctx = _trace_ctx.get()
    if ctx is not None:
        ctx.source_memory_ids.extend(str(i) for i in ids)


# ── serialization helpers ─────────────────────────────────────────────────────


def _json_default(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, bytes):
        return "<bytes>"
    if isinstance(obj, set):
        return list(obj)
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except Exception:
            return str(obj)
    return str(obj)


def _safe_json(data: Any, max_chars: int) -> str | None:
    """Serialize ``data`` to a JSON string, truncating beyond ``max_chars``.

    Returns None on total failure. Truncation preserves valid JSON by wrapping
    the oversized payload in a marker object with a preview.
    """
    if data is None:
        return None
    try:
        serialized = json.dumps(data, default=_json_default)
    except Exception:
        logger.debug("Failed to serialize llm trace data", exc_info=True)
        try:
            serialized = json.dumps(str(data))
        except Exception:
            return None
    if max_chars and max_chars > 0 and len(serialized) > max_chars:
        return json.dumps({"_truncated": True, "_original_chars": len(serialized), "preview": serialized[:max_chars]})
    return serialized


# ── record ────────────────────────────────────────────────────────────────────


@dataclass
class LLMRequestRecord:
    """A single LLM request trace row."""

    provider: str
    model: str | None
    scope: str
    status: str  # "success" | "error"
    started_at: datetime
    ended_at: datetime
    bank_id: str | None = None
    operation: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    input: Any = None
    output: Any = None
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    thoughts_tokens: int | None = None
    total_tokens: int | None = None
    llm_info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        return int((self.ended_at - self.started_at).total_seconds() * 1000)


# ── read models (returned by MemoryEngine query methods, served by the API) ───


class LLMRequestEntry(BaseModel):
    """A single LLM request trace row, as returned by the read API."""

    id: str
    bank_id: str | None
    operation: str | None
    scope: str | None
    trace_id: str | None
    span_id: str | None
    parent_span_id: str | None
    provider: str | None
    model: str | None
    status: str
    started_at: str | None
    ended_at: str | None
    duration_ms: int | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    thoughts_tokens: int | None
    total_tokens: int | None
    # Arbitrary JSON (message list, string, or object) — open `Any` so the
    # OpenAPI schema stays a plain open type the Go SDK generator can model.
    input: Any = None
    output: Any = None
    error: str | None
    llm_info: dict[str, Any]
    metadata: dict[str, Any]


class LLMRequestListResponse(BaseModel):
    """Paginated list of LLM request traces for a bank."""

    bank_id: str
    total: int
    limit: int
    offset: int
    items: list[LLMRequestEntry]


class LLMRequestTokenSums(BaseModel):
    """Token totals for a time bucket."""

    input: int
    output: int
    cached: int
    # Optional so a current generated client can still parse a stats response
    # from a server predating reasoning usage, which omits the field entirely.
    # This server always sends it (the SUM is COALESCEd to 0).
    thoughts: int | None = None
    total: int


class LLMRequestStatsBucket(BaseModel):
    """A single time bucket in LLM request stats."""

    time: str
    statuses: dict[str, int]
    total: int
    tokens: LLMRequestTokenSums


class LLMRequestStatsResponse(BaseModel):
    """LLM request counts and token sums grouped by time bucket."""

    bank_id: str
    period: str
    trunc: str
    start: str
    buckets: list[LLMRequestStatsBucket]


# ── recorder / writer ─────────────────────────────────────────────────────────

# Scopes of the calls that are not chat completions. Every model call Hindsight pays for lands
# in ``llm_requests``, told apart by scope alone: ``embedding_document`` / ``embedding_query``,
# ``rerank``, and ``decision_*`` for decision-model questions. Anything else is an LLM call.
EMBEDDING_SCOPE_PREFIX = "embedding_"
RERANK_SCOPE = "rerank"
DECISION_SCOPE_PREFIX = "decision_"


def _like_prefix(prefix: str) -> str:
    return prefix.replace("_", "\\_")


# What ``call_type`` filters on, as SQL over ``scope``. Derived rather than stored, so a row
# needs no second label and an old row is classified the same way as a new one.
_CALL_TYPE_SQL = {
    # ``_`` is a LIKE wildcard; escaped so the SQL matches the literal prefix, exactly
    # like the str.startswith the control plane and genai_operation_name use.
    "embedding": f"scope LIKE '{_like_prefix(EMBEDDING_SCOPE_PREFIX)}%' ESCAPE '\\'",
    "rerank": f"scope = '{RERANK_SCOPE}'",
    "decision": f"scope LIKE '{_like_prefix(DECISION_SCOPE_PREFIX)}%' ESCAPE '\\'",
}
# Everything else is an LLM call. COALESCE: a NULL scope would make NOT (...) NULL, dropping
# the row from every kind, where the control plane counts it as an LLM call.
_CALL_TYPE_SQL["llm"] = "NOT (" + " OR ".join(_CALL_TYPE_SQL.values()).replace("scope", "COALESCE(scope, '')") + ")"
LLMCallType = Literal["llm", "embedding", "rerank", "decision"]


def genai_operation_name(scope: str | None) -> str:
    """The OTel GenAI ``gen_ai.operation.name`` for a call, read off its scope like its kind.

    ``embeddings`` and ``chat`` are the semantic-convention values; reranker and
    decision-model calls have none, so they get their own rather than passing as chat.
    """
    if not scope:
        return "chat"
    if scope.startswith(EMBEDDING_SCOPE_PREFIX):
        return "embeddings"
    if scope == RERANK_SCOPE:
        return "rerank"
    if scope.startswith(DECISION_SCOPE_PREFIX):
        return "decision"
    return "chat"


def call_type_clause(call_type: str) -> str:
    """SQL condition over ``scope`` selecting one kind of call. Raises ValueError for an unknown kind."""
    try:
        return _CALL_TYPE_SQL[call_type]
    except KeyError:
        raise ValueError(f"call_type must be one of {', '.join(_CALL_TYPE_SQL)}, got {call_type!r}") from None


def _scope_matches(scope: str, patterns: frozenset[str]) -> bool:
    """Whether ``scope`` is one of ``patterns``; a pattern ending in ``*`` matches a prefix."""
    return any(scope.startswith(p[:-1]) if p.endswith("*") else scope == p for p in patterns)


_INSERT_COLUMNS = """
    (id, bank_id, operation, scope, trace_id, span_id, parent_span_id,
     provider, model, status,
     started_at, ended_at, duration_ms,
     input_tokens, output_tokens, cached_tokens, thoughts_tokens, total_tokens,
     input, output, error, llm_info, metadata)
VALUES
    ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
     $11, $12, $13, $14, $15, $16, $17, $18,
     $19::jsonb, $20::jsonb, $21, $22::jsonb, $23::jsonb)
"""

# Bound on how long shutdown waits for the last trace rows, so a stuck database cannot hang it.
_SHUTDOWN_DRAIN_SECONDS = 5.0


@dataclass(frozen=True)
class _BufferedRow:
    """A trace row waiting to be written, with the tenant schema it was recorded under."""

    schema: str
    record: LLMRequestRecord


class LLMTraceRecorder:
    """GenAI span recorder that writes per-bank model-call traces to ``llm_requests``.

    Implements ``record_llm_call`` so it can be registered with
    :func:`hindsight_api.tracing.register_span_recorder`. Rows are buffered in
    memory and written in batches, one connection and one ``executemany`` per
    batch, so a recall's embedding, rerank and decision calls do not each take a
    pooled connection. The cost of that: rows still in the buffer when the
    process dies are lost. Writes never surface errors into the calling path.
    Retention of old rows is handled by the background :class:`MaintenanceLoop`.
    """

    # How long a row may wait in the buffer, and how many trigger a write at once.
    FLUSH_INTERVAL_SECONDS = 0.5
    FLUSH_MAX_ROWS = 500

    def __init__(
        self,
        pool_getter: Callable[[], Any],
        schema_getter: Callable[[], str],
        enabled: bool,
        allowed_scopes: list[str],
        max_chars: int = 50000,
        excluded_scopes: list[str] | None = None,
    ) -> None:
        self._pool_getter = pool_getter
        self._schema_getter = schema_getter
        self._enabled = enabled
        self._allowed_scopes: frozenset[str] | None = frozenset(allowed_scopes) if allowed_scopes else None
        self._excluded_scopes = frozenset(excluded_scopes or ())
        self._max_chars = max_chars
        # Rows waiting to be written, each with the tenant schema it was recorded under: the
        # write runs on a background task, outside the request that knew the schema. A thread
        # lock, not an asyncio one: worker threads run their own event loops and record here too.
        self._buffer: list[_BufferedRow] = []
        self._buffer_lock = threading.Lock()
        self._flush_scheduled = False
        # In-flight background tasks: batch writes and flush timers under the default key,
        # memory-id patches under their own key (they wait on the default one).
        self._writes = PendingWrites("LLM trace write")
        # Trace ids that have actually produced a row. `_writes` cannot answer this: it is
        # emptied as writes complete, so an absent entry means "nothing in flight", not "nothing
        # was ever written". Without the distinction, `attach_memory_ids` issues an UPDATE for
        # every operation that created memories -- including one with no traced call, where it
        # matches zero rows and its only effect is to take a pooled connection. (A retain now
        # always has rows, its embeddings, so this mostly skips runs whose scopes are excluded.)
        # Bounded, and only ever holds ids: a trace id is ~36 bytes and this is capped, so it
        # cannot grow with traffic.
        self._rows_written: OrderedDict[str, None] = OrderedDict()

    def _writable(self) -> Any | None:
        """Return the pool to write through, or None if writing isn't possible.

        Covers the two lifecycle windows in which best-effort trace writes must
        be skipped rather than attempted: before the backend pool is created
        (``initialize()`` verifies the LLM before the DB is up) and during/after
        shutdown. Writes already in flight need no handling — the pools close
        gracefully, waiting for their connections to be released.
        """
        pool = self._pool_getter()
        if pool is None:
            return None
        # Backends declare readiness explicitly; a raw pool (some callers pass
        # one directly) has no lifecycle flag and is assumed usable.
        from .db.base import DatabaseBackend

        if isinstance(pool, DatabaseBackend) and not pool.is_ready:
            return None
        return pool

    def is_enabled(self, scope: str) -> bool:
        """Whether tracing is active for the given call scope."""
        if not self._enabled:
            return False
        if not _llm_requests_persistable():
            return False
        if _scope_matches(scope, self._excluded_scopes):
            return False
        if self._allowed_scopes is not None:
            return _scope_matches(scope, self._allowed_scopes)
        return True

    # ── GenAI recorder interface ──────────────────────────────────────────────

    def record_llm_call(
        self,
        provider: str,
        model: str,
        scope: str,
        messages: list[dict[str, Any]],
        response_content: Any = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration: float = 0.0,
        finish_reason: str | None = None,
        error: BaseException | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        cached_tokens: int = 0,
        thoughts_tokens: int | None = None,
        input_count: int | None = None,
        structured_input: Any = None,
        **_extra: Any,
    ) -> None:
        """Build a trace record from a GenAI call and schedule a DB write."""
        if not self.is_enabled(scope):
            return

        ctx = current_trace_context()
        ended_at = datetime.now(timezone.utc)
        started_at = ended_at - timedelta(seconds=max(0.0, duration))

        # Operation-level metadata + any per-call metadata (e.g. document_id).
        metadata = dict(ctx.metadata) if ctx else {}
        call_metadata = current_call_metadata()
        if call_metadata:
            metadata.update(call_metadata)

        llm_info: dict[str, Any] = {}
        request_params = current_request_context()
        if request_params:
            llm_info["request"] = dict(request_params)
        if finish_reason:
            llm_info["finish_reason"] = finish_reason
        if tool_calls:
            llm_info["tool_calls"] = [tc.get("name", "") for tc in tool_calls]
        if input_count is not None:
            llm_info["input_count"] = input_count

        record = LLMRequestRecord(
            provider=provider,
            model=model,
            scope=scope,
            status="error" if error is not None else "success",
            started_at=started_at,
            ended_at=ended_at,
            # Outside any operation run (a call no operation binds one for) the bank still
            # comes from the bank-scoped engine call that is running.
            bank_id=(ctx.bank_id if ctx else None) or get_current_bank_id(),
            operation=ctx.operation if ctx else None,
            # OTel-style hierarchy: all calls of one operation invocation share
            # the context's trace_id and point at its operation span; this call
            # gets its own span_id.
            trace_id=ctx.trace_id if ctx else None,
            span_id=str(uuid.uuid4()),
            parent_span_id=ctx.operation_span_id if ctx else None,
            # A copy: the row is serialized later, in a background write, and the
            # reflect loop keeps appending to this same list — its own tool call,
            # the tool result — so a reference would record messages sent AFTER
            # this call as if they had been part of its prompt.
            # A call that is not a chat (decision, embedding, rerank) hands its request as a
            # structure; ``messages`` then only carries the string form other recorders need.
            input=(
                structured_input
                if structured_input is not None
                else list(messages)
                if isinstance(messages, list)
                else messages
            ),
            output=None if error is not None else response_content,
            error=f"{type(error).__name__}: {error}" if error is not None else None,
            input_tokens=input_tokens or None,
            output_tokens=output_tokens or None,
            cached_tokens=cached_tokens or None,
            # ``or None`` like its siblings: a provider that reports no reasoning
            # is indistinguishable from one that reports nothing at all (a
            # transport failure stashes no usage and arrives here as 0), so a
            # stored 0 would claim a count nobody made. NULL reads uniformly as
            # "no reasoning usage reported" across old and new rows alike.
            thoughts_tokens=thoughts_tokens or None,
            total_tokens=(input_tokens + output_tokens) or None,
            llm_info=llm_info,
            metadata=metadata,
        )
        self._record_fire_and_forget(record)

    def _record_fire_and_forget(self, record: LLMRequestRecord) -> None:
        """Buffer a trace row; a background task writes it within ``FLUSH_INTERVAL_SECONDS``."""
        entry = _BufferedRow(self._schema_getter(), record)
        with self._buffer_lock:
            self._buffer.append(entry)
            batch = self._take_locked() if len(self._buffer) >= self.FLUSH_MAX_ROWS else None
            start_timer = batch is None and not self._flush_scheduled
            if start_timer:
                self._flush_scheduled = True
        if record.trace_id:
            self._mark_rows_written(record.trace_id)
        if batch:
            self._writes.schedule(self._write_batch(batch))
        elif start_timer and not self._writes.schedule(self._flush_later()):
            with self._buffer_lock:
                self._flush_scheduled = False

    def _take_locked(self) -> list[_BufferedRow]:
        batch, self._buffer = self._buffer, []
        return batch

    async def _flush_later(self) -> None:
        try:
            await asyncio.sleep(self.FLUSH_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            # The loop this timer ran on is going away (a worker thread's asyncio.run ending).
            # The rows stay buffered for the next timer, started by the next recorded call.
            with self._buffer_lock:
                self._flush_scheduled = False
            raise
        with self._buffer_lock:
            self._flush_scheduled = False
            batch = self._take_locked()
        if batch:
            await self._write_batch(batch)

    async def flush(self) -> None:
        """Write every buffered row now, and wait for the batch writes already under way."""
        with self._buffer_lock:
            batch = self._take_locked()
        if batch:
            await self._write_batch(batch)
        await self._writes.drain()

    async def flush_on_shutdown(self) -> None:
        """Write what is buffered and wait for every pending write; call before the pool closes.

        Bounded, so a stuck database cannot hang shutdown.
        """
        with self._buffer_lock:
            batch = self._take_locked()
        if batch:
            self._writes.schedule(self._write_batch(batch))
        await self._writes.drain_all(_SHUTDOWN_DRAIN_SECONDS)

    _ROWS_WRITTEN_MAX = 4096

    def _mark_rows_written(self, trace_id: str) -> None:
        self._rows_written[trace_id] = None
        self._rows_written.move_to_end(trace_id)
        while len(self._rows_written) > self._ROWS_WRITTEN_MAX:
            # Evicting the oldest can only cause a MISSED patch on a very long-lived trace, never
            # a wrong one -- and the patch is best-effort metadata either way.
            self._rows_written.popitem(last=False)

    def _row(self, record: LLMRequestRecord) -> list[Any]:
        """Positional INSERT parameters, in ``_INSERT_COLUMNS`` order."""
        return [
            uuid.uuid4(),
            record.bank_id,
            record.operation,
            record.scope,
            record.trace_id,
            record.span_id,
            record.parent_span_id,
            record.provider,
            record.model,
            record.status,
            record.started_at,
            record.ended_at,
            record.duration_ms,
            record.input_tokens,
            record.output_tokens,
            record.cached_tokens,
            record.thoughts_tokens,
            record.total_tokens,
            _safe_json(record.input, self._max_chars),
            _safe_json(record.output, self._max_chars),
            record.error,
            _safe_json(record.llm_info, self._max_chars) or "{}",
            _safe_json(record.metadata, self._max_chars) or "{}",
        ]

    async def _write_batch(self, batch: list[_BufferedRow]) -> None:
        """Write buffered rows, one ``executemany`` per tenant schema. Errors are logged, never raised."""
        pool = self._writable()
        if pool is None:
            logger.debug("LLM trace skipped: pool not available")
            return
        by_schema: dict[str, list[list[Any]]] = {}
        for row in batch:
            by_schema.setdefault(row.schema, []).append(self._row(row.record))
        try:
            async with acquire_with_retry(pool, max_retries=1) as conn:
                for schema, rows in by_schema.items():
                    query = f"INSERT INTO {schema}.llm_requests {_INSERT_COLUMNS}"
                    try:
                        await conn.executemany(query, rows)
                    except Exception as e:
                        # executemany is all-or-nothing: one row Postgres refuses (say, a
                        # payload jsonb rejects) would drop every other bank's rows in the
                        # batch with it. Retry one by one so only the bad row is lost, as it
                        # was when each call wrote its own row.
                        logger.warning(f"LLM trace batch write failed for {len(rows)} row(s) in schema={schema}: {e}")
                        for row in rows:
                            try:
                                await conn.execute(query, *row)
                            except Exception as row_error:
                                logger.warning(f"LLM trace write failed in schema={schema}: {row_error}")
        except Exception as e:
            logger.warning(f"LLM trace write failed for {len(batch)} row(s): {e}")

    def attach_memory_ids(
        self,
        trace_ctx: LLMTraceContext | None,
        *,
        created: list[str] | None = None,
        source: list[str] | None = None,
    ) -> None:
        """Map a finished operation's memory_units onto every row of its trace.

        Merges the explicitly passed ids with any accumulated on the context
        (``record_created_memory_ids`` / ``record_source_memory_ids``), de-dupes
        preserving order, and patches ``metadata.memory_ids`` (outputs created)
        and ``metadata.source_memory_ids`` (inputs consumed) on all rows sharing
        the trace_id. No-op when tracing is off or nothing was produced.

        Fire-and-forget: the snapshotted patch is applied on a background task so
        the retain/consolidation operation never waits on the trace write. The
        ids are snapshotted synchronously here because the caller may reset the
        context immediately after.
        """
        if not self._enabled or not _llm_requests_persistable() or trace_ctx is None or not trace_ctx.trace_id:
            return
        created_ids = list(dict.fromkeys([*(created or []), *trace_ctx.created_memory_ids]))
        source_ids = list(dict.fromkeys([*(source or []), *trace_ctx.source_memory_ids]))
        patch: dict[str, Any] = {}
        if created_ids:
            patch["memory_ids"] = created_ids
        if source_ids:
            patch["source_memory_ids"] = source_ids
        if not patch:
            return
        # Nothing was traced, so there is no row to patch: memories were created, so `patch` is
        # non-empty, but every call of the run was excluded by scope and the UPDATE would match
        # zero rows.
        if trace_ctx.trace_id not in self._rows_written:
            return
        # Not under the default key: this task drains that key itself, and a task
        # waiting on its own key would wait on itself forever.
        self._writes.schedule(self._attach_memory_ids(trace_ctx.bank_id, trace_ctx.trace_id, patch), key="memory_ids")

    async def _attach_memory_ids(self, bank_id: str | None, trace_id: str, patch: dict[str, Any]) -> None:
        """Background worker: flush this trace's writes, then patch its rows."""
        # The trace rows are buffered; write them first so the UPDATE patches rows
        # that already exist rather than racing ahead of them.
        await self.flush()
        pool = self._writable()
        if pool is None:
            logger.debug("LLM trace memory_id attach skipped: pool not available")
            return
        try:
            schema = self._schema_getter()
            table = f"{schema}.llm_requests"
            async with acquire_with_retry(pool, max_retries=1) as conn:
                await conn.execute(
                    f"UPDATE {table} SET metadata = metadata || $3::jsonb WHERE bank_id = $1 AND trace_id = $2",
                    bank_id,
                    trace_id,
                    json.dumps(patch),
                )
        except Exception as e:
            logger.warning(f"LLM trace memory_id attach failed for trace={trace_id}: {e}")

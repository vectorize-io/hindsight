"""Per-bank LLM request tracing.

Opt-in, fire-and-forget recording of every LLM call Hindsight makes (both
successes and failures) into the ``llm_requests`` table, per bank. Each row
captures the input messages, the model output, token usage (input / output /
cached / total), finish reason, and caller metadata. Disabled by default —
controlled by ``HINDSIGHT_API_LLM_TRACE_ENABLED``.

This plugs into the OpenTelemetry **GenAI** recording pattern: providers already
call ``tracing.get_span_recorder().record_llm_call(...)`` on success, so the DB
tracer is registered as one of those recorders (alongside the OTLP span
exporter) rather than hooking the call path with custom code. Failures, which
providers don't report to the recorder, are forwarded from the LLM wrapper.

Bank/operation attribution is carried via a ContextVar set by
``ConfiguredLLMProvider`` (see ``llm_wrapper.py``); outside a traced context
``bank_id`` is recorded as NULL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Callable, Iterable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel

from .db_utils import acquire_with_retry

logger = logging.getLogger(__name__)

_SWEEP_INTERVAL_SECONDS = 3600  # Run retention sweep every hour


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


class LLMTraceRecorder:
    """GenAI span recorder that writes per-bank LLM traces to ``llm_requests``.

    Implements ``record_llm_call`` so it can be registered with
    :func:`hindsight_api.tracing.register_span_recorder`. Writes are
    fire-and-forget and never surface errors into the calling path; an optional
    retention sweep deletes rows older than ``retention_days``.
    """

    def __init__(
        self,
        pool_getter: Callable[[], Any],
        schema_getter: Callable[[], str],
        enabled: bool,
        allowed_scopes: list[str],
        retention_days: int = -1,
        max_chars: int = 50000,
    ) -> None:
        self._pool_getter = pool_getter
        self._schema_getter = schema_getter
        self._enabled = enabled
        self._allowed_scopes: frozenset[str] | None = frozenset(allowed_scopes) if allowed_scopes else None
        self._retention_days = retention_days
        self._max_chars = max_chars
        self._sweep_task: asyncio.Task | None = None
        # In-flight fire-and-forget write tasks, bucketed by trace_id so
        # attach_memory_ids can await only *its own* operation's writes before the
        # post-operation UPDATE (otherwise the UPDATE could race ahead of the
        # INSERTs it patches — but it must not block on unrelated operations).
        self._pending: dict[str | None, set[asyncio.Task]] = {}

    def is_enabled(self, scope: str) -> bool:
        """Whether tracing is active for the given call scope."""
        if not self._enabled:
            return False
        if self._allowed_scopes is not None:
            return scope in self._allowed_scopes
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

        record = LLMRequestRecord(
            provider=provider,
            model=model,
            scope=scope,
            status="error" if error is not None else "success",
            started_at=started_at,
            ended_at=ended_at,
            bank_id=ctx.bank_id if ctx else None,
            operation=ctx.operation if ctx else None,
            # OTel-style hierarchy: all calls of one operation invocation share
            # the context's trace_id and point at its operation span; this call
            # gets its own span_id.
            trace_id=ctx.trace_id if ctx else None,
            span_id=str(uuid.uuid4()),
            parent_span_id=ctx.operation_span_id if ctx else None,
            input=messages,
            output=None if error is not None else response_content,
            error=f"{type(error).__name__}: {error}" if error is not None else None,
            input_tokens=input_tokens or None,
            output_tokens=output_tokens or None,
            cached_tokens=cached_tokens or None,
            total_tokens=(input_tokens + output_tokens) or None,
            llm_info=llm_info,
            metadata=metadata,
        )
        self._record_fire_and_forget(record)

    def _record_fire_and_forget(self, record: LLMRequestRecord) -> None:
        """Schedule a trace write as a background task."""
        try:
            task = asyncio.create_task(self._safe_write(record))
        except RuntimeError:
            # No running event loop (e.g. during shutdown)
            logger.debug("Cannot schedule llm trace write: no running event loop")
            return
        key = record.trace_id
        self._pending.setdefault(key, set()).add(task)
        task.add_done_callback(lambda t, k=key: self._discard_pending(k, t))

    def _discard_pending(self, key: str | None, task: asyncio.Task) -> None:
        bucket = self._pending.get(key)
        if bucket is not None:
            bucket.discard(task)
            if not bucket:
                self._pending.pop(key, None)

    async def _safe_write(self, record: LLMRequestRecord) -> None:
        """Write a trace row. Errors are logged, never raised."""
        pool = self._pool_getter()
        if pool is None:
            logger.debug("LLM trace skipped: pool not available")
            return
        try:
            schema = self._schema_getter()
            table = f"{schema}.llm_requests"
            async with acquire_with_retry(pool, max_retries=1) as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {table}
                        (id, bank_id, operation, scope, trace_id, span_id, parent_span_id,
                         provider, model, status,
                         started_at, ended_at, duration_ms,
                         input_tokens, output_tokens, cached_tokens, total_tokens,
                         input, output, error, llm_info, metadata)
                    VALUES
                        ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                         $11, $12, $13, $14, $15, $16, $17,
                         $18::jsonb, $19::jsonb, $20, $21::jsonb, $22::jsonb)
                    """,
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
                    record.total_tokens,
                    _safe_json(record.input, self._max_chars),
                    _safe_json(record.output, self._max_chars),
                    record.error,
                    _safe_json(record.llm_info, self._max_chars) or "{}",
                    _safe_json(record.metadata, self._max_chars) or "{}",
                )
        except Exception as e:
            logger.warning(f"LLM trace write failed for scope={record.scope}: {e}")

    async def _flush_pending(self, trace_id: str) -> None:
        """Await this trace's in-flight writes so its rows exist before an UPDATE."""
        pending = [t for t in self._pending.get(trace_id, ()) if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

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
        if not self._enabled or trace_ctx is None or not trace_ctx.trace_id:
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
        try:
            asyncio.create_task(self._attach_memory_ids(trace_ctx.bank_id, trace_ctx.trace_id, patch))
        except RuntimeError:
            logger.debug("Cannot schedule llm trace memory_id attach: no running event loop")

    async def _attach_memory_ids(self, bank_id: str | None, trace_id: str, patch: dict[str, Any]) -> None:
        """Background worker: flush this trace's writes, then patch its rows."""
        # The trace-row INSERTs are fire-and-forget; flush *this trace's* writes
        # so the UPDATE patches rows that already exist rather than racing ahead
        # of them (without blocking on unrelated operations' pending writes).
        await self._flush_pending(trace_id)
        pool = self._pool_getter()
        if pool is None:
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

    # ── retention sweep ───────────────────────────────────────────────────────

    def start_retention_sweep(self) -> None:
        """Start the periodic retention sweep if retention is configured."""
        if self._retention_days <= 0 or not self._enabled:
            return
        try:
            self._sweep_task = asyncio.create_task(self._sweep_loop())
        except RuntimeError:
            logger.debug("Cannot start llm trace retention sweep: no running event loop")

    async def stop_retention_sweep(self) -> None:
        """Stop the periodic retention sweep."""
        if self._sweep_task and not self._sweep_task.done():
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except asyncio.CancelledError:
                pass
            self._sweep_task = None

    async def _sweep_loop(self) -> None:
        while True:
            await self._run_sweep()
            await asyncio.sleep(_SWEEP_INTERVAL_SECONDS)

    async def _run_sweep(self) -> None:
        """Delete trace rows older than retention_days. Concurrent-safe."""
        pool = self._pool_getter()
        if pool is None:
            return
        try:
            schema = self._schema_getter()
            table = f"{schema}.llm_requests"
            async with acquire_with_retry(pool, max_retries=1) as conn:
                result = await conn.execute(
                    f"DELETE FROM {table} WHERE started_at < NOW() - INTERVAL '{self._retention_days} days'"
                )
                if result and result != "DELETE 0":
                    logger.info(f"LLM trace retention sweep: {result}")
        except Exception as e:
            logger.warning(f"LLM trace retention sweep failed: {e}")

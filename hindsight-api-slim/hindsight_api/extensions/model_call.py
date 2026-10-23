"""Model call extension: every model call the server makes, with what went in and came out, and
every operation run (a retain, recall, reflect, consolidation or mental-model refresh) with its wall
time and where it went.

Every LLM, embedding, rerank and decision-model call -- in the API and in the worker alike,
successes and failures -- is handed to the extension once it finishes, as a
:class:`ModelCall`: the tenant schema, bank, operation and trace it ran for, provider and
model, tokens, duration, and the request and response themselves. For keeping a per-call
record of model spend and content outside the ``llm_requests`` table: object storage, a
warehouse, an observability product.

It sees exactly the calls ``llm_requests`` tracing sees (it is fed by the same GenAI
recorders), whether or not that table is written.

``record_operation`` (optional) receives an :class:`OperationRun` when each operation ends. Every
:class:`ModelCall` made inside a run carries its ``operation_id``, so the two join. The same rules
apply: it runs on the operation's path, so buffer and return.

**``record`` runs on the call path, synchronously.** It must only buffer: no I/O, no
blocking, nothing slow. Send from a background task, and flush in ``on_shutdown``, which is
awaited when the server stops, after the workers and before the database pools close. An
exception it raises never fails the call; it is swallowed and logged at DEBUG only, so
catch and report your own errors.

Load it with::

    HINDSIGHT_API_MODEL_CALL_EXTENSION=mypackage.calls:MyRecorder
    HINDSIGHT_API_MODEL_CALL_SOME_OPTION=value   # -> config["some_option"]
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel

from hindsight_api.engine.response_models import LLMToolCall
from hindsight_api.extensions.base import Extension
from hindsight_api.schema_context import get_current_bank_id, get_current_schema

logger = logging.getLogger(__name__)


class ModelCall(BaseModel):
    """One finished model call."""

    #: Unique per call.
    id: str
    started_at: datetime
    ended_at: datetime
    duration_ms: int
    #: The tenant schema the call ran in.
    tenant: str
    bank_id: str | None
    #: ``retain``, ``consolidation``, ``reflect``, ... ``None`` outside an operation.
    operation: str | None
    #: Shared by every call of one operation run.
    trace_id: str | None
    #: The :class:`OperationRun` the call was made in; ``None`` outside one.
    operation_id: str | None = None
    #: What the call was for: ``retain_extract_facts``, ``embedding_query``, ``rerank``, ...
    scope: str
    #: ``chat``, ``embeddings``, ``rerank`` or ``decision``.
    call_type: str
    provider: str
    model: str
    #: The upstream that answered, when the provider reports one (OpenRouter's ``provider`` field).
    served_by: str | None
    success: bool
    #: ``ExceptionType: message`` when the call failed.
    error: str | None
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    thoughts_tokens: int
    #: Texts or documents sent, for embedding and rerank calls.
    input_count: int | None
    finish_reason: str | None
    #: The parameters the caller set (max tokens, temperature, response schema, ...).
    request_params: dict[str, Any]
    #: Operation and per-call metadata (e.g. ``document_id``).
    metadata: dict[str, Any]
    #: What was sent: the chat messages, or the embedding / rerank / decision request.
    input: Any
    #: What came back; ``None`` when the call failed.
    output: Any
    #: The tools the model asked to call (the reflect agent's turns); ``None`` when it asked none.
    tool_calls: list[LLMToolCall] | None


class OperationRun(BaseModel):
    """One finished operation run: its wall time, and where the time went."""

    #: Unique per run; every :class:`ModelCall` made in it carries it as ``operation_id``.
    id: str
    #: The run this one was opened in (a mental-model refresh's reflect pass), if any. A run's
    #: figures exclude its children's, so a parent and its children add up without overlap.
    parent_id: str | None
    #: ``retain``, ``recall``, ``reflect``, ``consolidation``, ``mental_model_refresh``, ...
    operation: str
    tenant: str
    bank_id: str | None
    #: The queued task the run executed in, when it ran in a worker.
    task_id: str | None
    #: When that task was queued (top-level runs of a task only): ``started_at - queued_at`` is
    #: the time it waited.
    queued_at: datetime | None
    started_at: datetime
    ended_at: datetime
    duration_ms: int
    success: bool
    #: ``ExceptionType: message`` when it failed.
    error: str | None
    #: Model calls made in the run, their summed time (concurrent calls overlap) and tokens.
    model_calls: int
    model_ms: int
    input_tokens: int
    output_tokens: int
    #: Calls into the memories store and their summed time, measured at its interface.
    store_calls: int
    store_ms: int
    #: The engine's own timed phases (``retain.extract``, ``recall.retrieve``, ...), in ms.
    phases_ms: dict[str, float]


class ModelCallExtension(Extension, ABC):
    """Receive every model call the server makes. ``record`` must only buffer (see module docs)."""

    @abstractmethod
    def record(self, call: ModelCall) -> None:
        """Take one finished call. Synchronous and on the call path: buffer it, return."""
        ...

    def record_operation(self, run: OperationRun) -> None:  # noqa: B027 - optional hook
        """Take one finished operation run. Synchronous: buffer it, return. Optional."""


# ── runtime plumbing (used by the engine, not by extensions) ────────────────────


class ModelCallRecorder:
    """The GenAI span recorder that turns ``record_llm_call(...)`` into a :class:`ModelCall`."""

    def __init__(self, extension: ModelCallExtension) -> None:
        self.extension = extension

    def record_llm_call(
        self,
        *,
        provider: str,
        model: str,
        scope: str,
        messages: Any = None,
        response_content: Any = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration: float = 0.0,
        finish_reason: str | None = None,
        error: BaseException | None = None,
        cached_tokens: int = 0,
        thoughts_tokens: int | None = None,
        served_by: str | None = None,
        input_count: int | None = None,
        structured_input: Any = None,
        tool_calls: list[dict[str, Any]] | None = None,
        **_extra: Any,
    ) -> None:
        # Lazy: the engine imports this module.
        from hindsight_api.engine.llm_trace import (
            current_call_metadata,
            current_request_context,
            current_trace_context,
            genai_operation_name,
        )
        from hindsight_api.operation_runs import add_model_call, current_operation_id

        ctx = current_trace_context()
        ended_at = datetime.now(timezone.utc)
        metadata = dict(ctx.metadata) if ctx else {}
        metadata.update(current_call_metadata() or {})
        served_by = served_by or getattr(error, "served_by", None)
        call = ModelCall(
            id=str(uuid.uuid4()),
            started_at=ended_at - timedelta(seconds=max(0.0, duration)),
            ended_at=ended_at,
            duration_ms=int(max(0.0, duration) * 1000),
            tenant=get_current_schema(),
            bank_id=(ctx.bank_id if ctx else None) or get_current_bank_id(),
            operation=ctx.operation if ctx else None,
            trace_id=ctx.trace_id if ctx else None,
            operation_id=current_operation_id(),
            scope=scope,
            call_type=genai_operation_name(scope),
            provider=provider,
            model=model or "",
            served_by=served_by if isinstance(served_by, str) else None,
            success=error is None,
            error=f"{type(error).__name__}: {error}" if error is not None else None,
            input_tokens=input_tokens or 0,
            cached_input_tokens=cached_tokens or 0,
            output_tokens=output_tokens or 0,
            thoughts_tokens=thoughts_tokens or 0,
            input_count=input_count,
            finish_reason=finish_reason,
            request_params=dict(current_request_context() or {}),
            metadata=metadata,
            # A copy: the reflect loop keeps appending to the same message list after this
            # call, and the extension may serialize it later, off the call path.
            input=structured_input if structured_input is not None else list(messages or []),
            output=None if error is not None else response_content,
            tool_calls=[LLMToolCall.model_validate(tc) for tc in tool_calls] if tool_calls else None,
        )
        add_model_call(duration, call.input_tokens, call.output_tokens + call.thoughts_tokens)
        self.extension.record(call)

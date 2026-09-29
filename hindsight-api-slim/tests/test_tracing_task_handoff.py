"""Trace-context hand-off from the API to the worker.

Async retain returns as soon as the operation row is queued; the extraction
itself runs in the worker process. Without a trace context in the payload the
API's span for the enqueue and the worker's ``hindsight.retain`` span are two
unrelated traces, and the API half contains nothing but the INSERT. These tests
assert the payload carries a W3C traceparent and that ``execute_task`` runs the
handler inside the trace it names.
"""

import types

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

import hindsight_api.tracing as tracing
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.tracing import (
    TASK_TRACE_CONTEXT_KEY,
    extract_task_trace_context,
    inject_task_trace_context,
)


@pytest.fixture
def tracing_on(monkeypatch):
    """Enable tracing with a real SDK provider, without an exporter."""
    provider = TracerProvider()
    monkeypatch.setattr(tracing, "_tracing_enabled", True)
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer(__name__))
    return provider.get_tracer(__name__)


def test_payload_gets_no_trace_key_when_tracing_is_disabled(monkeypatch):
    """A disabled pipeline must not grow a null passenger on every task."""
    monkeypatch.setattr(tracing, "_tracing_enabled", False)
    payload: dict = {}

    inject_task_trace_context(payload)

    assert payload == {}
    assert extract_task_trace_context({TASK_TRACE_CONTEXT_KEY: {"traceparent": "x"}}) is None


def test_traceparent_round_trips_through_the_payload(tracing_on):
    """The worker rebuilds the enqueueing request's trace from the payload."""
    payload: dict = {}
    with tracing_on.start_as_current_span("hindsight.retain") as span:
        enqueue_trace_id = span.get_span_context().trace_id
        inject_task_trace_context(payload)

    assert "traceparent" in payload[TASK_TRACE_CONTEXT_KEY]

    restored = extract_task_trace_context(payload)
    assert trace.get_current_span(restored).get_span_context().trace_id == enqueue_trace_id


def test_payload_without_a_traceparent_extracts_to_nothing(tracing_on):
    """Internally scheduled work (consolidation, refresh) has no caller trace."""
    assert extract_task_trace_context({"type": "consolidation"}) is None
    assert extract_task_trace_context({TASK_TRACE_CONTEXT_KEY: "not-a-carrier"}) is None


async def test_execute_task_runs_the_handler_in_the_enqueueing_trace(tracing_on):
    """The worker's spans nest under the request that queued the task."""
    payload: dict = {"type": "batch_retain", "bank_id": "bank-a"}
    with tracing_on.start_as_current_span("POST /memories") as span:
        enqueue_trace_id = span.get_span_context().trace_id
        inject_task_trace_context(payload)

    observed: list[int] = []

    async def _fake_execute(task_dict):
        with tracing_on.start_as_current_span("hindsight.retain") as worker_span:
            observed.append(worker_span.get_span_context().trace_id)

    stub = types.SimpleNamespace(_execute_task=_fake_execute)
    await MemoryEngine.execute_task(stub, payload)

    assert observed == [enqueue_trace_id]


async def test_execute_task_without_a_traceparent_starts_its_own_trace(tracing_on):
    """Internally scheduled tasks still trace, just as their own root."""
    observed: list[int] = []

    async def _fake_execute(task_dict):
        with tracing_on.start_as_current_span("hindsight.consolidation") as worker_span:
            observed.append(worker_span.get_span_context().trace_id)

    stub = types.SimpleNamespace(_execute_task=_fake_execute)
    await MemoryEngine.execute_task(stub, {"type": "consolidation", "bank_id": "bank-a"})

    assert observed and observed[0] != 0

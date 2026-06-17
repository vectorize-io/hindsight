"""Tests for per-bank LLM request tracing.

Capture flows through the OTel GenAI recorder: providers call
``record_llm_call`` on success, and the LLM wrapper forwards failures through
the same recorder. The DB tracer (``LLMTraceRecorder``) is registered as one of
those recorders. Covers serialization, record building, the wrapper
success/error paths, and the HTTP read API (list / stats / tokens).
"""

import asyncio
import json
from datetime import datetime, timezone

import httpx
import pytest
import pytest_asyncio
from pydantic import BaseModel

from hindsight_api import tracing
from hindsight_api.api import create_app
from hindsight_api.engine import llm_trace
from hindsight_api.engine.llm_trace import (
    LLMRequestRecord,
    LLMTraceContext,
    LLMTraceRecorder,
    _safe_json,
    current_trace_context,
    set_trace_context,
)
from hindsight_api.engine.llm_wrapper import LLMProvider

# ── serialization helpers ─────────────────────────────────────────────────────


def test_safe_json_none_returns_none():
    assert _safe_json(None, 1000) is None


def test_safe_json_handles_datetime_uuid_set_and_pydantic():
    import uuid

    class Item(BaseModel):
        name: str

    data = {
        "when": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "id": uuid.uuid4(),
        "tags": {"a", "b"},
        "model": Item(name="x"),
        "raw": b"bytes",
    }
    out = json.loads(_safe_json(data, 100_000))
    assert out["when"].startswith("2026-01-01")
    assert isinstance(out["id"], str)
    assert sorted(out["tags"]) == ["a", "b"]
    assert out["model"] == {"name": "x"}
    assert out["raw"] == "<bytes>"


def test_safe_json_truncates_oversized_payload_to_valid_json():
    big = {"text": "x" * 5000}
    parsed = json.loads(_safe_json(big, max_chars=200))  # must stay valid JSON
    assert parsed["_truncated"] is True
    assert parsed["_original_chars"] > 200
    assert len(parsed["preview"]) == 200


def test_context_var_is_unset_by_default():
    assert current_trace_context() is None


# ── recorder: enable / allowlist ──────────────────────────────────────────────


def _recorder(enabled=True, allowed=None):
    return LLMTraceRecorder(
        pool_getter=lambda: None,
        schema_getter=lambda: "public",
        enabled=enabled,
        allowed_scopes=allowed or [],
    )


def test_recorder_disabled_is_not_enabled_for_any_scope():
    assert _recorder(enabled=False).is_enabled("memory") is False


def test_recorder_scope_allowlist():
    r = _recorder(enabled=True, allowed=["reflect"])
    assert r.is_enabled("reflect") is True
    assert r.is_enabled("retain_extract_facts") is False


# ── recorder: record_llm_call builds correct records ──────────────────────────


class _CapturingRecorder(LLMTraceRecorder):
    """Captures records synchronously instead of writing to the DB."""

    def __init__(self, enabled=True, allowed=None):
        super().__init__(
            pool_getter=lambda: None,
            schema_getter=lambda: "public",
            enabled=enabled,
            allowed_scopes=allowed or [],
        )
        self.records: list[LLMRequestRecord] = []

    def _record_fire_and_forget(self, record: LLMRequestRecord) -> None:
        self.records.append(record)


def test_record_llm_call_success_with_context_and_tokens():
    rec = _CapturingRecorder()
    token = set_trace_context(LLMTraceContext(bank_id="bank-x", operation="retain", metadata={"k": "v"}))
    try:
        rec.record_llm_call(
            provider="gemini",
            model="gemini-2.5-flash-lite",
            scope="retain_extract_facts",
            messages=[{"role": "user", "content": "hi"}],
            response_content="some output",
            input_tokens=100,
            output_tokens=20,
            duration=0.5,
            finish_reason="stop",
            cached_tokens=40,
        )
    finally:
        llm_trace.reset_trace_context(token)

    assert len(rec.records) == 1
    r = rec.records[0]
    assert r.status == "success"
    assert r.bank_id == "bank-x"
    assert r.operation == "retain"
    assert r.metadata == {"k": "v"}
    assert r.input_tokens == 100
    assert r.output_tokens == 20
    assert r.cached_tokens == 40
    assert r.total_tokens == 120
    assert r.output == "some output"
    assert r.llm_info["finish_reason"] == "stop"


def test_record_llm_call_error_record():
    rec = _CapturingRecorder()
    rec.record_llm_call(
        provider="mock",
        model="mock",
        scope="memory",
        messages=[],
        error=RuntimeError("boom"),
        duration=0.1,
    )
    assert len(rec.records) == 1
    r = rec.records[0]
    assert r.status == "error"
    assert r.error == "RuntimeError: boom"
    assert r.output is None
    assert r.bank_id is None


def test_record_llm_call_disabled_scope_records_nothing():
    rec = _CapturingRecorder(allowed=["reflect"])
    rec.record_llm_call(provider="mock", model="mock", scope="memory", messages=[], duration=0.1)
    assert rec.records == []


# ── wrapper: success via provider, error forwarded ────────────────────────────


@pytest.fixture
def registered_recorder():
    """Register a capturing recorder with the GenAI registry for one test."""
    rec = _CapturingRecorder()
    tracing.register_span_recorder(rec)
    yield rec
    tracing.unregister_span_recorder(rec)


@pytest.mark.asyncio
async def test_wrapper_success_recorded_by_provider(registered_recorder):
    llm = LLMProvider(provider="mock", api_key="", base_url="", model="mock")
    result = await llm.call(messages=[{"role": "user", "content": "hello"}], scope="memory")

    assert result == "mock response"
    assert len(registered_recorder.records) == 1
    r = registered_recorder.records[0]
    assert r.status == "success"
    assert r.provider == "mock"
    assert r.scope == "memory"


@pytest.mark.asyncio
async def test_wrapper_error_forwarded_and_reraised(registered_recorder):
    llm = LLMProvider(provider="mock", api_key="", base_url="", model="mock")
    llm._provider_impl.set_mock_exception(RuntimeError("kaboom"))

    with pytest.raises(RuntimeError, match="kaboom"):
        await llm.call(messages=[{"role": "user", "content": "x"}], scope="memory")

    assert len(registered_recorder.records) == 1
    r = registered_recorder.records[0]
    assert r.status == "error"
    assert "kaboom" in r.error


@pytest.mark.asyncio
async def test_configured_provider_binds_bank_context(registered_recorder):
    llm = LLMProvider(provider="mock", api_key="", base_url="", model="mock")

    class _Cfg:
        llm_gemini_safety_settings = None

    configured = llm.with_config(_Cfg(), bank_id="bank-42", operation="reflect")
    await configured.call(messages=[{"role": "user", "content": "x"}], scope="memory")

    assert len(registered_recorder.records) == 1
    assert registered_recorder.records[0].bank_id == "bank-42"
    assert registered_recorder.records[0].operation == "reflect"
    assert current_trace_context() is None  # unwound after the call


@pytest.mark.asyncio
async def test_engine_teardown_unregisters_recorder_even_when_close_skipped():
    """Regression for #2229.

    Span recorders live in a process-global registry, and providers fan every call
    out to ALL registered recorders. The engine fixtures must remove their recorder
    on teardown even when ``close()`` is skipped (pool already closing/absent) or
    raises before the unregister step — otherwise a leaked, still-enabled recorder
    from an earlier test records a later test's LLM calls into the shared DB, which
    is what made ``test_disabled_writes_no_rows`` flaky. The teardown helper must
    leave the registry exactly as it found it.
    """
    from hindsight_api import tracing

    from tests.conftest import _teardown_memory_engine

    sentinel = object()
    tracing.register_span_recorder(sentinel)
    try:
        assert sentinel in tracing.get_span_recorder()._recorders

        # _pool=None makes the helper's gated close() a no-op, exercising the exact
        # leak path; the finally must still unregister the recorder.
        class _FakeEngine:
            _pool = None
            _llm_recorder = sentinel

        await _teardown_memory_engine(_FakeEngine())
        assert sentinel not in tracing.get_span_recorder()._recorders
    finally:
        # Belt-and-suspenders: don't leave the sentinel in the global registry if an
        # assertion above fails (idempotent — the helper normally already removed it).
        tracing.unregister_span_recorder(sentinel)


# ── HTTP read API (integration) ───────────────────────────────────────────────


@pytest_asyncio.fixture
async def trace_api_client(memory):
    """Test client with LLM tracing enabled on the engine's recorder."""
    memory._llm_recorder._enabled = True
    memory._llm_recorder._allowed_scopes = None  # All scopes

    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    memory._llm_recorder._enabled = False


@pytest.fixture
def bank_id():
    return f"llm_trace_test_{datetime.now().timestamp()}"


@pytest.mark.asyncio
async def test_list_empty(trace_api_client, bank_id):
    await trace_api_client.put(f"/v1/default/banks/{bank_id}", json={"name": "Trace Bank"})
    response = await trace_api_client.get(f"/v1/default/banks/{bank_id}/llm-requests")
    assert response.status_code == 200
    data = response.json()
    assert data["items"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_retain_creates_trace_rows_with_tokens(trace_api_client, bank_id):
    await trace_api_client.put(f"/v1/default/banks/{bank_id}", json={"name": "Trace Bank"})
    response = await trace_api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": "Alice likes cats", "context": "preferences"}]},
    )
    assert response.status_code == 200
    # retain triggers fact extraction + consolidation (2 LLM calls → 2 trace
    # writes); give the fire-and-forget tasks room under parallel test load.
    await asyncio.sleep(1.5)

    response = await trace_api_client.get(f"/v1/default/banks/{bank_id}/llm-requests")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1

    operations = {item["operation"] for item in data["items"]}
    assert "retain" in operations, f"Expected a 'retain' trace, got: {operations}"

    retain_entry = next(it for it in data["items"] if it["operation"] == "retain")
    assert retain_entry["status"] == "success"
    assert retain_entry["provider"] == "mock"
    assert retain_entry["bank_id"] == bank_id
    # The mock provider reports token usage via record_llm_call.
    assert retain_entry["input_tokens"] is not None
    assert retain_entry["total_tokens"] is not None
    # Requested params are captured (retain sets retain_max_completion_tokens=64000).
    assert retain_entry["llm_info"]["request"]["max_completion_tokens"] == 64000

    # The retain extraction call is attributed to its document, and the
    # document_id filter returns that run's calls.
    doc_id = retain_entry["metadata"]["document_id"]
    assert doc_id
    by_doc = (
        await trace_api_client.get(f"/v1/default/banks/{bank_id}/llm-requests", params={"document_id": doc_id})
    ).json()
    assert by_doc["total"] >= 1
    assert all(it["metadata"].get("document_id") == doc_id for it in by_doc["items"])

    entry = data["items"][0]

    # OTel-style grouping: every row carries trace ids, and each operation
    # invocation (retain vs consolidation) gets its own trace_id, while a call's
    # parent_span_id is its operation span.
    by_op = {item["operation"]: item for item in data["items"]}
    for item in data["items"]:
        assert item["trace_id"] and item["span_id"] and item["parent_span_id"]
    if "retain" in by_op and "consolidation" in by_op:
        assert by_op["retain"]["trace_id"] != by_op["consolidation"]["trace_id"]

    # Filtering by a trace_id returns only that operation run's calls.
    a_trace = entry["trace_id"]
    resp = await trace_api_client.get(f"/v1/default/banks/{bank_id}/llm-requests", params={"trace_id": a_trace})
    assert resp.status_code == 200
    filtered = resp.json()
    assert filtered["total"] >= 1
    assert all(it["trace_id"] == a_trace for it in filtered["items"])

    # group=true paginates by run: total counts distinct runs (here retain +
    # consolidation = 2), and every traced row is still returned.
    resp = await trace_api_client.get(f"/v1/default/banks/{bank_id}/llm-requests", params={"group": "true"})
    assert resp.status_code == 200
    grouped = resp.json()
    distinct_traces = {it["trace_id"] for it in data["items"]}
    assert grouped["total"] == len(distinct_traces)
    assert len(grouped["items"]) >= len(data["items"])


@pytest.mark.asyncio
async def test_delta_reretain_binds_document_id(trace_api_client, bank_id):
    """A delta re-retain (editing/appending a document) must tag its trace with
    the document_id, so the document accrues one trace per retain — not just the
    initial full retain. Regression: the delta path used a second extraction call
    site that bypassed the document_id attribution, so edits were orphaned.
    """
    await trace_api_client.put(f"/v1/default/banks/{bank_id}", json={"name": "Trace Bank"})
    document_id = "delta-doc-001"

    # v1: a single self-contained chunk.
    v1 = "Alice is a software engineer at Google. She works on search infrastructure."
    resp = await trace_api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": v1, "context": "people", "document_id": document_id}]},
    )
    assert resp.status_code == 200
    await asyncio.sleep(1.5)

    # v2: original content preserved + a new paragraph. The unchanged first chunk
    # forces the delta path (it needs ≥1 unchanged chunk), and the new chunk is
    # extracted via the instrumented delta extraction call site.
    v2 = v1 + "\n\nBob joined Google as a product manager in 2024. He previously worked at Meta."
    resp = await trace_api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": v2, "context": "people", "document_id": document_id}]},
    )
    assert resp.status_code == 200
    await asyncio.sleep(1.5)

    # The document_id filter must now return both the full retain and the delta
    # re-retain — i.e. ≥2 distinct retain trace_ids, every row tagged.
    by_doc = (
        await trace_api_client.get(
            f"/v1/default/banks/{bank_id}/llm-requests",
            params={"document_id": document_id, "operation": "retain"},
        )
    ).json()
    assert all(it["metadata"].get("document_id") == document_id for it in by_doc["items"])
    retain_traces = {it["trace_id"] for it in by_doc["items"]}
    assert len(retain_traces) >= 2, f"expected full + delta retain bound to document, got {len(retain_traces)}"


@pytest.mark.asyncio
async def test_memory_ids_mapped_to_retain_and_consolidation(trace_api_client, bank_id):
    """A retain trace maps to the facts it created; a consolidation trace maps to
    the source memories it consumed (and any observations it produced). These are
    attached to every row of the trace after the operation completes.
    """
    await trace_api_client.put(f"/v1/default/banks/{bank_id}", json={"name": "Trace Bank"})
    resp = await trace_api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": "Alice works at Google as a senior engineer.", "context": "people"}]},
    )
    assert resp.status_code == 200
    # retain + the consolidation it triggers are fire-and-forget; give the trace
    # writes and the post-operation memory_id UPDATE room under parallel load.
    await asyncio.sleep(2.0)

    data = (await trace_api_client.get(f"/v1/default/banks/{bank_id}/llm-requests")).json()
    by_op = {item["operation"]: item for item in data["items"]}

    # Retain maps to the created fact unit_ids (outputs).
    retain = by_op["retain"]
    created = retain["metadata"].get("memory_ids")
    assert created, f"retain trace should map to created facts, metadata={retain['metadata']}"

    # Consolidation maps to the source memories consumed (inputs). The retained
    # fact is what gets consolidated, so it appears among the sources.
    if "consolidation" in by_op:
        sources = by_op["consolidation"]["metadata"].get("source_memory_ids")
        assert sources, "consolidation trace should map to the source memories it consumed"

    # Reverse lookup: ?memory_id=<created fact> returns every run touching it —
    # the retain that produced it (memory_ids) and any consolidation that consumed
    # it as a source (source_memory_ids).
    by_mem = (
        await trace_api_client.get(f"/v1/default/banks/{bank_id}/llm-requests", params={"memory_id": created[0]})
    ).json()
    assert by_mem["total"] >= 1
    for it in by_mem["items"]:
        meta = it["metadata"]
        assert created[0] in (meta.get("memory_ids") or []) or created[0] in (meta.get("source_memory_ids") or [])
    assert retain["trace_id"] in {it["trace_id"] for it in by_mem["items"]}, "producing retain trace must be returned"


@pytest.mark.asyncio
async def test_filter_by_status_and_operation(trace_api_client, bank_id):
    await trace_api_client.put(f"/v1/default/banks/{bank_id}", json={"name": "Trace Bank"})
    await trace_api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": "test content", "context": "test"}]},
    )
    await asyncio.sleep(1.0)

    response = await trace_api_client.get(
        f"/v1/default/banks/{bank_id}/llm-requests",
        params={"status": "success", "operation": "retain"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    for item in data["items"]:
        assert item["status"] == "success"
        assert item["operation"] == "retain"

    response = await trace_api_client.get(f"/v1/default/banks/{bank_id}/llm-requests", params={"status": "error"})
    assert response.json()["total"] == 0


@pytest.mark.asyncio
async def test_stats_endpoint_includes_tokens(trace_api_client, bank_id):
    await trace_api_client.put(f"/v1/default/banks/{bank_id}", json={"name": "Trace Bank"})
    await trace_api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": "stats content", "context": "test"}]},
    )
    await asyncio.sleep(1.0)

    response = await trace_api_client.get(f"/v1/default/banks/{bank_id}/llm-requests/stats", params={"period": "1d"})
    assert response.status_code == 200
    data = response.json()
    assert data["trunc"] == "day"
    assert len(data["buckets"]) >= 1
    bucket = data["buckets"][0]
    assert "statuses" in bucket
    assert "tokens" in bucket
    assert set(bucket["tokens"].keys()) == {"input", "output", "cached", "total"}
    assert bucket["total"] >= 1


@pytest.mark.asyncio
async def test_disabled_writes_no_rows(memory):
    memory._llm_recorder._enabled = False

    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        bid = f"llm_trace_disabled_{datetime.now().timestamp()}"
        await client.put(f"/v1/default/banks/{bid}", json={"name": "No Trace"})
        await client.post(
            f"/v1/default/banks/{bid}/memories",
            json={"items": [{"content": "nope", "context": "x"}]},
        )
        await asyncio.sleep(0.5)

        response = await client.get(f"/v1/default/banks/{bid}/llm-requests")
        assert response.status_code == 200
        assert response.json()["total"] == 0


# ── real-LLM acceptance (provider matrix) ─────────────────────────────────────


@pytest.mark.hs_llm_mat
@pytest.mark.asyncio
async def test_real_llm_retain_and_consolidation_traced(memory_real_llm):
    """A real retain produces traced retain *and* consolidation calls with real
    token usage. Runs across providers in the hs_llm_mat matrix to confirm the
    GenAI record_llm_call path reports tokens for each provider."""
    memory_real_llm._llm_recorder._enabled = True
    memory_real_llm._llm_recorder._allowed_scopes = None  # All scopes

    app = create_app(memory_real_llm, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=120.0) as client:
        bank_id = f"llm_trace_real_{datetime.now().timestamp()}"
        await client.put(f"/v1/default/banks/{bank_id}", json={"name": "Real Trace"})

        resp = await client.post(
            f"/v1/default/banks/{bank_id}/memories",
            json={
                "items": [
                    {
                        "content": "Alice is a data engineer from Turin who loves hiking in the Dolomites.",
                        "context": "profile",
                    }
                ]
            },
        )
        assert resp.status_code == 200

        # Real LLM latency + fire-and-forget writes: poll until both operations
        # land (retain fact-extraction, then consolidation over the new facts).
        by_op: dict[str, list[dict]] = {}
        for _ in range(30):
            await asyncio.sleep(1.0)
            data = (await client.get(f"/v1/default/banks/{bank_id}/llm-requests?limit=50")).json()
            by_op = {}
            for item in data["items"]:
                by_op.setdefault(item["operation"], []).append(item)
            if "retain" in by_op and "consolidation" in by_op:
                break

        assert "retain" in by_op, f"expected a retain trace, got operations: {list(by_op)}"
        assert "consolidation" in by_op, f"expected a consolidation trace, got operations: {list(by_op)}"

        for operation in ("retain", "consolidation"):
            entry = by_op[operation][0]
            assert entry["status"] == "success", f"{operation} trace not successful: {entry}"
            assert entry["provider"] == memory_real_llm._llm_config.provider
            assert entry["input_tokens"] and entry["input_tokens"] > 0, f"no input tokens for {operation}"
            assert entry["output_tokens"] and entry["output_tokens"] > 0, f"no output tokens for {operation}"
            assert entry["total_tokens"] == entry["input_tokens"] + entry["output_tokens"]
            assert entry["input"] is not None  # the prompt messages were captured

        # The operations map to the memory_units they touched: retain to the
        # facts it created, consolidation to the source memories it consumed.
        assert by_op["retain"][0]["metadata"].get("memory_ids"), "retain trace missing created memory_ids"
        assert by_op["consolidation"][0]["metadata"].get("source_memory_ids"), (
            "consolidation trace missing source_memory_ids"
        )

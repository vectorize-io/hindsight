"""ModelCallExtension: every model call, with its input and output, handed to an extension.

The extension is how a deployment keeps a per-call record of model spend and content
outside ``llm_requests``, so it must get each chat, embedding, rerank and decision call
exactly once, attributed to the tenant, bank, operation and trace it ran for.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from hindsight_api import operation_runs, tracing
from hindsight_api.config import clear_config_cache
from hindsight_api.engine.llm_trace import operation_run
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.extensions import ModelCall, ModelCallExtension
from hindsight_api.extensions.model_call import ModelCallRecorder, OperationRun


class _Collect(ModelCallExtension):
    def __init__(self, config: dict[str, str] | None = None) -> None:
        super().__init__(config or {})
        self.calls: list[ModelCall] = []
        self.runs: list[OperationRun] = []
        self.started = self.stopped = False

    def record(self, call: ModelCall) -> None:
        self.calls.append(call)

    def record_operation(self, run: OperationRun) -> None:
        self.runs.append(run)

    async def on_startup(self) -> None:
        self.started = True

    async def on_shutdown(self) -> None:
        self.stopped = True


@pytest.fixture
def ext(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_TRACE_ENABLED", "false")
    extension = _Collect()
    recorder = ModelCallRecorder(extension)
    tracing.register_span_recorder(recorder)
    operation_runs.set_sink(extension.record_operation)
    yield extension
    operation_runs.set_sink(None)
    tracing.unregister_span_recorder(recorder)
    clear_config_cache()


# ── what a record carries ─────────────────────────────────────────────────────


async def test_a_chat_call_with_its_input_output_and_attribution(ext):
    llm = LLMProvider(provider="mock", api_key="", base_url="", model="mock")
    messages = [{"role": "user", "content": "hello"}]
    with operation_run("bank-1", "consolidation"):
        await llm.call(messages=messages, scope="consolidation", temperature=0.2)
        await llm.call(messages=messages, scope="consolidation")

    first, second = ext.calls
    assert (first.tenant, first.bank_id, first.operation, first.scope, first.call_type, first.provider) == (
        "public",
        "bank-1",
        "consolidation",
        "consolidation",
        "chat",
        "mock",
    )
    assert first.success and first.error is None
    assert first.input == messages and first.output == "mock response"
    assert first.request_params.get("temperature") == 0.2
    assert first.trace_id and first.trace_id == second.trace_id, "one operation run, one trace"
    assert first.id != second.id and first.ended_at >= first.started_at


async def test_the_input_is_what_was_sent_not_what_the_list_became(ext):
    """The reflect loop keeps appending to its message list; a record must not grow with it."""
    llm = LLMProvider(provider="mock", api_key="", base_url="", model="mock")
    messages = [{"role": "user", "content": "first"}]
    await llm.call(messages=messages, scope="reflect")
    messages.append({"role": "assistant", "content": "later"})
    assert ext.calls[0].input == [{"role": "user", "content": "first"}]


async def test_a_tool_call_turn_records_the_tools_it_asked_for(ext):
    """The reflect agent's turns answer with tool calls, not text; those are the output."""
    llm = LLMProvider(provider="mock", api_key="", base_url="", model="mock")
    llm.set_mock_response([{"name": "recall", "arguments": {"query": "cats"}}])
    tools = [{"type": "function", "function": {"name": "recall", "parameters": {"type": "object"}}}]
    await llm.call_with_tools(messages=[{"role": "user", "content": "q"}], tools=tools, scope="reflect")

    [call] = ext.calls
    assert call.tool_calls is not None
    assert [(tc.name, tc.arguments) for tc in call.tool_calls] == [("recall", {"query": "cats"})]
    assert call.finish_reason == "tool_calls"


def test_a_failed_call(ext):
    tracing.get_span_recorder().record_llm_call(
        provider="openai",
        model="m",
        scope="rerank",
        messages=[],
        structured_input={"query": "q", "documents": ["a", "b"]},
        response_content={"scores": [1.0]},
        input_tokens=5,
        output_tokens=0,
        duration=0.25,
        error=TimeoutError("slow"),
        input_count=2,
    )
    [call] = ext.calls
    assert call.tool_calls is None
    assert (call.success, call.error, call.call_type, call.input_count, call.duration_ms) == (
        False,
        "TimeoutError: slow",
        "rerank",
        2,
        250,
    )
    assert call.input == {"query": "q", "documents": ["a", "b"]} and call.output is None


# ── the engine loads and runs it ──────────────────────────────────────────────


class _Loadable(_Collect):
    pass


async def test_the_engine_loads_registers_and_unregisters_it():
    from hindsight_api.engine.memory_engine import MemoryEngine

    embeddings = MagicMock()
    embeddings.dimension = 384
    env = {
        "HINDSIGHT_API_LLM_PROVIDER": "none",
        "HINDSIGHT_API_LLM_MODEL": "none",
        "HINDSIGHT_API_LLM_API_KEY": "test-key",
        "HINDSIGHT_API_MODEL_CALL_EXTENSION": f"{__name__}:_Loadable",
    }
    try:
        with patch.dict(os.environ, env, clear=False):
            clear_config_cache()
            engine = MemoryEngine(db_url="postgresql://localhost/hindsight_test", embeddings=embeddings)
    finally:
        clear_config_cache()
    extension = engine._model_call_extension
    try:
        assert isinstance(extension, _Loadable)
        assert engine._model_call_recorder in tracing.get_span_recorder()._recorders
        await engine.close()
        assert engine._model_call_recorder not in tracing.get_span_recorder()._recorders
        assert extension.stopped, "on_shutdown runs on close, so a buffer can be written out"
    finally:
        tracing.unregister_span_recorder(engine._model_call_recorder)
        tracing.unregister_span_recorder(engine._llm_recorder)


def test_no_extension_no_recorder():
    from hindsight_api.engine.memory_engine import MemoryEngine

    embeddings = MagicMock()
    embeddings.dimension = 384
    env = {"HINDSIGHT_API_LLM_PROVIDER": "none", "HINDSIGHT_API_LLM_MODEL": "none", "HINDSIGHT_API_LLM_API_KEY": "k"}
    try:
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("HINDSIGHT_API_MODEL_CALL_EXTENSION", None)
            clear_config_cache()
            engine = MemoryEngine(db_url="postgresql://localhost/hindsight_test", embeddings=embeddings)
    finally:
        clear_config_cache()
    tracing.unregister_span_recorder(engine._llm_recorder)
    assert engine._model_call_extension is None and engine._model_call_recorder is None


async def test_a_retain_and_a_recall_record_every_model_call_they_make(ext, memory):
    """End to end on the real engine: extraction, embeddings and the recall's rerank."""
    from datetime import datetime

    import httpx

    from hindsight_api.api import create_app

    bank_id = f"model_call_ext_{datetime.now().timestamp()}"
    app = create_app(memory, initialize_memory=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        await client.put(f"/v1/default/banks/{bank_id}", json={"name": "Calls Bank"})
        retain = await client.post(
            f"/v1/default/banks/{bank_id}/memories", json={"items": [{"content": "Alice likes cats"}]}
        )
        assert retain.status_code == 200
        recall = await client.post(
            f"/v1/default/banks/{bank_id}/memories/recall", json={"query": "What does Alice like?"}
        )
        assert recall.status_code == 200
        reflect = await client.post(
            f"/v1/default/banks/{bank_id}/reflect", json={"query": "What does Alice like?", "budget": "low"}
        )
        assert reflect.status_code == 200

    calls = [c for c in ext.calls if c.bank_id == bank_id]
    seen = {(c.operation, c.scope) for c in calls}
    assert {("retain", "embedding_document"), ("recall", "embedding_query"), ("recall", "rerank")} <= seen
    assert any(c.operation == "retain" and c.call_type == "chat" for c in calls)
    query = next(c for c in calls if c.scope == "embedding_query" and c.operation == "recall")
    assert query.input == {"texts": ["What does Alice like?"]}

    # Each operation is one run, and every call it made names it.
    runs = {r.id: r for r in ext.runs if r.bank_id == bank_id}
    by_op = {r.operation: r for r in runs.values()}
    assert {"retain", "recall"} <= set(by_op)
    for c in calls:
        assert c.operation_id in runs, (
            f"{c.scope}/{c.operation} call names no run: {[(x.scope, x.operation, x.operation_id is None) for x in calls]}"
        )
    recall_run = by_op["recall"]
    recall_calls = [c for c in calls if c.operation_id == recall_run.id]
    assert recall_run.model_calls == len(recall_calls) > 0
    assert recall_run.success and recall_run.duration_ms >= max(c.duration_ms for c in recall_calls)
    assert recall_run.store_calls > 0, "recall reads the memories store"
    assert any(k.startswith("recall.") for k in recall_run.phases_ms)
    retain_run = by_op["retain"]
    assert retain_run.input_tokens == sum(c.input_tokens for c in calls if c.operation_id == retain_run.id)
    assert retain_run.store_calls > 0 and any(k.startswith("retain.") for k in retain_run.phases_ms)
    # A reflect answered by the API, never queued, is a run all the same: its wall time and calls.
    reflect_run = by_op["reflect"]
    assert reflect_run.task_id is None and reflect_run.queued_at is None and reflect_run.success
    reflect_calls = [c for c in calls if c.operation_id == reflect_run.id]
    assert reflect_calls and all(c.operation == "reflect" for c in reflect_calls if c.call_type == "chat")
    assert reflect_run.model_calls == len(reflect_calls)
    assert reflect_run.model_ms == pytest.approx(sum(c.duration_ms for c in reflect_calls), abs=len(reflect_calls))


async def test_a_decision_call_with_its_request_and_answers(ext):
    from aiohttp import web

    from hindsight_api.engine.decision_model import DecisionModel, DecisionQuestion
    from hindsight_api.engine.remote_retry import RetryPolicy
    from tests.aiohttp_stub import stub_server

    async def handler(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        return web.json_response(
            {"answers": {k: {"noul": 0.9} for k in body["questions"]}, "usage": {"input_tokens": 11}}
        )

    async with stub_server(handler) as base_url:
        model = DecisionModel(
            provider="typesafe",
            api_key="k",
            model="d",
            base_url=base_url,
            timeout=5.0,
            max_concurrent=2,
            max_request_tokens=26_000,
            retry_policy=RetryPolicy(max_retries=0, initial_backoff=0.0, max_backoff=0.0, budget_seconds=5.0),
        )
        with operation_run("bank-d", "consolidation"):
            await model.ask("state", {"q": DecisionQuestion(type="noul", instructions="?")}, scope="decision_test")

    [call] = ext.calls
    assert (call.bank_id, call.scope, call.call_type, call.input_tokens) == ("bank-d", "decision_test", "decision", 11)
    assert call.input["state"] == "state" and call.output["answers"]["q"]["noul"] == 0.9

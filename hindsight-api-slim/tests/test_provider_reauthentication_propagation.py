"""Confirmed broken credentials stop the operation, not just one provider dispatch."""

import asyncio
import json
import uuid
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from hindsight_api.config import HindsightConfig, LLMStrategyConfig
from hindsight_api.engine.consolidation.consolidator import (
    _consolidate_batch_with_llm,
    _gather_or_cancel,
    run_consolidation_job,
)
from hindsight_api.engine.cross_encoder import RRFPassthroughCrossEncoder
from hindsight_api.engine.embeddings import Embeddings
from hindsight_api.engine.llm_interface import OutputTooLongError, ProviderReauthenticationRequiredError
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.memory_engine import MemoryEngine, _is_non_retryable_task_error
from hindsight_api.engine.multi_llm import MultiLLMProvider
from hindsight_api.engine.providers.mock_llm import MockLLM
from hindsight_api.engine.reflect.agent import run_reflect_agent
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult
from hindsight_api.engine.retain.fact_extraction import extract_facts_from_contents, extract_facts_from_text
from hindsight_api.engine.retain.types import RetainContent
from hindsight_api.models import RequestContext
from hindsight_api.worker.exceptions import RetryTaskAt


class _SyntheticEmbeddings(Embeddings):
    """Keep worker tests deterministic and independent of model downloads."""

    @property
    def provider_name(self) -> str:
        return "synthetic"

    @property
    def dimension(self) -> int:
        return 384

    async def initialize(self) -> None:
        pass

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 383 for _ in texts]


@pytest.fixture(scope="session")
def embeddings() -> Embeddings:
    return _SyntheticEmbeddings()


@pytest.fixture(scope="session")
def cross_encoder() -> RRFPassthroughCrossEncoder:
    return RRFPassthroughCrossEncoder()


def _mock_provider() -> MockLLM:
    return MockLLM(provider="mock", api_key="", base_url="", model="mock-model")


async def _reflect(llm: Any, **kwargs: Any) -> Any:
    return await run_reflect_agent(
        llm_config=llm,
        bank_id="terminal-auth-test",
        query="What is known?",
        bank_profile={},
        search_mental_models_fn=AsyncMock(),
        search_observations_fn=AsyncMock(),
        recall_fn=AsyncMock(),
        expand_fn=AsyncMock(),
        include_observations=False,
        include_recall=False,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_reflect_does_not_restart_round_robin_after_terminal_auth() -> None:
    primary = _mock_provider()
    secondary = _mock_provider()
    terminal = ProviderReauthenticationRequiredError("Reauthentication required")
    primary.set_mock_exception(terminal)
    secondary.set_mock_response("fallback")
    primary_wrapper = LLMProvider(provider="mock", api_key="", base_url="", model="mock-model")
    secondary_wrapper = LLMProvider(provider="mock", api_key="", base_url="", model="mock-model")
    primary_wrapper._provider_impl = primary
    secondary_wrapper._provider_impl = secondary
    router = MultiLLMProvider([primary_wrapper, secondary_wrapper], LLMStrategyConfig(mode="round-robin"))

    with pytest.raises(ProviderReauthenticationRequiredError):
        await _reflect(router)

    assert len(primary.get_mock_calls()) == 1
    assert secondary.get_mock_calls() == []


@pytest.mark.asyncio
async def test_reflect_does_not_hide_terminal_structured_output_failure() -> None:
    llm = _mock_provider()
    terminal = ProviderReauthenticationRequiredError("Reauthentication required")

    def respond(messages: list[dict[str, Any]], scope: str) -> str:
        if scope == "reflect_structured":
            raise terminal
        return "An answer ready for structured extraction."

    llm.set_response_callback(respond)
    with pytest.raises(ProviderReauthenticationRequiredError) as raised:
        await _reflect(
            llm,
            max_iterations=1,
            response_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
        )

    assert raised.value is terminal
    assert len(llm.get_mock_calls()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_sibling", [True, False], ids=["mixed", "generic-only"])
async def test_retain_split_children_preserve_terminal_priority(terminal_sibling: bool) -> None:
    llm = _mock_provider()
    terminal = ProviderReauthenticationRequiredError("Reauthentication required")

    def respond(messages: list[dict[str, Any]], scope: str) -> str:
        prompt = messages[-1]["content"]
        if "leftmarker" in prompt and "rightmarker" in prompt:
            raise OutputTooLongError("split this synthetic input")
        if "leftmarker" in prompt:
            raise RuntimeError("first-child-outage")
        raise terminal if terminal_sibling else RuntimeError("second-child-outage")

    llm.set_response_callback(respond)
    text = json.dumps(
        [{"role": "user", "content": "leftmarker " * 80}, {"role": "user", "content": "rightmarker " * 80}]
    )
    expected = ProviderReauthenticationRequiredError if terminal_sibling else RuntimeError
    with pytest.raises(expected) as raised:
        await extract_facts_from_text(
            text=text,
            event_date=None,
            llm_config=llm,
            config=replace(HindsightConfig.from_env(), retain_chunk_size=10000, retain_structured_chunk_size=10000),
        )

    if terminal_sibling:
        assert raised.value is terminal
    else:
        assert "first-child-outage" in str(raised.value)
        assert "second-child-outage" not in str(raised.value)
    assert len(llm.get_mock_calls()) == 3


@pytest.mark.asyncio
async def test_reflect_still_recovers_from_generic_provider_error() -> None:
    llm = _mock_provider()

    def respond(messages: list[dict[str, Any]], scope: str) -> str:
        if len(llm.get_mock_calls()) == 1:
            raise RuntimeError("ordinary transient error")
        return "Recovered answer."

    llm.set_response_callback(respond)
    result = await _reflect(llm, max_iterations=2)

    assert result.text == "Recovered answer."
    assert len(llm.get_mock_calls()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_sibling", [True, False], ids=["mixed", "generic-only"])
async def test_reflect_split_synthesis_preserves_terminal_priority(terminal_sibling: bool) -> None:
    llm = _mock_provider()
    terminal = ProviderReauthenticationRequiredError("Reauthentication required")
    first_error = RuntimeError("first-map-outage")

    def respond(messages: list[dict[str, Any]], scope: str) -> LLMToolCallResult:
        if scope == "reflect_tool_call":
            return LLMToolCallResult(tool_calls=[LLMToolCall(id="recall", name="recall", arguments={"query": "q"})])
        if len(llm.get_mock_calls()) == 2:
            raise first_error
        raise terminal if terminal_sibling else RuntimeError("later-map-outage")

    llm.set_response_callback(respond)
    evidence = {"memories": [{"id": f"mem-{i}", "text": "fact " + "z" * 600} for i in range(60)]}
    expected = ProviderReauthenticationRequiredError if terminal_sibling else RuntimeError
    with pytest.raises(expected) as raised:
        await run_reflect_agent(
            llm_config=llm,
            bank_id="terminal-map-test",
            query="What is known?",
            bank_profile={},
            max_iterations=2,
            max_context_tokens=1000,
            search_mental_models_fn=AsyncMock(),
            search_observations_fn=AsyncMock(),
            recall_fn=AsyncMock(return_value=evidence),
            expand_fn=AsyncMock(),
            include_observations=False,
        )

    assert raised.value is (terminal if terminal_sibling else first_error)
    assert len(llm.get_mock_calls()) >= 3


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [True, False], ids=["terminal", "generic"])
async def test_worker_does_not_retry_confirmed_auth_failure(
    memory: MemoryEngine, request_context: RequestContext, terminal: bool
) -> None:
    bank_id = f"terminal-worker-{uuid.uuid4().hex}"
    operation_id = uuid.uuid4()
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        # Seed a pending worker operation; outcome assertions use the public API.
        pool = await memory._get_pool()
        await pool.execute(
            "INSERT INTO async_operations (operation_id, bank_id, operation_type, status) VALUES ($1, $2, 'retain', 'pending')",
            operation_id,
            bank_id,
        )
        provider = memory._retain_llm_config._provider_impl
        error = (
            ProviderReauthenticationRequiredError("Reauthentication required") if terminal else RuntimeError("outage")
        )
        provider.set_mock_exception(error)
        task = {
            "type": "batch_retain",
            "operation_id": str(operation_id),
            "bank_id": bank_id,
            "contents": [{"content": "A synthetic fact."}],
        }
        if terminal:
            await memory.execute_task(task)
            result = await memory.get_operation_status(bank_id, str(operation_id), request_context=request_context)
            assert result["status"] == "failed"
            assert "reauthentication" in result["error_message"].lower()
        else:
            with pytest.raises(RetryTaskAt):
                await memory.execute_task(task)
        assert len(provider.get_mock_calls()) == 1
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


def test_task_error_classifier_distinguishes_terminal_auth_from_generic_failure() -> None:
    """Supplement the public worker test with the existing retry-decision seam."""
    assert _is_non_retryable_task_error(ProviderReauthenticationRequiredError("Reauthentication required"))
    assert not _is_non_retryable_task_error(RuntimeError("ordinary outage"))


@pytest.mark.asyncio
async def test_consolidation_batch_does_not_return_a_bisection_retry_after_terminal_auth() -> None:
    """Supplement the public consolidation test at its existing batch seam."""
    llm = _mock_provider()
    terminal = ProviderReauthenticationRequiredError("Reauthentication required")
    llm.set_mock_exception(terminal)

    with pytest.raises(ProviderReauthenticationRequiredError) as raised:
        await _consolidate_batch_with_llm(
            llm_config=llm,
            memories=[{"id": "m1", "text": "One fact."}, {"id": "m2", "text": "Another fact."}],
            union_observations=[],
            union_source_facts={},
            config=replace(HindsightConfig.from_env(), consolidation_max_attempts=2),
        )

    assert raised.value is terminal
    assert len(llm.get_mock_calls()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_sibling", [True, False], ids=["mixed", "generic-only"])
async def test_consolidation_cleanup_preserves_observed_terminal_failure(terminal_sibling: bool) -> None:
    first = RuntimeError("first failure")
    second = (
        ProviderReauthenticationRequiredError("Reauthentication required") if terminal_sibling else ValueError("next")
    )

    async def fail(error: Exception) -> None:
        raise error

    expected = ProviderReauthenticationRequiredError if terminal_sibling else RuntimeError
    with pytest.raises(expected) as raised:
        await _gather_or_cancel([fail(first), fail(second)])

    assert raised.value is (second if terminal_sibling else first)


@pytest.mark.asyncio
async def test_consolidation_cleanup_preserves_cancellation_over_terminal_failure() -> None:
    async def fail(error: BaseException) -> None:
        raise error

    with pytest.raises(asyncio.CancelledError):
        await _gather_or_cancel(
            [fail(asyncio.CancelledError()), fail(ProviderReauthenticationRequiredError("Reauthentication required"))]
        )


@pytest.mark.asyncio
async def test_consolidation_job_stops_without_bisection_or_marking_facts_failed(
    memory: MemoryEngine, request_context: RequestContext
) -> None:
    bank_id = f"terminal-consolidation-{uuid.uuid4().hex}"
    await memory.update_bank_config(bank_id, {"enable_observations": False}, request_context=request_context)
    try:
        # Retain through the public API before enabling consolidation, so setup
        # cannot trigger the failing provider or pre-consolidate the test facts.
        ids = await memory.retain_batch_async(
            bank_id,
            [{"content": "Alice likes tea."}, {"content": "Bob likes coffee."}],
            request_context=request_context,
        )
        expected_pending = sum(len(item) for item in ids)
        assert expected_pending >= 2
        await memory.update_bank_config(
            bank_id, {"enable_observations": True, "consolidation_llm_batch_size": 8}, request_context=request_context
        )
        provider = memory._consolidation_llm_config._provider_impl
        provider.set_mock_exception(ProviderReauthenticationRequiredError("Reauthentication required"))
        with pytest.raises(ProviderReauthenticationRequiredError):
            await run_consolidation_job(memory_engine=memory, bank_id=bank_id, request_context=request_context)

        assert len(provider.get_mock_calls()) == 1
        stats = await memory.get_bank_stats(bank_id, request_context=request_context, force_refresh=True)
        assert stats["failed_consolidation"] == 0
        assert stats["pending_consolidation"] == expected_pending
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_sibling", [True, False], ids=["mixed", "generic-only"])
async def test_streaming_retain_preserves_terminal_priority(
    memory: MemoryEngine, request_context: RequestContext, terminal_sibling: bool
) -> None:
    bank_id = f"terminal-streaming-{uuid.uuid4().hex}"
    await memory.update_bank_config(
        bank_id,
        {"retain_chunk_size": 100, "retain_structured_chunk_size": 100, "retain_chunk_batch_size": 1},
        request_context=request_context,
    )
    provider = memory._retain_llm_config._provider_impl

    def respond(messages: list[dict[str, Any]], scope: str) -> str:
        if "rightmarker" in messages[-1]["content"] and terminal_sibling:
            raise ProviderReauthenticationRequiredError("Reauthentication required")
        raise RuntimeError("streaming-transient-error")

    provider.set_response_callback(respond)
    expected = ProviderReauthenticationRequiredError if terminal_sibling else RuntimeError
    try:
        with pytest.raises(expected) as raised:
            await memory.retain_async(
                bank_id,
                "leftmarker " * 100 + "\n\n" + "rightmarker " * 100,
                document_id="synthetic-document",
                request_context=request_context,
            )
        assert len(provider.get_mock_calls()) >= 2
        if not terminal_sibling:
            assert "streaming-transient-error" in str(raised.value)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_retain_preserves_terminal_chunk_error() -> None:
    llm = _mock_provider()
    terminal = ProviderReauthenticationRequiredError("Reauthentication required")
    llm.set_mock_exception(terminal)

    with pytest.raises(ProviderReauthenticationRequiredError) as raised:
        await extract_facts_from_text(
            text="A synthetic fact.", event_date=None, llm_config=llm, config=HindsightConfig.from_env()
        )

    assert raised.value is terminal
    assert len(llm.get_mock_calls()) == 1


@pytest.mark.asyncio
async def test_retain_terminal_document_outranks_earlier_transient_error() -> None:
    llm = _mock_provider()
    terminal = ProviderReauthenticationRequiredError("Reauthentication required")

    def respond(messages: list[dict[str, Any]], scope: str) -> str:
        if "terminal-document" in messages[-1]["content"]:
            raise terminal
        raise RuntimeError("ordinary transient error")

    llm.set_response_callback(respond)
    with pytest.raises(ProviderReauthenticationRequiredError) as raised:
        await extract_facts_from_contents(
            contents=[RetainContent(content="transient-document"), RetainContent(content="terminal-document")],
            llm_config=llm,
            config=HindsightConfig.from_env(),
        )

    assert raised.value is terminal
    assert len(llm.get_mock_calls()) == 2

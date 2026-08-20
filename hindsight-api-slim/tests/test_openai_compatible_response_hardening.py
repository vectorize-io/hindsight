import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.openai_compatible_llm import (
    OpenAICompatibleLLM,
    ProviderResponseError,
)
from hindsight_api.worker.stage import StageHolder, bind_holder, set_stage


class SimpleJsonResponse(BaseModel):
    ok: bool


def _llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openai",
        api_key="test-key",
        base_url="https://example.test/v1",
        model="gpt-4o-mini",
    )


def _response(*, content: str | None = '{"ok": true}', choices=None, error=None):
    response = SimpleNamespace(error=error, usage=None)
    if choices is not None:
        response.choices = choices
        return response

    choice = SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content=content, tool_calls=None, refusal=None),
    )
    response.choices = [choice]
    return response


@pytest.mark.asyncio
async def test_attempt_stage_is_published_only_after_permits_are_acquired():
    llm = _llm()
    holder = StageHolder()
    waiting_for_permit = asyncio.Event()
    release_permit = asyncio.Event()

    @asynccontextmanager
    async def blocked_attempt_context():
        waiting_for_permit.set()
        await release_permit.wait()
        yield

    async def create(**_kwargs):
        assert holder.stage == "llm.openai.memory.attempt=1/1"
        return _response(content="ok")

    llm._client.chat.completions.create = AsyncMock(side_effect=create)

    async def invoke():
        bind_holder(holder)
        set_stage("llm.openai.memory.queued")
        return await llm.call(
            messages=[{"role": "user", "content": "hello"}],
            max_retries=0,
            attempt_context=blocked_attempt_context,
        )

    task = asyncio.create_task(invoke())
    await asyncio.wait_for(waiting_for_permit.wait(), timeout=1)
    assert holder.stage == "llm.openai.memory.queued"
    release_permit.set()
    assert await task == "ok"


@pytest.mark.asyncio
async def test_json_object_call_adds_json_hint_to_user_message():
    llm = _llm()
    create = AsyncMock(return_value=_response())
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        result = await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=SimpleJsonResponse,
            max_retries=0,
        )

    assert result.ok is True
    sent_messages = create.call_args.kwargs["messages"]
    assert sent_messages[0]["content"].startswith("Return valid json only.")


@pytest.mark.asyncio
async def test_json_object_call_strips_gemma_thought_tags_before_parsing():
    llm = _llm()
    create = AsyncMock(
        return_value=_response(content='<thought>\nI should return a compact JSON object.\n</thought>\n{"ok": true}')
    )
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        result = await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=SimpleJsonResponse,
            max_retries=0,
        )

    assert result.ok is True


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["qwen/qwen3.6-35b-a3b", "openai/gpt-oss-120b"])
async def test_openrouter_verification_uses_larger_reasoning_safe_budget(model: str):
    llm = OpenAICompatibleLLM(
        provider="openrouter",
        api_key="test-key",
        base_url="",
        model=model,
    )
    create = AsyncMock(return_value=_response(content="ok"))
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await llm.verify_connection()

    sent = create.call_args.kwargs
    assert sent["model"] == model
    assert sent["messages"] == [{"role": "user", "content": "Say 'ok'"}]
    assert sent["max_tokens"] == 512
    assert "max_completion_tokens" not in sent


@pytest.mark.asyncio
async def test_verification_uses_larger_budget_for_other_compatible_gateways():
    llm = _llm()
    create = AsyncMock(return_value=_response(content="ok"))
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await llm.verify_connection()

    sent = create.call_args.kwargs
    assert sent["max_tokens"] == 512
    assert "max_completion_tokens" not in sent


@pytest.mark.asyncio
async def test_error_payload_with_no_choices_raises_clear_provider_error_without_retry():
    llm = _llm()
    create = AsyncMock(
        return_value=_response(
            choices=None,
            error={
                "message": "Response input messages must contain the word 'json'",
                "type": "invalid_request_error",
                "param": "input",
            },
        )
    )
    # Simulate SDK objects where the declared field exists but is null.
    create.return_value.choices = None
    llm._client.chat.completions.create = create

    with pytest.raises(ProviderResponseError, match="Provider returned error payload.*word 'json'"):
        await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=SimpleJsonResponse,
            max_retries=2,
        )

    assert create.await_count == 1


@pytest.mark.asyncio
async def test_missing_choices_are_retryable_provider_response_errors():
    llm = _llm()
    empty_response = _response(choices=[])
    valid_response = _response()
    create = AsyncMock(side_effect=[empty_response, valid_response])
    llm._client.chat.completions.create = create

    with (
        patch("hindsight_api.engine.providers.openai_compatible_llm.asyncio.sleep", new=AsyncMock()) as sleep_mock,
        patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
    ):
        result = await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=SimpleJsonResponse,
            max_retries=1,
            initial_backoff=0,
        )

    assert result.ok is True
    assert create.await_count == 2
    sleep_mock.assert_awaited_once()


def _ollama_llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="ollama",
        api_key="",
        base_url="http://localhost:11434/v1",
        model="qwen3",
    )


def _ollama_response(content: str) -> httpx.Response:
    body = {
        "model": "qwen3",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    return httpx.Response(200, json=body, request=request)


@pytest.mark.asyncio
async def test_repairable_json_is_recovered_instead_of_being_dropped():
    """A malformed but structurally repairable response must not be lost (#3683).

    The trailing comma is what json_repair exists to fix. Before #3683 this
    provider parsed with bare json.loads and raised once the retries ran out,
    so the facts in the response never reached the caller.
    """
    llm = _llm()
    create = AsyncMock(return_value=_response(content='{"ok": true,}'))
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        result = await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=SimpleJsonResponse,
            max_retries=1,
            initial_backoff=0,
        )

    assert result.ok is True


@pytest.mark.asyncio
async def test_an_identical_parse_failure_stops_re_rolling_the_same_request():
    """A deterministic parse failure must not burn the whole retry budget (#3683).

    Every attempt re-sends a byte-identical request, so a failure that repeats at
    the same position will repeat forever. Two attempts are enough to establish
    that; the remaining budget is spent on repair instead of on more generations.
    """
    llm = _llm()
    create = AsyncMock(return_value=_response(content='{"ok": true,}'))
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        result = await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=SimpleJsonResponse,
            max_retries=3,
            initial_backoff=0,
        )

    assert result.ok is True
    assert create.await_count == 2


@pytest.mark.asyncio
async def test_a_parse_failure_at_a_new_position_still_earns_a_fresh_generation():
    """Flaky malformed output keeps the clean re-roll it had before (#3683).

    Only a repeat of the same failure is treated as deterministic. A different
    failure each time is the case a fresh generation can actually fix, so the
    retry ladder must still run.
    """
    llm = _llm()
    create = AsyncMock(
        side_effect=[
            _response(content='{"ok": true,}'),
            _response(content='{"ok": '),
            _response(content='{"ok": true}'),
        ]
    )
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        result = await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=SimpleJsonResponse,
            max_retries=3,
            initial_backoff=0,
        )

    assert result.ok is True
    assert create.await_count == 3


@pytest.mark.asyncio
async def test_ollama_native_repairs_malformed_json_instead_of_dropping_it():
    """The native /api/chat path parses the same way and had the same gap (#3683)."""
    llm = _ollama_llm()
    mock_client = AsyncMock()
    mock_client.post.return_value = _ollama_response('{"ok": true,}')
    mock_client.__aenter__.return_value = mock_client

    with (
        patch(
            "hindsight_api.engine.providers.openai_compatible_llm.httpx.AsyncClient",
            return_value=mock_client,
        ),
        patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
    ):
        result = await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=SimpleJsonResponse,
            max_retries=1,
            initial_backoff=0,
        )

    assert result.ok is True

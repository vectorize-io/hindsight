from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.openai_compatible_llm import (
    OpenAICompatibleLLM,
    ProviderResponseError,
)


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
@pytest.mark.parametrize(
    ("model", "expected_max_tokens"),
    [
        ("qwen/qwen3.6-35b-a3b", 512),
        ("QWEN/QWEN3-32B", 512),
        ("qwen/qwen2.5-72b-instruct", 100),
        ("qwen/qwen30-7b", 100),
        ("my-qwen3-proxy", 100),
        ("other/qwen3-proxy", 100),
        ("openai/gpt-oss-120b", 100),
    ],
)
async def test_openrouter_verification_budget_is_scoped_to_qwen3_reasoning_models(model: str, expected_max_tokens: int):
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
    assert sent["max_tokens"] == expected_max_tokens
    assert "max_completion_tokens" not in sent


@pytest.mark.asyncio
async def test_verification_keeps_default_budget_for_other_compatible_gateways():
    llm = _llm()
    create = AsyncMock(return_value=_response(content="ok"))
    llm._client.chat.completions.create = create

    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await llm.verify_connection()

    sent = create.call_args.kwargs
    assert sent["max_tokens"] == 100
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

from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM, ProviderResponseError


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

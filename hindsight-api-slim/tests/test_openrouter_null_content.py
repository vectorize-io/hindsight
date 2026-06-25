"""Regression tests for providers (e.g. OpenRouter) that return null content.

Some OpenRouter free-tier models (e.g. nvidia/nemotron-3-super-120b-a12b:free,
openai/gpt-oss-120b:free) occasionally respond with
``response.choices[0].message.content == None`` despite a valid finish_reason.
Without a guard, downstream string operations such as ``_strip_code_fences``
crash with ``TypeError: 'NoneType' object is not subscriptable``, and every
retry hits the same unhandled error so the entire retry budget is wasted.

See https://github.com/vectorize-io/hindsight/issues/1334.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM, ProviderResponseError


class _Response(BaseModel):
    answer: str


def _make_llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openrouter",
        api_key="sk-test",
        base_url="",
        model="nvidia/nemotron-3-super-120b-a12b:free",
    )


def _make_chat_response(content: str | None) -> MagicMock:
    """Build a mock that matches the shape expected by _first_choice_or_error.

    Key fields that must be explicitly set (not left as auto-MagicMock):
    - response.error = None          (otherwise truthy MagicMock triggers error path)
    - response.model_dump()          (returns dict without 'error' key)
    - choice.message.tool_calls/refusal  (otherwise truthy MagicMock in error msg)
    - usage.completion_tokens_details  (otherwise reasoning-token math crashes, #2378)
    """
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = content
    choice.message.tool_calls = None
    choice.message.refusal = None

    response = MagicMock()
    response.error = None
    response.model_dump.return_value = {}
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 0 if content is None else 5
    response.usage.total_tokens = 10 if content is None else 15
    response.usage.completion_tokens_details = None
    response.choices = [choice]
    return response


@pytest.mark.asyncio
async def test_null_content_raises_after_retries_exhausted():
    """All retries return null content -> ProviderResponseError, not TypeError."""
    llm = _make_llm()

    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _make_chat_response(None)

        with pytest.raises(ProviderResponseError, match="empty message content"):
            await llm.call(
                messages=[{"role": "user", "content": "extract facts"}],
                response_format=_Response,
                max_retries=2,
                initial_backoff=0.0,
                max_backoff=0.0,
            )

    # 3 attempts = max_retries (2) + 1 initial
    assert mock_create.call_count == 3


@pytest.mark.asyncio
async def test_null_content_recovers_on_retry():
    """Provider returns null on first call, valid JSON on second -> request succeeds."""
    llm = _make_llm()

    responses = [
        _make_chat_response(None),
        _make_chat_response('{"answer": "ok"}'),
    ]

    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = responses

        result = await llm.call(
            messages=[{"role": "user", "content": "extract facts"}],
            response_format=_Response,
            max_retries=2,
            initial_backoff=0.0,
            max_backoff=0.0,
        )

    assert isinstance(result, _Response)
    assert result.answer == "ok"
    assert mock_create.call_count == 2

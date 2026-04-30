"""Regression tests for providers (e.g. OpenRouter) that return null content.

Some OpenRouter free-tier models (e.g. nvidia/nemotron-3-super-120b-a12b:free,
openai/gpt-oss-120b:free) occasionally respond with
``response.choices[0].message.content == None`` despite a valid finish_reason.
Without a guard, downstream string operations such as ``_strip_code_fences``
crash with ``TypeError: 'NoneType' object is not subscriptable``, and every
retry hits the same unhandled error so the entire retry budget is wasted.

See https://github.com/vectorize-io/hindsight/issues/1334.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


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
    response = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 0 if content is None else 5
    response.usage.total_tokens = 10 if content is None else 15
    response.choices[0].finish_reason = "stop"
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = None
    return response


@pytest.mark.asyncio
async def test_null_content_raises_after_retries_exhausted():
    """All retries return null content -> JSONDecodeError, not TypeError."""
    llm = _make_llm()

    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _make_chat_response(None)

        with pytest.raises(json.JSONDecodeError):
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

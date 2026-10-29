"""GPT-6 on the `openai` (chat/completions) provider — issue #4891.

Verified against the live API with `gpt-6-luna`: `temperature` -> HTTP 400
"Unsupported value: 'temperature' does not support 0.3 with this model. Only the
default (1) value is supported." The model is a reasoning model, but the frozen
request-shape list only knew `gpt-5`/`o1`/`o3`, so the suppression that already
covers GPT-5 did not apply.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


def _make_llm(model: str = "gpt-6-luna", reasoning_effort: str | None = None) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openai",
        api_key="test",
        base_url="",
        model=model,
        reasoning_effort=reasoning_effort,
    )


def _plain_response() -> MagicMock:
    response = MagicMock()
    response.error = None
    response.model_dump.return_value = {}
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15
    response.usage.completion_tokens_details = None
    response.error = None
    response.model_dump.return_value = {}
    response.choices[0].finish_reason = "stop"
    response.choices[0].message.content = "ok"
    response.choices[0].message.tool_calls = None
    return response


@pytest.mark.asyncio
async def test_temperature_is_suppressed_for_gpt6():
    # Intention: GPT-6 rejects any temperature but the default, exactly like GPT-5.
    # Expected: the parameter is dropped, and the reasoning-model request shape
    # (max_completion_tokens, not max_tokens) is used.
    llm = _make_llm()
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = _plain_response()
        await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.3,
            max_completion_tokens=1000,
            max_retries=0,
        )
    params = create.await_args.kwargs
    assert "temperature" not in params
    assert "max_completion_tokens" in params

"""Prompt-length 400s follow the normal APIStatusError retry path."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIStatusError

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


def _llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="zai",
        model="glm-5-turbo",
        api_key="test",
        base_url="https://example.com/v1",
    )


def _length_error() -> APIStatusError:
    response = MagicMock()
    response.status_code = 400
    response.text = '{"code": "1261", "message": "Prompt exceeds max length"}'
    return APIStatusError(
        "bad",
        response=response,
        body={"code": "1261", "message": "Prompt exceeds max length"},
    )


@pytest.mark.asyncio
async def test_prompt_length_400_is_retried():
    llm = _llm()
    create = AsyncMock(side_effect=_length_error())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    with patch("hindsight_api.engine.providers.openai_compatible_llm.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(APIStatusError):
            await llm.call(
                messages=[{"role": "user", "content": "x"}],
                scope="mental_model_delta_ops",
                max_retries=2,
            )

    assert create.await_count == 3

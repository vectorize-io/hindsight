"""
Regression tests for Ollama native API think parameter handling.

The gpt-oss models require a string thinking level such as "low",
while other Ollama reasoning models continue to use think=False.
"""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


class _SampleOutput(BaseModel):
    summary: str


def _make_ollama_llm(model: str) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="ollama",
        api_key="",
        base_url="http://localhost:11434/v1",
        model=model,
    )


def _mock_ollama_response(content: dict) -> httpx.Response:
    body = {
        "model": "test-model",
        "message": {
            "role": "assistant",
            "content": json.dumps(content),
        },
        "done": True,
    }
    return httpx.Response(200, json=body)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected_think"),
    [
        ("gpt-oss:20b", "low"),
        ("qwen3.5:2b", False),
    ],
)
async def test_ollama_native_think_parameter(model, expected_think):
    """Use a thinking level for gpt-oss while preserving False for other models."""

    llm = _make_ollama_llm(model)

    mock_client = AsyncMock()
    mock_client.post.return_value = _mock_ollama_response({"summary": "test"})
    mock_client.__aenter__.return_value = mock_client

    with patch(
        "hindsight_api.engine.providers.openai_compatible_llm.httpx.AsyncClient",
        return_value=mock_client,
    ):
        await llm._call_ollama_native(
            messages=[{"role": "user", "content": "hello"}],
            response_format=_SampleOutput,
            max_completion_tokens=512,
            temperature=0.1,
            max_retries=0,
            initial_backoff=1.0,
            max_backoff=10.0,
            skip_validation=True,
        )

    request = mock_client.post.call_args

    assert request is not None

    payload = request.kwargs["json"]

    assert payload["think"] == expected_think
    assert "format" in payload
    assert payload["options"]["num_predict"] == 512
    assert payload["options"]["temperature"] == 0.1

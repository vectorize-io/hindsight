"""
Regression tests for Ollama native API think parameter handling.

The gpt-oss models require a string thinking level such as "low",
while other Ollama reasoning models such as qwen3.5 continue to
use think=False.
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
    captured_payloads: list[dict] = []

    async def _capture_post(url, *, json=None, **kwargs):
        captured_payloads.append(json)
        return _mock_ollama_response({"summary": "test"})

    with patch(
        "httpx.AsyncClient.post",
        new_callable=lambda: lambda: AsyncMock(side_effect=_capture_post),
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

    assert len(captured_payloads) == 1

    payload = captured_payloads[0]

    assert payload["think"] == expected_think

    assert "format" in payload
    assert payload["options"]["num_predict"] == 512
    assert payload["options"]["temperature"] == 0.1

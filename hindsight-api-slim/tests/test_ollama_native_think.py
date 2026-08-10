"""
Regression tests for Ollama native API extra_body handling.

The native /api/chat payload has two tiers: native top-level fields (``think``,
``keep_alive``, ...) and a nested ``options`` object (``seed``, ``top_p``,
``num_ctx``, ...). Configured ``extra_body`` must reach both, so operators can
enable thinking for gpt-oss models (``{"think": "low"}``) or tune generation
options without a code change (see #3246).
"""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


class _SampleOutput(BaseModel):
    summary: str


def _make_ollama_llm(model: str, extra_body: dict | None = None) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="ollama",
        api_key="",
        base_url="http://localhost:11434/v1",
        model=model,
        extra_body=extra_body,
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
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    return httpx.Response(200, json=body, request=request)


async def _capture_payload(llm: OpenAICompatibleLLM) -> dict:
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
    return request.kwargs["json"]


@pytest.mark.asyncio
async def test_ollama_native_think_defaults_false():
    """Thinking is disabled by default and structured-output format is included."""
    payload = await _capture_payload(_make_ollama_llm("qwen3.5:2b"))

    assert payload["think"] is False
    assert "format" in payload
    assert payload["options"]["num_predict"] == 512
    assert payload["options"]["temperature"] == 0.1


@pytest.mark.asyncio
async def test_ollama_native_think_override_via_extra_body():
    """extra_body top-level field overrides the think default (gpt-oss path)."""
    payload = await _capture_payload(_make_ollama_llm("gpt-oss:20b", extra_body={"think": "low"}))

    assert payload["think"] == "low"
    # Computed options are preserved alongside the top-level override.
    assert payload["options"]["num_predict"] == 512
    assert payload["options"]["temperature"] == 0.1


@pytest.mark.asyncio
async def test_ollama_native_options_merge_via_extra_body():
    """An extra_body "options" sub-dict merges into native generation options."""
    payload = await _capture_payload(
        _make_ollama_llm(
            "qwen3.5:2b",
            extra_body={"options": {"seed": 42, "temperature": 0.9}},
        )
    )

    # New option added, and a user value wins over the computed default.
    assert payload["options"]["seed"] == 42
    assert payload["options"]["temperature"] == 0.9
    assert payload["options"]["num_predict"] == 512
    # "options" is not leaked as a top-level payload field.
    assert "options" in payload
    assert payload["think"] is False

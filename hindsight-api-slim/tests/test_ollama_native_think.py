"""
Regression tests for Ollama native API extra_body handling.

The native /api/chat payload has two tiers: native top-level fields (``think``,
``keep_alive``, ...) and a nested ``options`` object (``seed``, ``top_p``,
``num_ctx``, ...). Configured ``extra_body`` must reach both, so operators can
enable thinking for gpt-oss models (``{"think": "low"}``) or tune generation
options without a code change (see #3246).
"""

import json

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM
from tests.ollama_stub import chat_body, ollama_stub


class _SampleOutput(BaseModel):
    summary: str


async def _capture_payload(model: str, extra_body: dict | None = None) -> dict:
    async with ollama_stub(chat_body(json.dumps({"summary": "test"}))) as stub:
        llm = OpenAICompatibleLLM(
            provider="ollama",
            api_key="",
            base_url=stub.openai_base_url,
            model=model,
            extra_body=extra_body,
        )
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
        await llm.cleanup()

    assert len(stub.requests) == 1
    return stub.requests[0].json


@pytest.mark.asyncio
async def test_ollama_native_think_defaults_false():
    """Thinking is disabled by default and structured-output format is included."""
    payload = await _capture_payload("qwen3.5:2b")

    assert payload["think"] is False
    assert "format" in payload
    assert payload["options"]["num_predict"] == 512
    assert payload["options"]["temperature"] == 0.1


@pytest.mark.asyncio
async def test_ollama_native_think_override_via_extra_body():
    """extra_body top-level field overrides the think default (gpt-oss path)."""
    payload = await _capture_payload("gpt-oss:20b", extra_body={"think": "low"})

    assert payload["think"] == "low"
    # Computed options are preserved alongside the top-level override.
    assert payload["options"]["num_predict"] == 512
    assert payload["options"]["temperature"] == 0.1


@pytest.mark.asyncio
async def test_ollama_native_options_merge_via_extra_body():
    """An extra_body "options" sub-dict merges into native generation options."""
    payload = await _capture_payload("qwen3.5:2b", extra_body={"options": {"seed": 42, "temperature": 0.9}})

    # New option added, and a user value wins over the computed default.
    assert payload["options"]["seed"] == 42
    assert payload["options"]["temperature"] == 0.9
    assert payload["options"]["num_predict"] == 512
    # "options" is not leaked as a top-level payload field.
    assert "options" in payload
    assert payload["think"] is False

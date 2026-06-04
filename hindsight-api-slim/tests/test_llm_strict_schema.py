"""
Tests for HINDSIGHT_API_LLM_STRICT_SCHEMA / config.llm_strict_schema.

When enabled, structured-output calls on the OpenAI-compatible provider use
strict ``response_format`` ``json_schema`` (grammar-enforced) instead of the
soft ``json_object`` path. Soft mode relies on the model *voluntarily* emitting
valid JSON; weaker self-hosted instruction-followers (small llama.cpp models)
return prose preambles, markdown ```json fences, or invalid JSON that fails
parsing and wedges retain/consolidation. Backends that implement json_schema
(OpenAI, llama.cpp, vLLM) can enforce it — this flag opts into that.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


class _Resp(BaseModel):
    ok: bool


def _llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openai", api_key="test-key", base_url="https://example.test/v1", model="gpt-4o-mini"
    )


def _response(content: str = '{"ok": true}'):
    choice = SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=content, tool_calls=None, refusal=None)
    )
    return SimpleNamespace(error=None, usage=None, choices=[choice])


async def _response_format_used(*, strict_config: bool):
    """Run a structured call and return the response_format dict sent upstream."""
    from hindsight_api.config import get_config

    real = get_config()  # captured before patching to avoid recursion

    class _Cfg:
        llm_strict_schema = strict_config

        def __getattr__(self, name):
            return getattr(real, name)

    llm = _llm()
    create = AsyncMock(return_value=_response())
    llm._client.chat.completions.create = create

    with (
        patch("hindsight_api.config.get_config", lambda: _Cfg()),
        patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"),
    ):
        await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=_Resp,
            max_retries=0,
        )
    return create.call_args.kwargs.get("response_format")


@pytest.mark.asyncio
async def test_strict_schema_enabled_uses_grammar_enforced_json_schema():
    rf = await _response_format_used(strict_config=True)
    assert rf is not None and rf.get("type") == "json_schema"
    assert rf["json_schema"]["strict"] is True


@pytest.mark.asyncio
async def test_strict_schema_disabled_keeps_soft_json_object_default():
    rf = await _response_format_used(strict_config=False)
    assert rf is not None and rf.get("type") == "json_object"

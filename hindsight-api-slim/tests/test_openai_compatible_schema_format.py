"""Tool calls reach the OpenAI-compatible providers as a structured-output ``call``.

The wrapper asks for a turn's call with ``tools_via_schema.schema_format(...)`` as the
response format (strict, unvalidated); the schema must go out as ``json_schema`` and the
reply come back as the parsed dict. No ``tools``/``tool_choice`` is ever sent.
"""

import json
import types
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_REQUIRED
from hindsight_api.engine.providers.llamacpp_llm import LlamaCppLLM
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM
from hindsight_api.engine.providers.tools_via_schema import response_schema, schema_format

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    }
]
_REPLY = {"call": {"name": "recall", "arguments": {"query": "x"}}}


def _format():
    return schema_format(response_schema(_TOOLS, LLM_TOOL_CHOICE_REQUIRED))


def _response(content: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                finish_reason="stop",
                message=types.SimpleNamespace(content=content, tool_calls=None, refusal=None),
            )
        ],
        usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="m",
    )


@pytest.mark.asyncio
async def test_openai_compatible_sends_the_schema_and_returns_the_dict():
    llm = OpenAICompatibleLLM(provider="openai", api_key="sk-test", base_url="", model="gpt-4o-mini")
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = _response(json.dumps(_REPLY))
        result = await llm.call(
            messages=[{"role": "user", "content": "go"}],
            response_format=_format(),
            strict_schema=True,
            skip_validation=True,
            max_retries=0,
        )
    params = create.call_args.kwargs
    assert params["response_format"]["type"] == "json_schema"
    assert params["response_format"]["json_schema"]["schema"] == _format().schema
    assert not {"tools", "tool_choice", "functions", "function_call"} & params.keys()
    assert result.content == _REPLY


@pytest.mark.asyncio
async def test_llamacpp_hands_the_schema_to_its_openai_compatible_server():
    provider = LlamaCppLLM(provider="llamacpp", api_key="", base_url="", model="test-model")
    provider._initialized = True
    provider._delegate = AsyncMock()
    fmt = _format()
    await provider.call(
        messages=[{"role": "user", "content": "go"}], response_format=fmt, strict_schema=True, skip_validation=True
    )
    forwarded = provider._delegate.call.await_args.kwargs
    assert forwarded["response_format"] is fmt
    assert forwarded["strict_schema"] is True
    assert forwarded["skip_validation"] is True

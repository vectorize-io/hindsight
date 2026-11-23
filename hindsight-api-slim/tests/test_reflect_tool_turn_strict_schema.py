"""HINDSIGHT_API_LLM_STRICT_SCHEMA_REFLECT reaches reflect's tool turns (#6673).

A model that runs away under the grammar (Qwen 3.5 behind an OpenAI-compatible API wrote
whitespace until a time limit) could be taken off it for every reflect call but the tool
turns, which sent ``strict_schema=True`` as a literal.
"""

from unittest.mock import AsyncMock

import pytest

from hindsight_api.config import (
    ENV_LLM_STRICT_SCHEMA,
    ENV_LLM_STRICT_SCHEMA_REFLECT,
    HindsightConfig,
)
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.response_models import LLMCallResult, TokenUsage

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "recall over the bank",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    }
]


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, True),  # tool turns were always held; the global default (false) does not reach them
        ({ENV_LLM_STRICT_SCHEMA: "false"}, True),
        ({ENV_LLM_STRICT_SCHEMA: "true", ENV_LLM_STRICT_SCHEMA_REFLECT: "false"}, False),
        ({ENV_LLM_STRICT_SCHEMA_REFLECT: "true"}, True),
    ],
)
def test_only_the_reflect_flag_takes_tool_turns_off_the_grammar(monkeypatch, env, expected):
    monkeypatch.delenv(ENV_LLM_STRICT_SCHEMA, raising=False)
    monkeypatch.delenv(ENV_LLM_STRICT_SCHEMA_REFLECT, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert HindsightConfig.from_env().llm_strict_schema_reflect_tool_turns is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [True, False])
async def test_tool_turn_sends_the_strict_flag_it_is_given(strict):
    llm = LLMProvider(provider="openai", api_key="k", base_url="https://example.test/v1", model="qwen3.5")
    call = AsyncMock(
        return_value=LLMCallResult(
            content={"call": {"name": "recall", "arguments": {"query": "q"}}},
            usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        )
    )
    llm._provider_impl.call = call  # type: ignore[method-assign]

    result = await llm.call_with_tools(messages=[{"role": "user", "content": "q"}], tools=TOOLS, strict_schema=strict)

    assert [c.name for c in result.tool_calls] == ["recall"]
    assert call.call_args.kwargs["strict_schema"] is strict

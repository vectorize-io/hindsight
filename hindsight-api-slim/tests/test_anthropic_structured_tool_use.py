"""Regression tests for issue #1002 — Anthropic structured output via forced tool_use.

When strict_schema=True, AnthropicLLM.call() must request the schema through a single
forced tool_use tool (tool_choice={"type":"tool",...}) and read the validated args from
the tool_use block, NOT inject the schema as text and json.loads() the reply (which caused
a ~1:1 invalid-JSON retry storm / OOM in production).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel


class _Decision(BaseModel):
    action: str
    reason: str


def _make_anthropic_provider():
    with patch("anthropic.AsyncAnthropic") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        from hindsight_api.engine.providers.anthropic_llm import AnthropicLLM

        provider = AnthropicLLM(
            provider="anthropic",
            api_key="fake-key",
            base_url="",
            model="claude-sonnet-4-20250514",
        )
    provider._client = MagicMock()
    return provider


def _tool_use_response(args: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.name = "structured_response"
    block.input = args
    resp = MagicMock()
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=5, output_tokens=2, cache_read_input_tokens=0)
    resp.stop_reason = "tool_use"
    return resp


@pytest.mark.asyncio
async def test_strict_schema_uses_forced_tool_choice():
    """strict_schema=True ⇒ a single tool is defined and tool_choice forces it (no schema text-injection)."""
    provider = _make_anthropic_provider()
    provider._client.messages.create = AsyncMock(return_value=_tool_use_response({"action": "skip", "reason": "dup"}))
    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        result = await provider.call(
            messages=[{"role": "user", "content": "decide"}],
            response_format=_Decision,
            strict_schema=True,
            scope="test",
            max_retries=0,
        )
    kwargs = provider._client.messages.create.call_args.kwargs
    # forced tool_use requested
    assert "tools" in kwargs and len(kwargs["tools"]) == 1
    assert kwargs["tool_choice"] == {"type": "tool", "name": "structured_response"}
    # schema NOT injected as text into the system prompt
    assert "valid JSON matching this schema" not in (kwargs.get("system") or "")
    # validated model returned straight from tool_use.input
    assert isinstance(result, _Decision)
    assert result.action == "skip"


@pytest.mark.asyncio
async def test_strict_schema_tool_use_never_hits_json_retry_loop():
    """A tool_use response is structurally valid → no second messages.create call (no retry storm)."""
    provider = _make_anthropic_provider()
    create = AsyncMock(return_value=_tool_use_response({"action": "keep", "reason": "novel"}))
    provider._client.messages.create = create
    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        await provider.call(
            messages=[{"role": "user", "content": "x"}],
            response_format=_Decision,
            strict_schema=True,
            scope="test",
            max_retries=10,  # would allow 11 attempts on the old text-parse path
        )
    assert create.await_count == 1  # exactly one call — the bug was N retries on malformed text


@pytest.mark.asyncio
async def test_non_strict_keeps_text_injection_fallback():
    """strict_schema=False (default) preserves the legacy schema-in-prompt behavior."""
    provider = _make_anthropic_provider()
    block = MagicMock()
    block.type = "text"
    block.text = '{"action":"skip","reason":"d"}'
    resp = MagicMock()
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=5, output_tokens=2, cache_read_input_tokens=0)
    resp.stop_reason = "end_turn"
    provider._client.messages.create = AsyncMock(return_value=resp)
    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        result = await provider.call(
            messages=[{"role": "user", "content": "decide"}],
            response_format=_Decision,
            strict_schema=False,
            scope="test",
            max_retries=0,
        )
    kwargs = provider._client.messages.create.call_args.kwargs
    assert "tools" not in kwargs  # no forced tool when not strict
    # system is a cache_control-marked block list; the schema text-injection
    # lands inside the (single) block.
    system_text = "".join(block["text"] for block in (kwargs.get("system") or []))
    assert "valid JSON matching this schema" in system_text
    assert isinstance(result, _Decision)

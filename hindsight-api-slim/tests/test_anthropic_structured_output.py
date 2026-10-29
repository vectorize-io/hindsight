"""Regression tests for issue #1002 — Anthropic structured output, natively constrained.

When strict_schema=True, AnthropicLLM.call() must constrain the reply with Anthropic's
native structured output (``output_config.format``), NOT inject the schema as text and
json.loads() a free reply (which caused a ~1:1 invalid-JSON retry storm / OOM in production).
It used to force a single ``tool_use`` tool; Hindsight sends no tools any more.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field


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


def _json_response(args: dict):
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(args)
    resp = MagicMock()
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=5, output_tokens=2, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    resp.stop_reason = "end_turn"
    return resp


@pytest.mark.asyncio
async def test_strict_schema_uses_native_structured_output():
    """strict_schema=True ⇒ the schema goes as output_config, no tools and no schema text-injection."""
    provider = _make_anthropic_provider()
    provider._client.messages.create = AsyncMock(return_value=_json_response({"action": "skip", "reason": "dup"}))
    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        result = (
            await provider.call(
                messages=[{"role": "user", "content": "decide"}],
                response_format=_Decision,
                strict_schema=True,
                scope="test",
                max_retries=0,
            )
        ).content
    kwargs = provider._client.messages.create.call_args.kwargs
    assert "tools" not in kwargs and "tool_choice" not in kwargs
    fmt = kwargs["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert set(fmt["schema"]["properties"]) == {"action", "reason"}
    assert fmt["schema"]["additionalProperties"] is False
    # schema NOT injected as text into the system prompt
    assert "valid JSON matching this schema" not in (kwargs.get("system") or "")
    assert isinstance(result, _Decision)
    assert result.action == "skip"


@pytest.mark.asyncio
async def test_strict_schema_never_hits_json_retry_loop():
    """A constrained reply is valid JSON → no second messages.create call (no retry storm)."""
    provider = _make_anthropic_provider()
    create = AsyncMock(return_value=_json_response({"action": "keep", "reason": "novel"}))
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
    resp.usage = MagicMock(input_tokens=5, output_tokens=2, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    resp.stop_reason = "end_turn"
    provider._client.messages.create = AsyncMock(return_value=resp)
    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        result = (
            await provider.call(
                messages=[{"role": "user", "content": "decide"}],
                response_format=_Decision,
                strict_schema=False,
                scope="test",
                max_retries=0,
            )
        ).content
    kwargs = provider._client.messages.create.call_args.kwargs
    assert "output_config" not in kwargs  # not constrained when not strict
    # system is a cache_control-marked block list; the schema text-injection
    # lands inside the (single) block.
    system_text = "".join(block["text"] for block in (kwargs.get("system") or []))
    assert "valid JSON matching this schema" in system_text
    assert isinstance(result, _Decision)


@pytest.mark.asyncio
async def test_keywords_native_output_refuses_are_dropped():
    """Anthropic 400s on maxItems / string lengths / minItems > 1 ("For 'array' type, property
    'maxItems' is not supported"); a property merely named like one survives."""

    class _Bounded(BaseModel):
        ids: list[str] = Field(max_length=3, min_length=2)
        maxItems: str = Field(max_length=10)  # noqa: N815 - a property named like the keyword

    provider = _make_anthropic_provider()
    provider._client.messages.create = AsyncMock(return_value=_json_response({"ids": ["a", "b"], "maxItems": "x"}))
    with patch("hindsight_api.engine.providers.anthropic_llm.get_metrics_collector"):
        await provider.call(
            messages=[{"role": "user", "content": "x"}], response_format=_Bounded, strict_schema=True, max_retries=0
        )
    schema = provider._client.messages.create.call_args.kwargs["output_config"]["format"]["schema"]
    assert schema["properties"]["ids"] == {"items": {"type": "string"}, "title": "Ids", "type": "array"}
    assert "maxLength" not in schema["properties"]["maxItems"]


def test_a_refused_schema_keyword_is_not_a_refused_output_config():
    """Only an endpoint without output_config switches to the prompt for good."""
    from hindsight_api.engine.providers.anthropic_llm import _rejects_output_format

    assert _rejects_output_format(Exception("output_config: Extra inputs are not permitted"))
    assert not _rejects_output_format(
        Exception("output_config.format.schema: For 'array' type, property 'maxItems' is not supported")
    )

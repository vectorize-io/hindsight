"""Regression tests for Codex reasoning-effort request serialization."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.providers.codex_llm import CodexLLM
from tests.codex_stream_stub import stub_codex_stream


def build_llm(reasoning_effort: str | None = "high") -> CodexLLM:
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        return CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.6-luna",
            reasoning_effort=reasoning_effort,
        )


@pytest.mark.asyncio
async def test_call_sends_reasoning_effort_separately_from_summary() -> None:
    llm = build_llm("high")
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None

    with (
        stub_codex_stream(llm, response) as mock_stream,
        patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock, return_value="ok"),
    ):
        await llm.call(messages=[{"role": "user", "content": "hello"}], max_retries=0)

    payload = mock_stream.call_args.kwargs["json"]
    # No summary and no encrypted reasoning: nothing reads them, and they cost quota.
    assert payload["reasoning"] == {"effort": "high"}
    assert "include" not in payload


@pytest.mark.asyncio
async def test_call_with_tools_sends_reasoning_effort_separately_from_summary() -> None:
    llm = build_llm("low")
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status.return_value = None

    with (
        stub_codex_stream(llm, response) as mock_stream,
        patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock, return_value=(None, [])),
    ):
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
            max_retries=0,
        )

    payload = mock_stream.call_args.kwargs["json"]
    assert payload["reasoning"] == {"effort": "low"}
    assert "include" not in payload


@pytest.mark.asyncio
async def test_unconfigured_reasoning_effort_is_omitted_from_the_payload() -> None:
    """Unset means the Codex backend picks the effort, not Hindsight.

    The config layer used to resolve unset to "low", so every Codex deployment sent an
    effort nobody had configured — the effort is the operator's to set (issue #3449).
    """
    llm = build_llm(None)
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None

    with (
        stub_codex_stream(llm, response) as mock_stream,
        patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock, return_value="ok"),
    ):
        await llm.call(messages=[{"role": "user", "content": "hello"}], max_retries=0)

    assert mock_stream.call_args.kwargs["json"]["reasoning"] == {}


@pytest.mark.asyncio
async def test_prompt_cache_key_is_stable_across_calls_with_the_same_prompt() -> None:
    """Every chunk of a retain shares the extraction prompt; a fresh key per call
    (the old uuid4) made each one a cache miss billed to the account's quota."""
    llm = build_llm(None)
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None
    system = {"role": "system", "content": "extract facts"}

    with (
        stub_codex_stream(llm, response) as mock_stream,
        patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock, return_value="ok"),
    ):
        await llm.call(messages=[system, {"role": "user", "content": "chunk one"}], max_retries=0)
        await llm.call(messages=[system, {"role": "user", "content": "chunk two"}], max_retries=0)
        await llm.call(messages=[{"role": "system", "content": "other prompt"}], max_retries=0)

    keys = [c.kwargs["json"]["prompt_cache_key"] for c in mock_stream.call_args_list]
    assert keys[0] == keys[1]
    assert keys[2] != keys[0]


@pytest.mark.asyncio
async def test_tool_loop_keeps_one_prompt_cache_key_as_the_conversation_grows() -> None:
    """Reflect's tool loop appends turns every iteration; the key must not change with them."""
    llm = build_llm(None)
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None
    system = {"role": "system", "content": "reflect"}

    with (
        stub_codex_stream(llm, response) as mock_stream,
        patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock, return_value=(None, [])),
    ):
        await llm.call_with_tools(messages=[system, {"role": "user", "content": "q"}], tools=[], max_retries=0)
        await llm.call_with_tools(
            messages=[system, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            tools=[],
            max_retries=0,
        )

    keys = [c.kwargs["json"]["prompt_cache_key"] for c in mock_stream.call_args_list]
    assert keys[0] == keys[1]

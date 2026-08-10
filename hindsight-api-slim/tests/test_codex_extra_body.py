"""Regression tests for Codex extra request-body parameters."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.llm_wrapper import create_llm_provider
from hindsight_api.engine.providers.codex_llm import CodexLLM


def build_llm(extra_body: dict | None = None) -> CodexLLM:
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        return CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.6-luna",
            extra_body=extra_body,
        )


def test_factory_forwards_extra_body() -> None:
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        llm = create_llm_provider(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.6-luna",
            reasoning_effort="low",
            extra_body={"service_tier": "priority"},
        )

    assert llm._extra_body == {"service_tier": "priority"}


@pytest.mark.asyncio
async def test_call_merges_extra_body_into_request() -> None:
    llm = build_llm({"service_tier": "priority"})
    response = MagicMock()
    response.raise_for_status.return_value = None

    with (
        patch.object(llm._client, "post", new_callable=AsyncMock) as mock_post,
        patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock, return_value="ok"),
    ):
        mock_post.return_value = response
        await llm.call(messages=[{"role": "user", "content": "hello"}], max_retries=0)

    assert mock_post.call_args.kwargs["json"]["service_tier"] == "priority"


@pytest.mark.asyncio
async def test_call_with_tools_merges_extra_body_into_request() -> None:
    llm = build_llm({"service_tier": "priority"})
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None

    with (
        patch.object(llm._client, "post", new_callable=AsyncMock) as mock_post,
        patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock, return_value=(None, [])),
    ):
        mock_post.return_value = response
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
            max_retries=0,
        )

    assert mock_post.call_args.kwargs["json"]["service_tier"] == "priority"

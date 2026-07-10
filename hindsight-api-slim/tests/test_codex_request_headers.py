"""Regression tests for Codex request identity headers."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hindsight_api.engine.providers.codex_llm import CodexLLM


def build_llm() -> CodexLLM:
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        return CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.6-luna",
        )


def assert_codex_request_identity(headers: httpx.Headers) -> None:
    assert headers["originator"] == "codex_cli_rs"
    assert headers["User-Agent"] == "codex_cli_rs/0.0.0 (Hindsight)"


@pytest.mark.asyncio
async def test_call_sends_codex_request_identity() -> None:
    llm = build_llm()
    response = MagicMock()
    response.raise_for_status.return_value = None

    with (
        patch.object(llm._client, "post", new_callable=AsyncMock) as mock_post,
        patch.object(llm, "_parse_sse_stream", new_callable=AsyncMock, return_value="ok"),
    ):
        mock_post.return_value = response
        await llm.call(messages=[{"role": "user", "content": "hello"}], max_retries=0)

    assert_codex_request_identity(mock_post.call_args.kwargs["headers"])


@pytest.mark.asyncio
async def test_call_with_tools_sends_codex_request_identity() -> None:
    llm = build_llm()
    response = MagicMock()
    response.status_code = 200
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

    assert_codex_request_identity(mock_post.call_args.kwargs["headers"])

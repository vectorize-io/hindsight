"""Codex usage-limit responses defer work until the reported reset."""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from hindsight_api.engine.llm_interface import ProviderRateLimitResetError
from hindsight_api.engine.providers.codex_llm import CodexLLM
from tests.codex_stream_stub import stub_codex_stream


def _build_llm() -> CodexLLM:
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        return CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-test",
        )


def _quota_response(resets_at: object, *, private_detail: str = "private diagnostic") -> httpx.Response:
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
    return httpx.Response(
        429,
        request=request,
        json={
            "error": {
                "type": "usage_limit_reached",
                "message": "The usage limit has been reached",
                "plan_type": private_detail,
                "resets_at": resets_at,
                "resets_in_seconds": 3600,
            }
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["call", "call_with_tools"])
async def test_usage_limit_with_valid_reset_defers_both_call_paths_without_retry(method: str) -> None:
    llm = _build_llm()
    retry_at = (datetime.now(UTC) + timedelta(hours=2)).replace(microsecond=0)
    response = _quota_response(int(retry_at.timestamp()))

    with (
        stub_codex_stream(llm, response) as stream,
        patch("hindsight_api.engine.providers.codex_llm.asyncio.sleep", new_callable=AsyncMock) as sleep,
        pytest.raises(ProviderRateLimitResetError) as exc_info,
    ):
        if method == "call":
            await llm.call(messages=[{"role": "user", "content": "x"}], max_retries=2)
        else:
            await llm.call_with_tools(messages=[{"role": "user", "content": "x"}], tools=[], max_retries=2)

    assert stream.call_count == 1
    sleep.assert_not_awaited()
    assert exc_info.value.retry_at == retry_at


@pytest.mark.asyncio
async def test_invalid_reset_timestamp_keeps_normal_call_retry_behavior() -> None:
    llm = _build_llm()
    response = _quota_response("not-an-epoch")

    with (
        stub_codex_stream(llm, response) as stream,
        patch("hindsight_api.engine.providers.codex_llm.asyncio.sleep", new_callable=AsyncMock) as sleep,
        pytest.raises(httpx.HTTPStatusError),
    ):
        await llm.call(messages=[{"role": "user", "content": "x"}], max_retries=2)

    assert stream.call_count == 3
    assert sleep.await_count == 2


@pytest.mark.asyncio
async def test_invalid_reset_timestamp_keeps_tool_call_http_error() -> None:
    llm = _build_llm()
    response = _quota_response(None)

    with stub_codex_stream(llm, response) as stream, pytest.raises(httpx.HTTPStatusError):
        await llm.call_with_tools(messages=[{"role": "user", "content": "x"}], tools=[])

    assert stream.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["call", "call_with_tools"])
async def test_quota_defer_message_and_logs_do_not_expose_response_body(method: str, caplog) -> None:
    llm = _build_llm()
    retry_at = datetime.now(UTC) + timedelta(hours=2)
    private_detail = "customer-private-plan-marker"
    response = _quota_response(int(retry_at.timestamp()), private_detail=private_detail)

    with stub_codex_stream(llm, response), pytest.raises(ProviderRateLimitResetError) as exc_info:
        if method == "call":
            await llm.call(messages=[{"role": "user", "content": "x"}], max_retries=2)
        else:
            await llm.call_with_tools(messages=[{"role": "user", "content": "x"}], tools=[], max_retries=2)

    assert private_detail not in str(exc_info.value)
    assert private_detail not in caplog.text
    assert json.dumps(response.json()) not in caplog.text

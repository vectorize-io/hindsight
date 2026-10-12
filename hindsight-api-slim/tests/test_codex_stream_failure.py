"""Regression tests for #5505: a failed Codex stream must raise, not return empty content.

The ChatGPT backend reports an overload after a 200 by streaming an ``error`` event and
then ``response.failed``. Both SSE parsers used to skip those events, so ``call()`` ran
``json.loads("")`` (a misleading ``JSONDecodeError``) and ``call_with_tools()`` returned
no tool calls (a misleading ``ReflectToolCallError`` upstream).
"""

from unittest.mock import patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.codex_llm import CodexLLM, CodexStreamError
from hindsight_api.engine.remote_retry import is_transient_remote_error
from tests.codex_stream_stub import CodexReply, stub_codex_stream

pytestmark = pytest.mark.asyncio

_FAILED_STREAM = (
    "event: response.created\n"
    'data: {"type": "response.created"}\n\n'
    "event: error\n"
    'data: {"type":"service_unavailable_error","code":"server_is_overloaded",'
    '"message":"Our servers are currently overloaded. Please try again later."}\n\n'
    "event: response.failed\n"
    'data: {"response":{"status":"failed","error":{"code":"server_is_overloaded",'
    '"message":"Our servers are currently overloaded. Please try again later."}}}\n\n'
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Recall memories",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]


class _Facts(BaseModel):
    facts: list[str]


def _build_llm() -> CodexLLM:
    with (
        patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")),
        patch.object(CodexLLM, "_load_codex_refresh_token", return_value=None),
    ):
        return CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.4-mini",
        )


async def test_structured_call_raises_backend_error_and_retries():
    llm = _build_llm()
    with stub_codex_stream(llm, CodexReply(text=_FAILED_STREAM)) as server:
        with pytest.raises(CodexStreamError, match="server_is_overloaded") as exc_info:
            await llm.call(
                messages=[{"role": "user", "content": "extract"}],
                response_format=_Facts,
                max_retries=1,
                initial_backoff=0,
                max_backoff=0,
            )
        # Treated like any transport failure: retried within the budget.
        assert server.call_count == 2
    assert is_transient_remote_error(exc_info.value)


async def test_tool_call_raises_backend_error():
    llm = _build_llm()
    with stub_codex_stream(llm, CodexReply(text=_FAILED_STREAM)):
        with pytest.raises(CodexStreamError, match="Our servers are currently overloaded"):
            await llm.call_with_tools(
                messages=[{"role": "user", "content": "recall"}],
                tools=_TOOLS,
                max_retries=0,
            )

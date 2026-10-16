"""Regression tests for #5541: Codex reports the token usage the backend streams.

The ``/codex/responses`` stream ends with ``response.completed``, whose
``response.usage`` is the Responses API usage block. Both SSE parsers used to drop
it, so ``call()`` returned a char/4 estimate and ``call_with_tools()`` returned 0.
"""

import json
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.codex_llm import CodexLLM
from tests.codex_stream_stub import CodexReply, stub_codex_stream

pytestmark = pytest.mark.asyncio

_USAGE = {
    "input_tokens": 1200,
    "input_tokens_details": {"cached_tokens": 1000},
    "output_tokens": 90,
    "output_tokens_details": {"reasoning_tokens": 60},
    "total_tokens": 1290,
}
_COMPLETED = (
    "event: response.completed\n"
    + "data: "
    + json.dumps({"type": "response.completed", "response": {"usage": _USAGE}})
    + "\n\n"
)
_TEXT_STREAM = "event: response.text.delta\n" + "data: " + json.dumps({"delta": "y" * 40}) + "\n\n" + _COMPLETED


def _tool_stream(name: str, arguments: dict) -> str:
    item = {"type": "function_call", "status": "completed", "name": name, "arguments": json.dumps(arguments)}
    return "event: response.output_item.done\n" + "data: " + json.dumps({"item": item}) + "\n\n" + _COMPLETED


class _Fact(BaseModel):
    fact: str


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


def _assert_backend_counts(input_tokens, output_tokens, cached_tokens, thoughts_tokens) -> None:
    # Visible output only: 90 output tokens minus the 60 reasoning tokens.
    assert (input_tokens, output_tokens, cached_tokens, thoughts_tokens) == (1200, 30, 1000, 60)


async def test_call_reports_backend_usage():
    llm = _build_llm()
    with stub_codex_stream(llm, CodexReply(text=_TEXT_STREAM)):
        result = await llm.call(messages=[{"role": "user", "content": "x" * 400}], max_retries=0)

    assert result.content == "y" * 40
    usage = result.usage
    _assert_backend_counts(usage.input_tokens, usage.output_tokens, usage.cached_tokens, usage.thoughts_tokens)
    assert usage.total_tokens == 1230


async def test_strict_schema_call_reports_backend_usage():
    llm = _build_llm()
    with stub_codex_stream(llm, CodexReply(text=_tool_stream("structured_response", {"fact": "ok"}))):
        result = await llm.call(
            messages=[{"role": "user", "content": "x"}],
            response_format=_Fact,
            strict_schema=True,
            max_retries=0,
        )

    assert result.content == _Fact(fact="ok")
    usage = result.usage
    _assert_backend_counts(usage.input_tokens, usage.output_tokens, usage.cached_tokens, usage.thoughts_tokens)


async def test_call_with_tools_reports_backend_usage():
    llm = _build_llm()
    with stub_codex_stream(llm, CodexReply(text=_tool_stream("recall", {"query": "q"}))):
        result = await llm.call_with_tools(messages=[{"role": "user", "content": "x"}], tools=[], max_retries=0)

    assert [tc.name for tc in result.tool_calls] == ["recall"]
    _assert_backend_counts(result.input_tokens, result.output_tokens, result.cached_tokens, result.thoughts_tokens)


async def test_call_without_usage_block_falls_back_to_estimate():
    stream = "event: response.text.delta\n" + "data: " + json.dumps({"delta": "y" * 40}) + "\n\n"
    llm = _build_llm()
    with stub_codex_stream(llm, CodexReply(text=stream)):
        result = await llm.call(messages=[{"role": "user", "content": "x" * 400}], max_retries=0)

    assert (result.usage.input_tokens, result.usage.output_tokens) == (100, 10)

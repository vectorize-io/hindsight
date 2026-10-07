"""Regression test for surfacing the CLI's real error text (issue #2702).

The Claude Code CLI can report a failure with ``is_error=True`` while
``subtype`` still reads ``"success"``, putting the actual detail in
``result`` — e.g. quota exhaustion:

    {"type":"result","subtype":"success","is_error":true,
     "api_error_status":429,
     "result":"You've hit your weekly limit · resets Jul 18, 12pm (UTC)"}

The Agent SDK's fallback exception is built from ``errors`` (empty here)
or ``subtype``, producing the misleading "Claude Code returned an error
result: success". These tests assert that both provider call paths inspect
the ResultMessage directly and raise with the CLI's actual error text.

A limit message that names its reset time, like the one above, is raised as
``ProviderRateLimitResetError`` so the worker parks the task until then (#5394);
any other error text stays a ``RuntimeError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from hindsight_api.engine.llm_interface import ProviderRateLimitResetError
from hindsight_api.engine.providers.claude_code_llm import _limit_reset_at

QUOTA_ERROR_TEXT = "You've hit your weekly limit · resets Jul 18, 12pm (UTC)"
OTHER_ERROR_TEXT = "API Error: 500 Internal server error"


class _StructuredResponse(BaseModel):
    fact: str


@dataclass
class _FakeOptions:
    """Stand-in for ClaudeAgentOptions; captures kwargs without importing SDK."""

    system_prompt: str | None = None
    max_turns: int | None = None
    allowed_tools: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    model: str | None = None


class _FakeAssistantMessage:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResultMessage:
    def __init__(self, subtype: str, is_error: bool, result: str | None) -> None:
        self.subtype = subtype
        self.is_error = is_error
        self.result = result


def _instantiate_provider():
    from hindsight_api.engine.providers.claude_code_llm import ClaudeCodeLLM

    return ClaudeCodeLLM(
        provider="claude-code",
        api_key="",
        base_url="",
        model="claude-haiku-4-5",
        reasoning_effort="low",
    )


@pytest.mark.asyncio
async def test_call_raises_with_result_text_on_error_result(monkeypatch):
    """call() must surface ResultMessage.result, not the 'success' subtype."""
    import claude_agent_sdk

    async def fake_query(prompt: str, options: _FakeOptions):
        yield _FakeResultMessage(subtype="success", is_error=True, result=OTHER_ERROR_TEXT)

    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(claude_agent_sdk, "AssistantMessage", _FakeAssistantMessage)
    monkeypatch.setattr(claude_agent_sdk, "TextBlock", _FakeTextBlock)
    monkeypatch.setattr(claude_agent_sdk, "ResultMessage", _FakeResultMessage)
    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    provider = _instantiate_provider()
    with pytest.raises(RuntimeError) as excinfo:
        await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=0,
            scope="test",
        )

    assert OTHER_ERROR_TEXT in str(excinfo.value)
    assert "error result: success" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_call_falls_back_to_subtype_when_result_empty(monkeypatch):
    """With no result text, the subtype is still better than nothing."""
    import claude_agent_sdk

    async def fake_query(prompt: str, options: _FakeOptions):
        yield _FakeResultMessage(subtype="error_max_turns", is_error=True, result=None)

    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(claude_agent_sdk, "AssistantMessage", _FakeAssistantMessage)
    monkeypatch.setattr(claude_agent_sdk, "TextBlock", _FakeTextBlock)
    monkeypatch.setattr(claude_agent_sdk, "ResultMessage", _FakeResultMessage)
    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    provider = _instantiate_provider()
    with pytest.raises(RuntimeError, match="error_max_turns"):
        await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=0,
            scope="test",
        )


@pytest.mark.asyncio
async def test_call_ignores_non_error_result_message(monkeypatch):
    """A normal is_error=False ResultMessage must not affect the response."""
    import claude_agent_sdk

    async def fake_query(prompt: str, options: _FakeOptions):
        yield _FakeAssistantMessage(content=[_FakeTextBlock(text="ok")])
        yield _FakeResultMessage(subtype="success", is_error=False, result="ok")

    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(claude_agent_sdk, "AssistantMessage", _FakeAssistantMessage)
    monkeypatch.setattr(claude_agent_sdk, "TextBlock", _FakeTextBlock)
    monkeypatch.setattr(claude_agent_sdk, "ResultMessage", _FakeResultMessage)
    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    provider = _instantiate_provider()
    result = (
        await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=0,
            scope="test",
        )
    ).content

    assert result == "ok"


@pytest.mark.asyncio
async def test_call_records_span_for_unvalidated_dict(monkeypatch):
    """skip_validation returns a dict, which must still be serialized into the span."""
    import claude_agent_sdk

    import hindsight_api.tracing as tracing

    async def fake_query(prompt: str, options: _FakeOptions):
        yield _FakeAssistantMessage(content=[_FakeTextBlock(text='{"fact": "x"}')])
        yield _FakeResultMessage(subtype="success", is_error=False, result='{"fact": "x"}')

    span_recorder = MagicMock()
    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(claude_agent_sdk, "AssistantMessage", _FakeAssistantMessage)
    monkeypatch.setattr(claude_agent_sdk, "TextBlock", _FakeTextBlock)
    monkeypatch.setattr(claude_agent_sdk, "ResultMessage", _FakeResultMessage)
    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    monkeypatch.setattr(tracing, "get_span_recorder", lambda: span_recorder)

    result = (
        await _instantiate_provider().call(
            messages=[{"role": "user", "content": "extract facts"}],
            response_format=_StructuredResponse,
            skip_validation=True,
            max_retries=0,
            scope="retain_extract_facts",
        )
    ).content

    assert result == {"fact": "x"}
    assert span_recorder.record_llm_call.call_args.kwargs["response_content"] == '{"fact": "x"}'


@pytest.mark.asyncio
async def test_call_survives_span_recorder_failure(monkeypatch):
    """A raising span recorder must be logged, never break the call (best-effort, #3025)."""
    import claude_agent_sdk

    import hindsight_api.tracing as tracing

    async def fake_query(prompt: str, options: _FakeOptions):
        yield _FakeAssistantMessage(content=[_FakeTextBlock(text="ok")])
        yield _FakeResultMessage(subtype="success", is_error=False, result="ok")

    span_recorder = MagicMock()
    span_recorder.record_llm_call.side_effect = RuntimeError("recorder exploded")
    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(claude_agent_sdk, "AssistantMessage", _FakeAssistantMessage)
    monkeypatch.setattr(claude_agent_sdk, "TextBlock", _FakeTextBlock)
    monkeypatch.setattr(claude_agent_sdk, "ResultMessage", _FakeResultMessage)
    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    monkeypatch.setattr(tracing, "get_span_recorder", lambda: span_recorder)

    result = (
        await _instantiate_provider().call(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=0,
            scope="test",
        )
    ).content

    assert result == "ok"
    span_recorder.record_llm_call.assert_called_once()


@pytest.mark.asyncio
async def test_call_with_tools_raises_with_result_text_on_error_result(monkeypatch):
    """call_with_tools() must surface ResultMessage.result the same way."""
    import claude_agent_sdk

    class _FakeClient:
        def __init__(self, options: _FakeOptions) -> None:
            self.options = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self):
            yield _FakeResultMessage(subtype="success", is_error=True, result=QUOTA_ERROR_TEXT)

    @dataclass
    class _FakeSdkMcpTool:
        name: str
        description: str
        input_schema: dict[str, Any]
        handler: Any

    def fake_create_sdk_mcp_server(name: str, version: str, tools=None):
        return {"name": name, "version": version, "tools": tools}

    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(claude_agent_sdk, "AssistantMessage", _FakeAssistantMessage)
    monkeypatch.setattr(claude_agent_sdk, "TextBlock", _FakeTextBlock)
    monkeypatch.setattr(claude_agent_sdk, "ResultMessage", _FakeResultMessage)
    monkeypatch.setattr(claude_agent_sdk, "ToolUseBlock", type("ToolUseBlock", (), {}))
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _FakeClient)
    monkeypatch.setattr(claude_agent_sdk, "SdkMcpTool", _FakeSdkMcpTool)
    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server", fake_create_sdk_mcp_server)

    provider = _instantiate_provider()
    with pytest.raises(ProviderRateLimitResetError) as excinfo:
        await provider.call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "function": {
                        "name": "noop",
                        "description": "no-op",
                        "parameters": {"type": "object", "properties": {}},
                    }
                }
            ],
            max_retries=2,
            scope="test",
        )

    assert QUOTA_ERROR_TEXT in str(excinfo.value)
    assert excinfo.value.retry_at > datetime.now(UTC)


@pytest.mark.asyncio
async def test_call_keeps_a_fence_marker_inside_a_json_value(monkeypatch):
    """A ``` inside a string value must not cut the payload short (#4819)."""
    import claude_agent_sdk

    fenced = '```json\n{"fact": "wrap it in ```json fences"}\n```'

    async def fake_query(prompt: str, options: _FakeOptions):
        yield _FakeAssistantMessage(content=[_FakeTextBlock(text=fenced)])
        yield _FakeResultMessage(subtype="success", is_error=False, result=fenced)

    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(claude_agent_sdk, "AssistantMessage", _FakeAssistantMessage)
    monkeypatch.setattr(claude_agent_sdk, "TextBlock", _FakeTextBlock)
    monkeypatch.setattr(claude_agent_sdk, "ResultMessage", _FakeResultMessage)
    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    result = (
        await _instantiate_provider().call(
            messages=[{"role": "user", "content": "extract facts"}],
            response_format=_StructuredResponse,
            max_retries=0,
            scope="retain_extract_facts",
        )
    ).content

    assert result.fact == "wrap it in ```json fences"


@pytest.mark.asyncio
async def test_call_turns_a_subscription_limit_into_a_reset_time(monkeypatch):
    """#5394: a session limit must reach the worker as its defer signal, raised on the
    first attempt — retrying inside the backoff cannot outlast the limit."""
    import claude_agent_sdk

    calls = 0

    async def fake_query(prompt: str, options: _FakeOptions):
        nonlocal calls
        calls += 1
        yield _FakeResultMessage(
            subtype="success", is_error=True, result="You've hit your session limit · resets 12:20am (UTC)"
        )

    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(claude_agent_sdk, "AssistantMessage", _FakeAssistantMessage)
    monkeypatch.setattr(claude_agent_sdk, "TextBlock", _FakeTextBlock)
    monkeypatch.setattr(claude_agent_sdk, "ResultMessage", _FakeResultMessage)
    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    with pytest.raises(ProviderRateLimitResetError) as excinfo:
        await _instantiate_provider().call(messages=[{"role": "user", "content": "hi"}], max_retries=2, scope="test")

    assert calls == 1
    assert excinfo.value.retry_at > datetime.now(UTC)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The field report: a bare time means its next occurrence, here after midnight.
        ("You've hit your session limit · resets 12:20am (UTC)", datetime(2026, 10, 4, 0, 20, tzinfo=UTC)),
        ("You've hit your limit · resets 11pm (UTC)", datetime(2026, 10, 3, 23, 0, tzinfo=UTC)),
        ("You've hit your weekly limit · resets Oct 9, 5pm (Europe/Rome)", datetime(2026, 10, 9, 15, 0, tzinfo=UTC)),
        # A date already past this year is next year's.
        (QUOTA_ERROR_TEXT, datetime(2027, 7, 18, 12, 0, tzinfo=UTC)),
        # No usable reset time: stay an ordinary error.
        ("You've hit your limit · resets 11pm (Nowhere/Zone)", None),
        (OTHER_ERROR_TEXT, None),
    ],
)
def test_limit_reset_at(text, expected):
    assert _limit_reset_at(text, datetime(2026, 10, 3, 22, 10, tzinfo=UTC)) == expected

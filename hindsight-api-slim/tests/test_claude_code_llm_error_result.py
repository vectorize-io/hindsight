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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

QUOTA_ERROR_TEXT = "You've hit your weekly limit · resets Jul 18, 12pm (UTC)"


@dataclass
class _FakeOptions:
    """Stand-in for ClaudeAgentOptions; captures kwargs without importing SDK."""

    system_prompt: str | None = None
    max_turns: int | None = None
    allowed_tools: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    mcp_servers: dict[str, Any] = field(default_factory=dict)


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
        yield _FakeResultMessage(subtype="success", is_error=True, result=QUOTA_ERROR_TEXT)

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

    assert QUOTA_ERROR_TEXT in str(excinfo.value)
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
    result = await provider.call(
        messages=[{"role": "user", "content": "hi"}],
        max_retries=0,
        scope="test",
    )

    assert result == "ok"


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
    with pytest.raises(RuntimeError) as excinfo:
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
            max_retries=0,
            scope="test",
        )

    assert QUOTA_ERROR_TEXT in str(excinfo.value)

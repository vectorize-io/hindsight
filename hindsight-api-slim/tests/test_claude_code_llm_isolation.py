"""Regression test for the Claude Code provider's subprocess isolation.

When an operator runs the API with HINDSIGHT_API_*_LLM_PROVIDER=claude-code,
the Claude Agent SDK spawns the `claude` CLI as a subprocess for every LLM
call. If that subprocess inherits the host's CLAUDE_CONFIG_DIR, it loads any
user-installed plugins (e.g. hindsight-memory). Their Stop hooks then retain
the subprocess's own transcript back into the same bank, causing a recursive
feedback loop documented in issue #1751.

The fix redirects the subprocess to an isolated config dir via env. This test
mocks the SDK to capture the ClaudeAgentOptions passed at call time and asserts
the isolation env is present on both code paths (`call` and `call_with_tools`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest


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


def _instantiate_provider():
    from hindsight_api.engine.providers.claude_code_llm import ClaudeCodeLLM

    return ClaudeCodeLLM(
        provider="claude-code",
        api_key="",
        base_url="",
        model="claude-haiku-4-5",
        reasoning_effort="low",
    )


def _assert_isolation_env(env: dict[str, str]) -> None:
    assert "CLAUDE_CONFIG_DIR" in env, "Subprocess env missing CLAUDE_CONFIG_DIR redirect"
    assert env["CLAUDE_CONFIG_DIR"], "CLAUDE_CONFIG_DIR must be a non-empty path"
    assert env["CLAUDE_CONFIG_DIR"].startswith("/"), "CLAUDE_CONFIG_DIR must be an absolute path"
    assert env.get("CLAUDE_SECURESTORAGE_CONFIG_DIR") == "", (
        "CLAUDE_SECURESTORAGE_CONFIG_DIR must be set to '' to force the un-suffixed keychain entry"
    )


@pytest.mark.asyncio
async def test_call_passes_isolation_env_to_sdk_options(monkeypatch):
    """call() must pass CLAUDE_CONFIG_DIR + SECURESTORAGE_CONFIG_DIR='' to the spawned CLI."""
    import claude_agent_sdk

    captured: dict[str, _FakeOptions] = {}

    async def fake_query(prompt: str, options: _FakeOptions):
        captured["options"] = options
        yield _FakeAssistantMessage(content=[_FakeTextBlock(text="ok")])

    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(claude_agent_sdk, "AssistantMessage", _FakeAssistantMessage)
    monkeypatch.setattr(claude_agent_sdk, "TextBlock", _FakeTextBlock)
    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    provider = _instantiate_provider()
    result = await provider.call(
        messages=[{"role": "user", "content": "hi"}],
        max_retries=0,
        scope="test",
    )

    assert result == "ok"
    assert "options" in captured, "fake query was not called"
    _assert_isolation_env(captured["options"].env)


@pytest.mark.asyncio
async def test_call_with_tools_passes_isolation_env_to_sdk_options(monkeypatch):
    """call_with_tools() must apply the same isolation env."""
    import claude_agent_sdk

    captured: dict[str, _FakeOptions] = {}

    class _FakeClient:
        def __init__(self, options: _FakeOptions) -> None:
            captured["options"] = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self):
            yield _FakeAssistantMessage(content=[_FakeTextBlock(text="ok")])

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
    monkeypatch.setattr(claude_agent_sdk, "ToolUseBlock", type("ToolUseBlock", (), {}))
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _FakeClient)
    monkeypatch.setattr(claude_agent_sdk, "SdkMcpTool", _FakeSdkMcpTool)
    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server", fake_create_sdk_mcp_server)

    provider = _instantiate_provider()
    result = await provider.call_with_tools(
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

    assert result.content == "ok"
    assert "options" in captured, "fake client was not constructed"
    _assert_isolation_env(captured["options"].env)


def test_isolation_dir_is_reused_across_calls():
    """The isolated dir is created once per process; concurrent calls share it."""
    from hindsight_api.engine.providers.claude_code_llm import _get_isolated_claude_env

    env_a = _get_isolated_claude_env()
    env_b = _get_isolated_claude_env()
    assert env_a is env_b, "Isolated env dict must be cached for the process lifetime"
    assert env_a["CLAUDE_CONFIG_DIR"] == env_b["CLAUDE_CONFIG_DIR"]


def test_isolation_dir_is_not_home():
    """Guard against the isolated dir accidentally pointing at $HOME (would not isolate plugins)."""
    import os

    from hindsight_api.engine.providers.claude_code_llm import _get_isolated_claude_env

    env = _get_isolated_claude_env()
    assert env["CLAUDE_CONFIG_DIR"] != os.path.expanduser("~"), (
        "Isolated dir must not be $HOME, or it would still load installed_plugins.json"
    )

"""Regression tests for issue #2966.

``call_with_tools()`` is a *single round* of an agentic loop the caller drives:
the model proposes tool call(s), the provider returns them, and the orchestrator
(reflect/agent.py) executes the REAL tools and feeds the results back on the next
call. See providers/openai_compatible_llm.py for the reference contract.

The Claude Agent SDK, however, runs its own in-process loop and invokes our SDK
MCP handlers — which are deliberate placeholders returning
``[Tool <name> called successfully]`` with no real data. With ``max_turns >= 2``
the model calls a search tool, sees the empty placeholder, re-queries, exhausts
the turn budget, and the run ends in ``error_max_turns`` *with its tool calls
discarded* — the "0 tool calls / no information" reflect failure in #2966.

The fix caps the SDK at ``max_turns=1`` and:
  * breaks out of the stream as soon as a tool call is proposed (so the SDK never
    acts on a placeholder result), returning the call to the caller; and
  * treats the ``error_max_turns`` ResultMessage that necessarily follows a
    single-turn tool call as non-fatal — the tool call is the result, not an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_AUTO


@dataclass
class _FakeOptions:
    """Stand-in for ClaudeAgentOptions; captures kwargs without importing the SDK."""

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


class _FakeToolUseBlock:
    def __init__(self, id: str, name: str, input: dict[str, Any]) -> None:
        self.id = id
        self.name = name
        self.input = input


class _FakeResultMessage:
    def __init__(self, subtype: str, is_error: bool, result: str | None) -> None:
        self.subtype = subtype
        self.is_error = is_error
        self.result = result


@dataclass
class _FakeSdkMcpTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Any


def _fake_create_sdk_mcp_server(name: str, version: str, tools=None):
    return {"name": name, "version": version, "tools": tools}


def _install_fake_sdk(monkeypatch, client_cls) -> None:
    import claude_agent_sdk

    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(claude_agent_sdk, "AssistantMessage", _FakeAssistantMessage)
    monkeypatch.setattr(claude_agent_sdk, "TextBlock", _FakeTextBlock)
    monkeypatch.setattr(claude_agent_sdk, "ToolUseBlock", _FakeToolUseBlock)
    monkeypatch.setattr(claude_agent_sdk, "ResultMessage", _FakeResultMessage)
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", client_cls)
    monkeypatch.setattr(claude_agent_sdk, "SdkMcpTool", _FakeSdkMcpTool)
    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server", _fake_create_sdk_mcp_server)


def _instantiate_provider():
    from hindsight_api.engine.providers.claude_code_llm import ClaudeCodeLLM

    return ClaudeCodeLLM(
        provider="claude-code",
        api_key="",
        base_url="",
        model="claude-haiku-4-5",
        reasoning_effort="low",
    )


_RECALL_TOOL = {
    "function": {
        "name": "recall",
        "description": "Search the memory bank.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
}


@pytest.mark.asyncio
async def test_tool_call_returned_despite_error_max_turns(monkeypatch):
    """A single-turn tool call followed by error_max_turns must be returned, not raised.

    This is the exact #2966 failure shape: the model emits a tool call and, because
    the single turn was spent on it, the CLI reports ``error_max_turns``. The tool
    call is the intended result and must reach the caller.
    """
    captured_options: dict[str, Any] = {}

    class _FakeClient:
        def __init__(self, options: _FakeOptions) -> None:
            captured_options["opts"] = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self):
            yield _FakeAssistantMessage(
                content=[
                    _FakeToolUseBlock(
                        id="tu_1",
                        name="mcp__hindsight_tools__recall",
                        input={"query": "domain family repositories"},
                    )
                ]
            )
            # With max_turns=1, the tool call consumes the only turn, so the CLI
            # necessarily follows up with error_max_turns.
            yield _FakeResultMessage(subtype="error_max_turns", is_error=True, result=None)

    _install_fake_sdk(monkeypatch, _FakeClient)

    provider = _instantiate_provider()
    result = await provider.call_with_tools(
        messages=[{"role": "user", "content": "What repos are in the domain family?"}],
        tools=[_RECALL_TOOL],
        max_retries=0,
        scope="reflect",
        tool_choice=LLM_TOOL_CHOICE_AUTO,
    )

    assert result.finish_reason == "tool_calls"
    assert [tc.name for tc in result.tool_calls] == ["recall"]
    assert result.tool_calls[0].arguments == {"query": "domain family repositories"}
    # The MCP prefix must be stripped for the caller.
    assert not result.tool_calls[0].name.startswith("mcp__")
    # The provider must cap the SDK at one turn so it never acts on placeholder results.
    assert captured_options["opts"].max_turns == 1
    # The configured model must be pinned on the SDK options (issue #2881), otherwise
    # the CLI silently runs its own default model.
    assert captured_options["opts"].model == "claude-haiku-4-5"


@pytest.mark.asyncio
async def test_stops_after_first_tool_round(monkeypatch):
    """The provider must stop consuming the stream once a tool call is proposed.

    If it kept reading, the SDK's next placeholder-driven turn would append more
    (duplicate/reworded) recall calls — the runaway loop that exhausts the budget.
    """
    later_turns_consumed = {"value": False}

    class _FakeClient:
        def __init__(self, options: _FakeOptions) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self):
            yield _FakeAssistantMessage(
                content=[
                    _FakeToolUseBlock(
                        id="tu_1",
                        name="mcp__hindsight_tools__recall",
                        input={"query": "first"},
                    )
                ]
            )
            # These would only be produced by the SDK re-prompting the model with a
            # placeholder result. The provider must break before pulling them.
            later_turns_consumed["value"] = True
            yield _FakeAssistantMessage(
                content=[
                    _FakeToolUseBlock(
                        id="tu_2",
                        name="mcp__hindsight_tools__recall",
                        input={"query": "reworded"},
                    )
                ]
            )

    _install_fake_sdk(monkeypatch, _FakeClient)

    provider = _instantiate_provider()
    result = await provider.call_with_tools(
        messages=[{"role": "user", "content": "hi"}],
        tools=[_RECALL_TOOL],
        max_retries=0,
        scope="reflect",
        tool_choice=LLM_TOOL_CHOICE_AUTO,
    )

    assert [tc.arguments["query"] for tc in result.tool_calls] == ["first"]
    assert later_turns_consumed["value"] is False


@pytest.mark.asyncio
async def test_text_only_answer_returned(monkeypatch):
    """A text-only turn (no tool call) is returned as a normal 'stop' response.

    This is the loop's final round: the orchestrator has already fed tool results
    back and the model answers in prose.
    """

    class _FakeClient:
        def __init__(self, options: _FakeOptions) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self):
            yield _FakeAssistantMessage(content=[_FakeTextBlock(text="The domain family has 8 repositories.")])
            yield _FakeResultMessage(subtype="success", is_error=False, result="done")

    _install_fake_sdk(monkeypatch, _FakeClient)

    provider = _instantiate_provider()
    result = await provider.call_with_tools(
        messages=[{"role": "user", "content": "answer now"}],
        tools=[_RECALL_TOOL],
        max_retries=0,
        scope="reflect",
        tool_choice=LLM_TOOL_CHOICE_AUTO,
    )

    assert result.finish_reason == "stop"
    assert result.tool_calls == []
    assert result.content == "The domain family has 8 repositories."


@pytest.mark.asyncio
async def test_call_pins_configured_model(monkeypatch):
    """call() must pass the configured model to the SDK (issue #2881).

    Without model= the spawned CLI runs its own default model regardless of
    HINDSIGHT_API_*_LLM_MODEL, while metrics/logs still print the configured one.
    """
    import claude_agent_sdk

    captured: dict[str, Any] = {}

    async def fake_query(prompt: str, options: _FakeOptions):
        captured["opts"] = options
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
    assert captured["opts"].model == "claude-haiku-4-5"


@pytest.mark.asyncio
async def test_genuine_error_without_tool_calls_still_raises(monkeypatch):
    """An error ResultMessage with nothing collected must still surface (issue #2702)."""

    class _FakeClient:
        def __init__(self, options: _FakeOptions) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self):
            yield _FakeResultMessage(
                subtype="success",
                is_error=True,
                result="You've hit your weekly limit · resets Jul 18, 12pm (UTC)",
            )

    _install_fake_sdk(monkeypatch, _FakeClient)

    provider = _instantiate_provider()
    with pytest.raises(RuntimeError, match="weekly limit"):
        await provider.call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[_RECALL_TOOL],
            max_retries=0,
            scope="reflect",
            tool_choice=LLM_TOOL_CHOICE_AUTO,
        )

"""Consumer-style tests using real Claude Agent SDK types and MCP server objects."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import mcp.types as mcp_types
import pytest
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, ResultMessage
from hindsight_claude_agent_sdk import MemoryHookConfig, create_hindsight_server, create_memory_hooks


def _mock_client():
    client = MagicMock()
    client.aretain = AsyncMock()
    client.arecall = AsyncMock()
    client.areflect = AsyncMock()
    return client


def _mock_recall_response(texts: list[str]):
    response = MagicMock()
    response.results = []
    for text in texts:
        result = MagicMock()
        result.text = text
        response.results.append(result)
    return response


def _mock_reflect_response(text: str):
    response = MagicMock()
    response.text = text
    return response


def _transcript_path(tmp_path: Path, result_text: str) -> str:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "duration_ms": 1,
                "duration_api_ms": 1,
                "is_error": False,
                "num_turns": 1,
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "result": result_text,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return str(transcript)


@pytest.mark.asyncio
async def test_sdk_server_lists_tools_with_expected_annotations():
    client = _mock_client()
    server_config = create_hindsight_server(bank_id="test-bank", client=client)

    options = ClaudeAgentOptions(
        mcp_servers={"hindsight": server_config},
        allowed_tools=["mcp__hindsight__*"],
    )
    assert options.mcp_servers["hindsight"]["type"] == "sdk"

    server = server_config["instance"]
    result = await server.request_handlers[mcp_types.ListToolsRequest](None)
    tools = result.root.tools

    assert [tool.name for tool in tools] == [
        "hindsight_retain",
        "hindsight_recall",
        "hindsight_reflect",
    ]
    assert tools[0].annotations.readOnlyHint is False
    assert tools[1].annotations.readOnlyHint is True
    assert tools[1].annotations.idempotentHint is True
    assert tools[2].annotations.openWorldHint is True


@pytest.mark.asyncio
async def test_sdk_server_call_tool_executes_retain_recall_and_reflect():
    client = _mock_client()
    client.arecall.return_value = _mock_recall_response(["User prefers async Python"])
    client.areflect.return_value = _mock_reflect_response("The user prefers async Python with pytest.")

    server = create_hindsight_server(bank_id="test-bank", client=client)["instance"]
    call_handler = server.request_handlers[mcp_types.CallToolRequest]

    retain_request = mcp_types.CallToolRequest.model_validate(
        {
            "method": "tools/call",
            "params": {
                "name": "hindsight_retain",
                "arguments": {"content": "The user prefers async Python"},
            },
        }
    )
    retain_result = await call_handler(retain_request)
    assert retain_result.root.isError is False
    assert retain_result.root.content[0].text == "Memory stored successfully."

    recall_request = mcp_types.CallToolRequest.model_validate(
        {
            "method": "tools/call",
            "params": {
                "name": "hindsight_recall",
                "arguments": {"query": "user preferences"},
            },
        }
    )
    recall_result = await call_handler(recall_request)
    assert recall_result.root.isError is False
    assert recall_result.root.content[0].text == "1. User prefers async Python"

    reflect_request = mcp_types.CallToolRequest.model_validate(
        {
            "method": "tools/call",
            "params": {
                "name": "hindsight_reflect",
                "arguments": {"query": "Summarize the user's preferences"},
            },
        }
    )
    reflect_result = await call_handler(reflect_request)
    assert reflect_result.root.isError is False
    assert reflect_result.root.content[0].text == "The user prefers async Python with pytest."


@pytest.mark.asyncio
async def test_sdk_server_call_tool_validates_input_schema():
    client = _mock_client()
    server = create_hindsight_server(bank_id="test-bank", client=client)["instance"]
    call_handler = server.request_handlers[mcp_types.CallToolRequest]

    invalid_request = mcp_types.CallToolRequest.model_validate(
        {
            "method": "tools/call",
            "params": {
                "name": "hindsight_recall",
                "arguments": {},
            },
        }
    )

    result = await call_handler(invalid_request)

    assert result.root.isError is True
    assert "Input validation error" in result.root.content[0].text


@pytest.mark.asyncio
async def test_hooks_are_consumable_via_claude_agent_options_and_use_real_shapes(tmp_path):
    client = _mock_client()
    client.arecall.return_value = _mock_recall_response(["User prefers pytest", "User uses FastAPI"])

    hooks = create_memory_hooks(
        bank_id="test-bank",
        client=client,
        hook_config=MemoryHookConfig(retain_on_tools=["Bash"]),
    )

    options = ClaudeAgentOptions(hooks=hooks)
    assert isinstance(options.hooks["UserPromptSubmit"][0], HookMatcher)
    assert isinstance(options.hooks["Stop"][0], HookMatcher)
    assert options.hooks["PostToolUse"][0].matcher == "Bash"

    recall_hook = hooks["UserPromptSubmit"][0].hooks[0]
    recall_result = await recall_hook(
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": "123e4567-e89b-12d3-a456-426614174000",
            "transcript_path": str(tmp_path / "unused.jsonl"),
            "cwd": str(tmp_path),
            "prompt": "What stack do I prefer?",
        },
        None,
        {"signal": None},
    )
    assert "systemMessage" in recall_result
    assert "1. User prefers pytest" in recall_result["systemMessage"]
    assert "2. User uses FastAPI" in recall_result["systemMessage"]

    tool_hook = hooks["PostToolUse"][0].hooks[0]
    await tool_hook(
        {
            "hook_event_name": "PostToolUse",
            "session_id": "123e4567-e89b-12d3-a456-426614174000",
            "transcript_path": str(tmp_path / "unused.jsonl"),
            "cwd": str(tmp_path),
            "tool_name": "Bash",
            "tool_input": {"command": "pytest -q"},
            "tool_response": "42 passed in 0.71s with detailed output",
            "tool_use_id": "tool-123",
        },
        "tool-123",
        {"signal": None},
    )
    retain_call = client.aretain.call_args_list[-1].kwargs
    assert "pytest -q" in retain_call["content"]
    assert "42 passed" in retain_call["content"]
    assert "tool:Bash" in retain_call["tags"]

    stop_hook = hooks["Stop"][0].hooks[0]
    transcript_path = _transcript_path(
        tmp_path,
        "I updated the tests and documented the preferred FastAPI patterns.",
    )
    await stop_hook(
        {
            "hook_event_name": "Stop",
            "session_id": "123e4567-e89b-12d3-a456-426614174000",
            "transcript_path": transcript_path,
            "cwd": str(tmp_path),
            "stop_hook_active": False,
        },
        None,
        {"signal": None},
    )
    final_retain_call = client.aretain.call_args_list[-1].kwargs
    assert "preferred FastAPI patterns" in final_retain_call["content"]


def test_recipe_style_result_message_shape_is_available_for_consumers():
    message = ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="123e4567-e89b-12d3-a456-426614174000",
        result="Done",
    )

    assert message.subtype == "success"
    assert message.result == "Done"

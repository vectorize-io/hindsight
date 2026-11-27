"""The deterministic piece of the coding-agents session runner: reading tool calls out of stream-json.

The evals assert on these calls ("did the turn capture an initiative?"), so a parser that silently
returned nothing would turn every "the agent must not capture" check into a pass.
"""

from __future__ import annotations

from hindsight_system_evals.coding_agents.session import ToolCall, _offered, _tool_calls


def test_reads_every_tool_use_block_and_strips_the_mcp_prefix() -> None:
    event = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "Recording the initiative."},
                {
                    "type": "tool_use",
                    "name": "mcp__hindsight__hindsight_capture_initiative",
                    "input": {"title": "Gift cards", "summary": "…"},
                },
                {"type": "tool_use", "name": "Edit", "input": {"file_path": "a.py"}},
            ]
        },
    }
    assert _tool_calls(event) == [
        ToolCall(name="hindsight_capture_initiative", input={"title": "Gift cards", "summary": "…"}),
        ToolCall(name="Edit", input={"file_path": "a.py"}),
    ]


def test_a_message_without_tool_calls_has_none() -> None:
    assert _tool_calls({"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"}]}}) == []
    assert _tool_calls({"type": "assistant"}) == []


def test_the_init_event_lists_the_offered_tools() -> None:
    event = {
        "type": "system",
        "subtype": "init",
        "tools": ["Edit", "mcp__hindsight__hindsight_capture_initiative"],
    }
    assert _offered(event) == ["Edit", "hindsight_capture_initiative"]
    assert _offered({"type": "system", "subtype": "init"}) == []

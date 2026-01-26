"""
Tests for LLM tool calling functionality.
"""

import pytest

from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult


# Sample tools for testing
SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
]


class TestMockToolCalling:
    """Test tool calling with mock provider."""

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_tool_calls(self):
        """Test that mock provider can return tool calls."""
        llm = LLMProvider(provider="mock", api_key="", base_url="", model="mock")

        # Set mock response to return tool calls
        llm.set_mock_response([
            {"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}},
        ])

        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=SAMPLE_TOOLS,
        )

        assert isinstance(result, LLMToolCallResult)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"location": "Paris", "unit": "celsius"}
        assert result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_call_with_tools_returns_content(self):
        """Test that mock provider can return plain content."""
        llm = LLMProvider(provider="mock", api_key="", base_url="", model="mock")

        # Default mock response is plain content
        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "Hello"}],
            tools=SAMPLE_TOOLS,
        )

        assert isinstance(result, LLMToolCallResult)
        assert result.content == "mock response"
        assert len(result.tool_calls) == 0
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_call_with_tools_records_calls(self):
        """Test that mock calls are recorded."""
        llm = LLMProvider(provider="mock", api_key="", base_url="", model="mock")
        llm.clear_mock_calls()

        await llm.call_with_tools(
            messages=[{"role": "user", "content": "Test message"}],
            tools=SAMPLE_TOOLS,
            scope="test_scope",
        )

        calls = llm.get_mock_calls()
        assert len(calls) == 1
        assert calls[0]["scope"] == "test_scope"
        assert "get_weather" in calls[0]["tools"]
        assert "search" in calls[0]["tools"]

    @pytest.mark.asyncio
    async def test_call_with_tools_multiple_tool_calls(self):
        """Test handling multiple tool calls in one response."""
        llm = LLMProvider(provider="mock", api_key="", base_url="", model="mock")

        llm.set_mock_response([
            {"name": "get_weather", "arguments": {"location": "Paris"}},
            {"name": "search", "arguments": {"query": "weather forecast"}},
        ])

        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "Weather in Paris and search for forecasts"}],
            tools=SAMPLE_TOOLS,
        )

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[1].name == "search"

    @pytest.mark.asyncio
    async def test_call_with_tools_accepts_llm_tool_call_result(self):
        """Test that mock can accept LLMToolCallResult directly."""
        llm = LLMProvider(provider="mock", api_key="", base_url="", model="mock")

        expected_result = LLMToolCallResult(
            content="Here's the info",
            tool_calls=[LLMToolCall(id="call_123", name="search", arguments={"query": "test"})],
            finish_reason="tool_calls",
        )
        llm.set_mock_response(expected_result)

        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "Search for test"}],
            tools=SAMPLE_TOOLS,
        )

        assert result == expected_result


class TestToolCallConversation:
    """Test tool call conversation flow."""

    @pytest.mark.asyncio
    async def test_tool_result_message_format(self):
        """Test that tool result messages can be passed in subsequent calls."""
        llm = LLMProvider(provider="mock", api_key="", base_url="", model="mock")

        # First call returns tool call
        llm.set_mock_response([{"name": "get_weather", "arguments": {"location": "Paris"}}])

        result1 = await llm.call_with_tools(
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=SAMPLE_TOOLS,
        )

        # Build conversation with tool result
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": result1.tool_calls[0].id,
                        "type": "function",
                        "function": {
                            "name": result1.tool_calls[0].name,
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": result1.tool_calls[0].id,
                "content": '{"temperature": 20, "conditions": "sunny"}',
            },
        ]

        # Second call should work with tool result in history
        llm.set_mock_response(None)  # Reset to default
        result2 = await llm.call_with_tools(
            messages=messages,
            tools=SAMPLE_TOOLS,
        )

        assert result2.content == "mock response"


class TestToolSchemas:
    """Test tool schema handling."""

    @pytest.mark.asyncio
    async def test_empty_tools_list(self):
        """Test calling with empty tools list."""
        llm = LLMProvider(provider="mock", api_key="", base_url="", model="mock")

        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[],
        )

        assert result.content == "mock response"

    @pytest.mark.asyncio
    async def test_tool_with_no_required_params(self):
        """Test tool with no required parameters."""
        llm = LLMProvider(provider="mock", api_key="", base_url="", model="mock")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_items",
                    "description": "List all items",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]

        llm.set_mock_response([{"name": "list_items", "arguments": {}}])

        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "List items"}],
            tools=tools,
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "list_items"
        assert result.tool_calls[0].arguments == {}


class TestReflectToolSchemas:
    """Test reflect agent tool schemas."""

    def test_get_reflect_tools_default(self):
        """Test getting default reflect tools."""
        from hindsight_api.engine.reflect.tools_schema import get_reflect_tools

        tools = get_reflect_tools()

        tool_names = [t["function"]["name"] for t in tools]
        assert "search_mental_models" in tool_names
        assert "search_observations" in tool_names
        assert "recall" in tool_names
        assert "expand" in tool_names
        assert "done" in tool_names

    def test_get_reflect_tools_with_directives(self):
        """Test getting reflect tools with directive rules."""
        from hindsight_api.engine.reflect.tools_schema import get_reflect_tools

        tools = get_reflect_tools(directive_rules=["Always respond in French"])

        tool_names = [t["function"]["name"] for t in tools]
        assert "recall" in tool_names
        assert "done" in tool_names

        # Done tool should have directive_compliance field when directives are present
        done_tool = next(t for t in tools if t["function"]["name"] == "done")
        params = done_tool["function"]["parameters"]["properties"]
        assert "directive_compliance" in params

    def test_get_reflect_tools_answer_mode(self):
        """Test getting reflect tools with answer output mode."""
        from hindsight_api.engine.reflect.tools_schema import get_reflect_tools

        tools = get_reflect_tools()

        done_tool = next(t for t in tools if t["function"]["name"] == "done")
        params = done_tool["function"]["parameters"]["properties"]

        assert "answer" in params
        assert "memory_ids" in params
        assert "observation_ids" in params
        assert "mental_model_ids" in params


class TestLLMToolCallResult:
    """Test LLMToolCallResult model."""

    def test_tool_call_result_defaults(self):
        """Test default values for LLMToolCallResult."""
        result = LLMToolCallResult()

        assert result.content is None
        assert result.tool_calls == []
        assert result.finish_reason is None

    def test_tool_call_result_with_content(self):
        """Test LLMToolCallResult with content."""
        result = LLMToolCallResult(content="Hello", finish_reason="stop")

        assert result.content == "Hello"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"

    def test_tool_call_result_with_tool_calls(self):
        """Test LLMToolCallResult with tool calls."""
        result = LLMToolCallResult(
            tool_calls=[
                LLMToolCall(id="call_1", name="test_tool", arguments={"arg": "value"}),
            ],
            finish_reason="tool_calls",
        )

        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "test_tool"
        assert result.finish_reason == "tool_calls"


class TestLLMToolCall:
    """Test LLMToolCall model."""

    def test_tool_call_basic(self):
        """Test basic LLMToolCall creation."""
        call = LLMToolCall(id="call_123", name="get_weather", arguments={"location": "Paris"})

        assert call.id == "call_123"
        assert call.name == "get_weather"
        assert call.arguments == {"location": "Paris"}

    def test_tool_call_empty_arguments(self):
        """Test LLMToolCall with empty arguments."""
        call = LLMToolCall(id="call_456", name="list_items", arguments={})

        assert call.arguments == {}

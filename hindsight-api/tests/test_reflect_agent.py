"""
Tests for the reflect agent with mocked LLM outputs.

These tests verify:
1. Tool name normalization for various LLM output formats
2. Recovery from unknown tool calls
3. Recovery from tool execution errors
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from hindsight_api.engine.reflect.agent import (
    _normalize_tool_name,
    _is_done_tool,
    _clean_answer_text,
    _clean_done_answer,
    run_reflect_agent,
)
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage


class TestCleanAnswerText:
    """Test cleanup of answer text that includes done() tool call syntax."""

    def test_clean_text_with_done_call(self):
        """Text ending with done() call should have it stripped."""
        text = '''The team's OKRs focus on performance.done({"answer":"The team's OKRs","memory_ids":[]})'''
        cleaned = _clean_answer_text(text)
        assert cleaned == "The team's OKRs focus on performance."
        assert "done(" not in cleaned

    def test_clean_text_with_done_call_and_whitespace(self):
        """done() call with whitespace should be stripped."""
        text = '''Answer text here. done( {"answer": "short", "memory_ids": []} )'''
        cleaned = _clean_answer_text(text)
        assert cleaned == "Answer text here."

    def test_clean_text_without_done_call(self):
        """Text without done() call should be unchanged."""
        text = "This is a normal answer without any tool calls."
        cleaned = _clean_answer_text(text)
        assert cleaned == text

    def test_clean_text_with_done_word_in_content(self):
        """The word 'done' in regular text should not be stripped."""
        text = "The task is done and completed successfully."
        cleaned = _clean_answer_text(text)
        assert cleaned == text

    def test_clean_empty_text(self):
        """Empty text should return empty."""
        assert _clean_answer_text("") == ""

    def test_clean_text_multiline_done(self):
        """done() call spanning multiple lines should be stripped."""
        text = '''Summary of findings.done({
            "answer": "Summary",
            "memory_ids": ["id1", "id2"]
        })'''
        cleaned = _clean_answer_text(text)
        assert cleaned == "Summary of findings."


class TestCleanDoneAnswer:
    """Test cleanup of answer field from done() tool call that leaks structured output."""

    def test_clean_answer_with_leaked_json_code_block(self):
        """Answer with leaked JSON code block at the end should be cleaned."""
        text = '''The user's favorite color is blue.

```json
{"observation_ids": ["obs-1", "obs-2"]}
```'''
        cleaned = _clean_done_answer(text)
        assert cleaned == "The user's favorite color is blue."
        assert "observation_ids" not in cleaned

    def test_clean_answer_with_memory_ids_code_block(self):
        """Answer with leaked memory_ids JSON code block should be cleaned."""
        text = '''Here is the answer.

```json
{"memory_ids": ["mem-1"]}
```'''
        cleaned = _clean_done_answer(text)
        assert cleaned == "Here is the answer."

    def test_clean_answer_with_raw_json_object(self):
        """Answer with raw JSON object containing IDs at the end should be cleaned."""
        text = 'The answer is 42. {"observation_ids": ["obs-1"]}'
        cleaned = _clean_done_answer(text)
        assert cleaned == "The answer is 42."

    def test_clean_answer_with_trailing_ids_pattern(self):
        """Answer with 'observation_ids: [...]' pattern at the end should be cleaned."""
        text = "This is the answer.\n\nobservation_ids: [\"obs-1\", \"obs-2\"]"
        cleaned = _clean_done_answer(text)
        assert cleaned == "This is the answer."

    def test_clean_answer_with_memory_ids_equals(self):
        """Answer with 'memory_ids = [...]' pattern at the end should be cleaned."""
        text = "Answer text here.\nmemory_ids = [\"mem-1\"]"
        cleaned = _clean_done_answer(text)
        assert cleaned == "Answer text here."

    def test_clean_normal_answer_unchanged(self):
        """Normal answer without leaked output should be unchanged."""
        text = "This is a normal answer about observation strategies."
        cleaned = _clean_done_answer(text)
        assert cleaned == text

    def test_clean_empty_answer(self):
        """Empty answer should return empty."""
        assert _clean_done_answer("") == ""

    def test_clean_answer_with_observation_word_in_content(self):
        """The word 'observation' in regular text should not be stripped."""
        text = "Based on my observation, the user prefers dark mode."
        cleaned = _clean_done_answer(text)
        assert cleaned == text

    def test_clean_answer_multiline_with_markdown(self):
        """Answer with markdown and leaked JSON at end should clean only the leak."""
        text = '''Summary:
- Point 1
- Point 2

```json
{"mental_model_ids": ["mm-1"]}
```'''
        cleaned = _clean_done_answer(text)
        assert "Point 1" in cleaned
        assert "Point 2" in cleaned
        assert "mental_model_ids" not in cleaned


class TestToolNameNormalization:
    """Test tool name normalization for various LLM output formats."""

    def test_normalize_standard_name(self):
        """Standard tool names should pass through unchanged."""
        assert _normalize_tool_name("done") == "done"
        assert _normalize_tool_name("recall") == "recall"
        assert _normalize_tool_name("search_mental_models") == "search_mental_models"
        assert _normalize_tool_name("search_observations") == "search_observations"
        assert _normalize_tool_name("expand") == "expand"

    def test_normalize_functions_prefix(self):
        """Tool names with 'functions.' prefix should be normalized."""
        assert _normalize_tool_name("functions.done") == "done"
        assert _normalize_tool_name("functions.recall") == "recall"
        assert _normalize_tool_name("functions.search_mental_models") == "search_mental_models"

    def test_normalize_call_equals_prefix(self):
        """Tool names with 'call=' prefix should be normalized."""
        assert _normalize_tool_name("call=done") == "done"
        assert _normalize_tool_name("call=recall") == "recall"

    def test_normalize_call_equals_functions_prefix(self):
        """Tool names with 'call=functions.' prefix should be normalized."""
        assert _normalize_tool_name("call=functions.done") == "done"
        assert _normalize_tool_name("call=functions.recall") == "recall"
        assert _normalize_tool_name("call=functions.search_observations") == "search_observations"

    def test_normalize_special_token_suffix(self):
        """Tool names with malformed special tokens should be normalized."""
        assert _normalize_tool_name("done<|channel|>commentary") == "done"
        assert _normalize_tool_name("recall<|endoftext|>") == "recall"
        assert _normalize_tool_name("search_observations<|im_end|>extra") == "search_observations"

    def test_is_done_tool(self):
        """Test _is_done_tool helper."""
        # Standard
        assert _is_done_tool("done") is True
        assert _is_done_tool("recall") is False

        # With prefixes
        assert _is_done_tool("functions.done") is True
        assert _is_done_tool("call=done") is True
        assert _is_done_tool("call=functions.done") is True

        # With malformed special tokens
        assert _is_done_tool("done<|channel|>commentary") is True
        assert _is_done_tool("done<|endoftext|>") is True

        # Not done
        assert _is_done_tool("functions.recall") is False
        assert _is_done_tool("call=functions.recall") is False
        assert _is_done_tool("recall<|channel|>done") is False


class TestReflectAgentMocked:
    """Test reflect agent with mocked LLM outputs."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        llm = MagicMock()
        llm.call_with_tools = AsyncMock()
        # Also mock call() for final iteration fallback - returns (response, usage) tuple
        llm.call = AsyncMock(
            return_value=("Fallback answer from final iteration", TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150))
        )
        return llm

    @pytest.fixture
    def mock_functions(self):
        """Create mock search/recall functions."""
        return {
            "search_mental_models_fn": AsyncMock(return_value={"mental_models": []}),
            "search_observations_fn": AsyncMock(return_value={"observations": []}),
            "recall_fn": AsyncMock(return_value={"memories": [{"id": "mem-1", "content": "test memory"}]}),
            "expand_fn": AsyncMock(return_value={"memories": []}),
        }

    @pytest.mark.asyncio
    async def test_handles_functions_prefix_in_done(self, mock_llm, mock_functions):
        """Test that 'functions.done' is handled correctly."""
        # First call: LLM calls recall
        # Second call: LLM calls functions.done
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="recall", arguments={"query": "test"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(
                        id="2",
                        name="functions.done",
                        arguments={"answer": "Test answer", "memory_ids": ["mem-1"]},
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            **mock_functions,
        )

        assert result.text == "Test answer"
        assert "mem-1" in result.used_memory_ids

    @pytest.mark.asyncio
    async def test_handles_call_equals_functions_prefix(self, mock_llm, mock_functions):
        """Test that 'call=functions.done' is handled correctly."""
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="recall", arguments={"query": "test"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(
                        id="2",
                        name="call=functions.done",
                        arguments={"answer": "Test answer", "memory_ids": ["mem-1"]},
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            **mock_functions,
        )

        assert result.text == "Test answer"

    @pytest.mark.asyncio
    async def test_recovery_from_unknown_tool(self, mock_llm, mock_functions):
        """Test that LLM can recover after calling an unknown tool."""
        # First call: LLM calls unknown tool
        # Second call: LLM calls valid recall after seeing error
        # Third call: LLM calls done
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="invalid_tool", arguments={"foo": "bar"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="2", name="recall", arguments={"query": "test"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(
                        id="3",
                        name="done",
                        arguments={"answer": "Recovered successfully", "memory_ids": ["mem-1"]},
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            **mock_functions,
        )

        assert result.text == "Recovered successfully"
        # Verify the LLM was called 3 times (initial + recovery + done)
        assert mock_llm.call_with_tools.call_count == 3

    @pytest.mark.asyncio
    async def test_recovery_from_tool_execution_error(self, mock_llm, mock_functions):
        """Test that LLM can recover after a tool execution fails."""
        # Make recall fail the first time, succeed the second time
        mock_functions["recall_fn"].side_effect = [
            Exception("Database connection failed"),
            {"memories": [{"id": "mem-1", "content": "test memory"}]},
        ]

        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="recall", arguments={"query": "test"})],
                finish_reason="tool_calls",
            ),
            # LLM tries again after seeing error
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="2", name="recall", arguments={"query": "test retry"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(
                        id="3",
                        name="done",
                        arguments={"answer": "Recovered from error", "memory_ids": ["mem-1"]},
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            **mock_functions,
        )

        assert result.text == "Recovered from error"
        assert mock_llm.call_with_tools.call_count == 3

    @pytest.mark.asyncio
    async def test_normalizes_tool_names_in_other_tools(self, mock_llm, mock_functions):
        """Test that tool names are normalized for all tools, not just done."""
        mock_llm.call_with_tools.side_effect = [
            # LLM calls 'functions.recall' instead of 'recall'
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="functions.recall", arguments={"query": "test"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(
                        id="2",
                        name="done",
                        arguments={"answer": "Test answer", "memory_ids": ["mem-1"]},
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            **mock_functions,
        )

        assert result.text == "Test answer"
        # Verify recall was actually called (normalization worked)
        mock_functions["recall_fn"].assert_called_once()

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self, mock_llm, mock_functions):
        """Test that agent stops after max iterations even with errors."""
        # LLM keeps calling unknown tools
        mock_llm.call_with_tools.return_value = LLMToolCallResult(
            tool_calls=[LLMToolCall(id="1", name="unknown_tool", arguments={})],
            finish_reason="tool_calls",
        )

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            max_iterations=3,
            **mock_functions,
        )

        # Should have a result even if no memories found
        assert result is not None
        assert result.iterations == 3

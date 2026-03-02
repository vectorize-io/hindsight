"""
Tests for the reflect agent with mocked LLM outputs.

These tests verify:
1. Tool name normalization for various LLM output formats
2. Recovery from unknown tool calls
3. Recovery from tool execution errors
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.reflect.agent import (
    _clean_answer_text,
    _clean_done_answer,
    _count_messages_tokens,
    _is_context_overflow_error,
    _is_done_tool,
    _normalize_tool_name,
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


class TestContextOverflowHelpers:
    """Unit tests for context-overflow detection helpers."""

    def test_count_messages_tokens_basic(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        count = _count_messages_tokens(messages)
        assert count > 0
        # Rough sanity check: ~10 tokens for each message
        assert count < 100

    def test_count_messages_tokens_with_tool_result(self):
        """A large tool result should substantially increase the count."""
        small_messages = [{"role": "user", "content": "hi"}]
        large_messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "tool",
                "tool_call_id": "x",
                "name": "recall",
                "content": '{"memories": [' + ', '.join([f'{{"id": "m{i}", "content": "A long memory fact about some topic that goes on and on."}}' for i in range(50)]) + ']}',
            },
        ]
        small = _count_messages_tokens(small_messages)
        large = _count_messages_tokens(large_messages)
        assert large > small + 200

    def test_is_context_overflow_error_openai(self):
        assert _is_context_overflow_error(Exception("context_length_exceeded: too many tokens"))
        assert _is_context_overflow_error(Exception("This model's maximum context length is 128000 tokens. However, your messages resulted in 142164 tokens."))

    def test_is_context_overflow_error_anthropic(self):
        assert _is_context_overflow_error(Exception("prompt_too_long"))
        assert _is_context_overflow_error(Exception("prompt is too long for this model"))

    def test_is_context_overflow_error_gemini(self):
        assert _is_context_overflow_error(Exception("RESOURCE_EXHAUSTED: quota exceeded"))

    def test_is_context_overflow_error_generic(self):
        assert _is_context_overflow_error(Exception("input is too long to process"))
        assert _is_context_overflow_error(Exception("too many tokens in the request"))

    def test_is_context_overflow_error_unrelated(self):
        assert not _is_context_overflow_error(Exception("connection timeout"))
        assert not _is_context_overflow_error(Exception("rate limit exceeded"))
        assert not _is_context_overflow_error(ValueError("invalid argument"))


class TestContextOverflowBehavior:
    """Test that the reflect agent handles context overflow gracefully."""

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.call_with_tools = AsyncMock()
        llm.call = AsyncMock(
            return_value=("Synthesized answer from gathered evidence.", TokenUsage(input_tokens=50, output_tokens=20, total_tokens=70))
        )
        return llm

    @pytest.fixture
    def mock_functions_with_large_output(self):
        """Mock functions that return a large enough payload to exceed a tiny token budget."""
        large_memories = [
            {"id": f"mem-{i}", "content": f"Memory fact number {i}: " + "A" * 200}
            for i in range(20)
        ]
        return {
            "search_mental_models_fn": AsyncMock(return_value={"mental_models": []}),
            "search_observations_fn": AsyncMock(return_value={"observations": []}),
            "recall_fn": AsyncMock(return_value={"memories": large_memories}),
            "expand_fn": AsyncMock(return_value={"memories": []}),
        }

    @pytest.mark.asyncio
    async def test_proactive_guard_fires_when_budget_exceeded(self, mock_llm, mock_functions_with_large_output):
        """When token count exceeds max_context_tokens after a tool call, the agent
        should immediately synthesize from gathered evidence instead of making
        another LLM call that would overflow."""
        # First call: LLM calls recall (forced by iter 0 with no mental models)
        mock_llm.call_with_tools.return_value = LLMToolCallResult(
            tool_calls=[LLMToolCall(id="1", name="recall", arguments={"query": "test"})],
            finish_reason="tool_calls",
        )

        # Set a tiny token budget — the recall result alone will blow past it
        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="What do you know?",
            bank_profile={"name": "Test", "mission": "Testing"},
            max_context_tokens=100,
            **mock_functions_with_large_output,
        )

        assert result.text == "Synthesized answer from gathered evidence."
        # call_with_tools was called once (for the forced recall), then the guard
        # kicked in — no further tool-call iterations
        assert mock_llm.call_with_tools.call_count == 1
        # llm.call() was invoked to generate the final synthesis
        mock_llm.call.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_overflow_error_skips_retry(self, mock_llm, mock_functions_with_large_output):
        """A context_length_exceeded error from the LLM should NOT be retried —
        it should immediately fall back to final synthesis."""
        mock_llm.call_with_tools.side_effect = Exception(
            "context_length_exceeded: messages resulted in 150000 tokens."
        )

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="What do you know?",
            bank_profile={"name": "Test", "mission": "Testing"},
            max_iterations=5,
            **mock_functions_with_large_output,
        )

        assert result is not None
        # Should have attempted only 1 iteration (no retry on overflow error)
        assert mock_llm.call_with_tools.call_count == 1
        # Final synthesis was called
        mock_llm.call.assert_called_once()


class TestContextOverflowIntegration:
    """Integration test: real LLM with a very small max_context_tokens.

    The agent will make one real LLM call (forced tool choice), receive a large
    tool result that exceeds the tiny budget, then synthesize from it via a second
    real LLM call — all without raising a context_length_exceeded error.
    """

    @pytest.mark.asyncio
    async def test_reflect_completes_with_tiny_context_budget(self, memory, request_context):
        """End-to-end: reflect on a bank with max_context_tokens=1 (tiny budget).

        Setting max_context_tokens=1 guarantees the proactive guard fires as soon
        as the first tool result is received and evidence is available.
        The result must be a non-empty string with no exception raised.
        """
        import uuid
        from unittest.mock import patch

        bank_id = f"test-ctx-overflow-{uuid.uuid4().hex[:8]}"
        try:
            # Retain a handful of facts so the recall tool has something to return
            await memory.retain_async(
                bank_id=bank_id,
                content="Alice is a software engineer who enjoys hiking on weekends.",
                request_context=request_context,
            )
            await memory.retain_async(
                bank_id=bank_id,
                content="Bob is a designer who loves cooking Italian food.",
                request_context=request_context,
            )

            # Patch get_config where memory_engine uses it, injecting a tiny
            # max_context_tokens.  Everything else delegates to the real config.
            real_config = memory._get_raw_config() if hasattr(memory, "_get_raw_config") else None
            from hindsight_api.config import get_config as _real_get_config

            class _TinyContextProxy:
                """Forwards all attribute access to the real config proxy except
                reflect_max_context_tokens which is forced to 1."""
                _real = _real_get_config()

                def __getattr__(self, name: str):
                    if name == "reflect_max_context_tokens":
                        return 1
                    return getattr(self._real, name)

            with patch("hindsight_api.engine.memory_engine.get_config", return_value=_TinyContextProxy()):
                result = await memory.reflect_async(
                    bank_id=bank_id,
                    query="Tell me about the people you know.",
                    request_context=request_context,
                )

            assert result.text, "reflect must return a non-empty answer"
            assert result.usage.total_tokens > 0

        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

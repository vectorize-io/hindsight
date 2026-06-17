"""
Tests for the reflect agent with mocked LLM outputs.

These tests verify:
1. Tool name normalization for various LLM output formats
2. Recovery from unknown tool calls
3. Recovery from tool execution errors
4. Wall-clock timeout enforcement
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.reflect.agent import (
    _all_mental_models_are_usable_and_fresh,
    _clean_answer_text,
    _clean_done_answer,
    _count_messages_tokens,
    _is_context_overflow_error,
    _is_done_tool,
    _normalize_tool_name,
    run_reflect_agent,
)
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from tests.llm_judge import assert_meets_criteria


class TestCleanAnswerText:
    """Test cleanup of answer text that includes done() tool call syntax."""

    def test_clean_text_with_done_call(self):
        """Text ending with done() call should have it stripped."""
        text = """The team's OKRs focus on performance.done({"answer":"The team's OKRs","memory_ids":[]})"""
        cleaned = _clean_answer_text(text)
        assert cleaned == "The team's OKRs focus on performance."
        assert "done(" not in cleaned

    def test_clean_text_with_done_call_and_whitespace(self):
        """done() call with whitespace should be stripped."""
        text = """Answer text here. done( {"answer": "short", "memory_ids": []} )"""
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
        text = """Summary of findings.done({
            "answer": "Summary",
            "memory_ids": ["id1", "id2"]
        })"""
        cleaned = _clean_answer_text(text)
        assert cleaned == "Summary of findings."


class TestCleanDoneAnswer:
    """Test cleanup of answer field from done() tool call that leaks structured output."""

    def test_clean_answer_with_leaked_json_code_block(self):
        """Answer with leaked JSON code block at the end should be cleaned."""
        text = """The user's favorite color is blue.

```json
{"observation_ids": ["obs-1", "obs-2"]}
```"""
        cleaned = _clean_done_answer(text)
        assert cleaned == "The user's favorite color is blue."
        assert "observation_ids" not in cleaned

    def test_clean_answer_with_memory_ids_code_block(self):
        """Answer with leaked memory_ids JSON code block should be cleaned."""
        text = """Here is the answer.

```json
{"memory_ids": ["mem-1"]}
```"""
        cleaned = _clean_done_answer(text)
        assert cleaned == "Here is the answer."

    def test_clean_answer_with_raw_json_object(self):
        """Answer with raw JSON object containing IDs at the end should be cleaned."""
        text = 'The answer is 42. {"observation_ids": ["obs-1"]}'
        cleaned = _clean_done_answer(text)
        assert cleaned == "The answer is 42."

    def test_clean_answer_with_trailing_ids_pattern(self):
        """Answer with 'observation_ids: [...]' pattern at the end should be cleaned."""
        text = 'This is the answer.\n\nobservation_ids: ["obs-1", "obs-2"]'
        cleaned = _clean_done_answer(text)
        assert cleaned == "This is the answer."

    def test_clean_answer_with_memory_ids_equals(self):
        """Answer with 'memory_ids = [...]' pattern at the end should be cleaned."""
        text = 'Answer text here.\nmemory_ids = ["mem-1"]'
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
        text = """Summary:
- Point 1
- Point 2

```json
{"mental_model_ids": ["mm-1"]}
```"""
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


class TestMentalModelFreshnessHelper:
    """Deterministic freshness/usability guard for short-circuiting forced retrieval."""

    def test_all_fresh_and_non_empty_is_usable(self):
        output = {
            "mental_models": [
                {"id": "mm-1", "content": "Fresh content.", "is_stale": False},
                {"id": "mm-2", "content": "More fresh content.", "is_stale": False},
            ]
        }
        assert _all_mental_models_are_usable_and_fresh(output) is True

    def test_any_stale_model_is_not_usable(self):
        output = {
            "mental_models": [
                {"id": "mm-1", "content": "Fresh content.", "is_stale": False},
                {"id": "mm-2", "content": "Old content.", "is_stale": True},
            ]
        }
        assert _all_mental_models_are_usable_and_fresh(output) is False

    def test_missing_staleness_flag_is_not_usable(self):
        # An unknown/missing staleness flag must be treated as unsafe.
        output = {"mental_models": [{"id": "mm-1", "content": "Fresh content."}]}
        assert _all_mental_models_are_usable_and_fresh(output) is False

    def test_blank_content_is_not_usable(self):
        output = {"mental_models": [{"id": "mm-1", "content": "   ", "is_stale": False}]}
        assert _all_mental_models_are_usable_and_fresh(output) is False

    def test_empty_list_is_vacuously_usable(self):
        # The caller gates on a non-empty list separately; the helper itself is
        # only responsible for freshness/content of the models it is given.
        assert _all_mental_models_are_usable_and_fresh({"mental_models": []}) is True
        assert _all_mental_models_are_usable_and_fresh({}) is True


class TestReflectAgentMocked:
    """Test reflect agent with mocked LLM outputs."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        llm = MagicMock()
        llm.call_with_tools = AsyncMock()
        # Also mock call() for final iteration fallback - returns (response, usage) tuple
        llm.call = AsyncMock(
            return_value=(
                "Fallback answer from final iteration",
                TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            )
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

    @staticmethod
    def _mm_call(call_id: str = "1", query: str = "test query") -> LLMToolCallResult:
        return LLMToolCallResult(
            tool_calls=[
                LLMToolCall(id=call_id, name="search_mental_models", arguments={"reason": "curated", "query": query})
            ],
            finish_reason="tool_calls",
        )

    @pytest.mark.asyncio
    async def test_fresh_mental_model_releases_forced_retrieval(self, mock_llm, mock_functions):
        """A fresh, usable mental model stops forced lower-level retrieval — with no extra LLM call.

        The agent answers on the very next (auto) iteration, so search_observations
        and recall are never invoked.
        """
        mock_functions["search_mental_models_fn"].return_value = {
            "query": "test query",
            "mental_models": [
                {"id": "mm-1", "name": "User prefs", "content": "The user prefers concise answers.", "is_stale": False}
            ],
        }
        mock_llm.call_with_tools.side_effect = [
            self._mm_call(),
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="2", name="done", arguments={"answer": "Be concise.", "mental_model_ids": ["mm-1"]})
                ],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            has_mental_models=True,
            budget="low",
            max_iterations=5,
            **mock_functions,
        )

        assert result.text == "Be concise."
        # The fix's whole point: no extra LLM round-trip to decide sufficiency.
        mock_llm.call.assert_not_called()
        mock_functions["search_observations_fn"].assert_not_called()
        mock_functions["recall_fn"].assert_not_called()
        # First iteration forced mental models; second was released to auto.
        first_choice = mock_llm.call_with_tools.await_args_list[0].kwargs["tool_choice"]
        assert first_choice == {"type": "function", "function": {"name": "search_mental_models"}}
        assert mock_llm.call_with_tools.await_args_list[1].kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_short_circuited_agent_may_still_retrieve_under_auto(self, mock_llm, mock_functions):
        """After release, the agent can still choose to retrieve deeper itself (its own query)."""
        mock_functions["search_mental_models_fn"].return_value = {
            "query": "test query",
            "mental_models": [
                {"id": "mm-1", "name": "Status", "content": "Launch was planned for Friday.", "is_stale": False}
            ],
        }
        mock_llm.call_with_tools.side_effect = [
            self._mm_call(),
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(
                        id="2", name="recall", arguments={"reason": "verify", "query": "launch completion proof"}
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="3", name="done", arguments={"answer": "Confirmed.", "memory_ids": ["mem-1"]})
                ],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            has_mental_models=True,
            budget="low",
            max_iterations=5,
            **mock_functions,
        )

        assert result.text == "Confirmed."
        # recall ran because the model chose it under auto, not because it was forced,
        # and it used the model's own targeted query (not a forced override).
        assert mock_llm.call_with_tools.await_args_list[1].kwargs["tool_choice"] == "auto"
        mock_functions["recall_fn"].assert_called_once()
        assert mock_functions["recall_fn"].await_args.args[0] == "launch completion proof"
        mock_functions["search_observations_fn"].assert_not_called()

    @pytest.mark.asyncio
    async def test_stale_mental_model_keeps_forced_retrieval(self, mock_llm, mock_functions):
        """A stale mental model must not short-circuit; the full forced path continues."""
        mock_functions["search_mental_models_fn"].return_value = {
            "query": "test query",
            "mental_models": [
                {
                    "id": "mm-1",
                    "name": "Old status",
                    "content": "Old summary.",
                    "is_stale": True,
                    "staleness_reason": "newer facts exist",
                }
            ],
        }
        mock_llm.call_with_tools.side_effect = [
            self._mm_call(),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="2", name="search_observations", arguments={"query": "q"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="3", name="recall", arguments={"query": "q"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="4", name="done", arguments={"answer": "Verified.", "memory_ids": ["mem-1"]})
                ],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            has_mental_models=True,
            budget="low",
            max_iterations=5,
            **mock_functions,
        )

        assert result.text == "Verified."
        mock_functions["search_observations_fn"].assert_called_once()
        mock_functions["recall_fn"].assert_called_once()
        choices = [c.kwargs["tool_choice"] for c in mock_llm.call_with_tools.await_args_list[:3]]
        assert choices == [
            {"type": "function", "function": {"name": "search_mental_models"}},
            {"type": "function", "function": {"name": "search_observations"}},
            {"type": "function", "function": {"name": "recall"}},
        ]

    @pytest.mark.asyncio
    async def test_high_budget_keeps_forced_path_for_fresh_mental_model(self, mock_llm, mock_functions):
        """High budget preserves the full verification path even for fresh mental models."""
        mock_functions["search_mental_models_fn"].return_value = {
            "query": "test query",
            "mental_models": [
                {"id": "mm-1", "name": "Prefs", "content": "Fresh and directly relevant.", "is_stale": False}
            ],
        }
        mock_llm.call_with_tools.side_effect = [
            self._mm_call(),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="2", name="search_observations", arguments={"query": "q"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="3", name="recall", arguments={"query": "q"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="4", name="done", arguments={"answer": "Verified.", "memory_ids": ["mem-1"]})
                ],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            has_mental_models=True,
            budget="high",
            max_iterations=5,
            **mock_functions,
        )

        assert result.text == "Verified."
        mock_functions["search_observations_fn"].assert_called_once()
        mock_functions["recall_fn"].assert_called_once()
        assert mock_llm.call_with_tools.await_args_list[1].kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "search_observations"},
        }

    @pytest.mark.asyncio
    async def test_no_mental_models_keeps_forced_retrieval(self, mock_llm, mock_functions):
        """An empty mental-model result must not short-circuit the forced path."""
        mock_functions["search_mental_models_fn"].return_value = {"query": "test query", "mental_models": []}
        mock_llm.call_with_tools.side_effect = [
            self._mm_call(),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="2", name="search_observations", arguments={"query": "q"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="3", name="recall", arguments={"query": "q"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="4", name="done", arguments={"answer": "Done.", "memory_ids": ["mem-1"]})],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            has_mental_models=True,
            budget="low",
            max_iterations=5,
            **mock_functions,
        )

        assert result.text == "Done."
        mock_functions["search_observations_fn"].assert_called_once()
        mock_functions["recall_fn"].assert_called_once()

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
    async def test_short_circuit_answer_is_capped_by_max_tokens(self, mock_llm, mock_functions):
        """When the LLM short-circuits (returns text without calling a tool) and the text
        exceeds max_tokens, the agent must rewrite it through a capped call so the final
        user-visible answer respects the configured limit.
        """
        # Build a long response that's well over the cap in cl100k_base tokens.
        long_answer = " ".join(
            ["This is a detailed paragraph about the team, their roles, and their recurring meetings."] * 80
        )
        # The short-circuit path: tool_calls empty, content populated.
        mock_llm.call_with_tools.return_value = LLMToolCallResult(
            tool_calls=[],
            content=long_answer,
            finish_reason="stop",
            input_tokens=10,
            output_tokens=500,
        )
        mock_llm.call = AsyncMock(
            return_value=(
                "Short rewritten answer.",
                TokenUsage(input_tokens=50, output_tokens=10, total_tokens=60),
            )
        )

        cap = 50
        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            max_tokens=cap,
            **mock_functions,
        )

        # The rewrite call must have been made, and it must carry the cap.
        assert mock_llm.call.await_count == 1, (
            f"expected exactly one capped rewrite call, got {mock_llm.call.await_count}"
        )
        rewrite_kwargs = mock_llm.call.await_args.kwargs
        assert rewrite_kwargs.get("max_completion_tokens") == cap, (
            f"rewrite call should use max_completion_tokens={cap}, got {rewrite_kwargs.get('max_completion_tokens')}"
        )

        # The final answer is the rewritten text, not the oversized original.
        assert result.text == "Short rewritten answer."

        # The trace records the rewrite step so we can see it was invoked.
        assert any(entry.scope == "final_rewrite" for entry in result.llm_trace), (
            f"llm_trace should include a final_rewrite entry, got {result.llm_trace}"
        )

    @pytest.mark.asyncio
    async def test_short_circuit_answer_under_cap_is_not_rewritten(self, mock_llm, mock_functions):
        """If the short-circuit answer already fits within max_tokens, no extra rewrite
        call should happen — we don't want to pay for a second LLM call in the common case.
        """
        short_answer = "Small answer that already fits."
        mock_llm.call_with_tools.return_value = LLMToolCallResult(
            tool_calls=[],
            content=short_answer,
            finish_reason="stop",
            input_tokens=10,
            output_tokens=8,
        )

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            max_tokens=200,
            **mock_functions,
        )

        assert result.text == short_answer
        mock_llm.call.assert_not_called()

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

    @pytest.mark.asyncio
    async def test_wall_clock_timeout(self, mock_llm: MagicMock, mock_functions: dict[str, AsyncMock]) -> None:
        """Test that asyncio.wait_for can enforce a wall-clock timeout on run_reflect_agent."""

        async def slow_llm_call(*args: object, **kwargs: object) -> LLMToolCallResult:
            await asyncio.sleep(10)  # Simulate a slow LLM call
            return LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="recall", arguments={"query": "test"})],
                finish_reason="tool_calls",
            )

        mock_llm.call_with_tools.side_effect = slow_llm_call

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                run_reflect_agent(
                    llm_config=mock_llm,
                    bank_id="test-bank",
                    query="test query",
                    bank_profile={"name": "Test", "mission": "Testing"},
                    max_iterations=5,
                    **mock_functions,
                ),
                timeout=0.1,  # Very short timeout to trigger quickly
            )


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
                "content": '{"memories": ['
                + ", ".join(
                    [
                        f'{{"id": "m{i}", "content": "A long memory fact about some topic that goes on and on."}}'
                        for i in range(50)
                    ]
                )
                + "]}",
            },
        ]
        small = _count_messages_tokens(small_messages)
        large = _count_messages_tokens(large_messages)
        assert large > small + 200

    def test_is_context_overflow_error_openai(self):
        assert _is_context_overflow_error(Exception("context_length_exceeded: too many tokens"))
        assert _is_context_overflow_error(
            Exception(
                "This model's maximum context length is 128000 tokens. However, your messages resulted in 142164 tokens."
            )
        )

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
            return_value=(
                "Synthesized answer from gathered evidence.",
                TokenUsage(input_tokens=50, output_tokens=20, total_tokens=70),
            )
        )
        return llm

    @pytest.fixture
    def mock_functions_with_large_output(self):
        """Mock functions that return a large enough payload to exceed a tiny token budget."""
        large_memories = [{"id": f"mem-{i}", "content": f"Memory fact number {i}: " + "A" * 200} for i in range(20)]
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
        mock_llm.call_with_tools.side_effect = Exception("context_length_exceeded: messages resulted in 150000 tokens.")

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


class TestDirectiveLeakageOnEmptyBank:
    """Test that directives don't leak into the answer when the bank has no data.

    Uses a real LLM to verify the behaviour end-to-end.
    """

    @pytest.mark.asyncio
    async def test_directive_not_echoed_on_empty_bank(self, memory, request_context):
        """When a bank has a directive but zero memories, reflect must NOT
        parrot the directive text back as its answer.
        """
        import uuid

        directive_text = (
            "When making SEO or content decisions, prefer observed performance data "
            "over industry best practices. Always check the Content Performance page "
            "before recommending a format or approach."
        )

        bank_id = f"test-directive-leak-{uuid.uuid4().hex[:8]}"
        try:
            # Ensure bank exists (auto-creates it), but retain nothing.
            await memory.get_bank_profile(bank_id, request_context=request_context)

            await memory.create_directive(
                bank_id=bank_id,
                name="SEO Directive",
                content=directive_text,
                request_context=request_context,
            )

            result = await memory.reflect_async(
                bank_id=bank_id,
                query="What content strategy should we use?",
                request_context=request_context,
            )

            # The directive content must NOT leak into the answer.
            assert directive_text not in result.text, (
                f"Directive content leaked into the answer verbatim. Got: {result.text!r}"
            )
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.hs_llm_core
class TestContextOverflowIntegration:
    """Integration test: real LLM with a very small max_context_tokens.

    The agent will make one real LLM call (forced tool choice), receive a large
    tool result that exceeds the tiny budget, then synthesize from it via a second
    real LLM call — all without raising a context_length_exceeded error.
    """

    @pytest.fixture
    def memory(self, memory_real_llm):
        """Override to use real LLM for this class."""
        return memory_real_llm

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


@pytest.mark.hs_llm_core
@pytest.mark.flaky(reruns=3, reruns_delay=2)
class TestMentalModelShortCircuitRealLLM:
    """End-to-end, real-LLM coverage for the fresh-mental-model short-circuit.

    The deterministic release-to-auto mechanism is covered by the MockLLM tests
    above. What only a real model can verify is the behaviour *after* release:
    that a real agent, once it is no longer forced, actually answers off a fresh
    sufficient mental model — and, when the model is fresh but incomplete, that
    it chooses to retrieve deeper itself with its own targeted query.

    The search functions are stubbed so the mental-model content is controlled,
    but ``llm_config`` drives the real agent loop.
    """

    @staticmethod
    def _stub_functions(mental_models, recall_memories=None, observations=None):
        async def search_mental_models_fn(query, max_results):
            return {"query": query, "count": len(mental_models), "mental_models": mental_models}

        return {
            "search_mental_models_fn": AsyncMock(side_effect=search_mental_models_fn),
            "search_observations_fn": AsyncMock(return_value={"observations": observations or []}),
            "recall_fn": AsyncMock(return_value={"memories": recall_memories or []}),
            "expand_fn": AsyncMock(return_value={"memories": []}),
        }

    @pytest.mark.asyncio
    async def test_real_fresh_mental_model_answers_without_lower_retrieval(self, llm_config):
        """A fresh, sufficient mental model lets the real agent answer without obs/recall."""
        functions = self._stub_functions(
            mental_models=[
                {
                    "id": "mm-comm",
                    "name": "Architecture Communication Preference",
                    "content": (
                        "For architecture decisions, the team prefers asynchronous written communication. "
                        "They use concise ADRs (Architecture Decision Records) and explicitly avoid settling "
                        "complex design questions in live meetings."
                    ),
                    "relevance": 0.94,
                    "is_stale": False,
                }
            ]
        )

        result = await run_reflect_agent(
            llm_config=llm_config,
            bank_id="test-bank",
            query="According to what you know, how does the team prefer to communicate about architecture decisions?",
            bank_profile={"name": "Test", "mission": "Answer from curated knowledge"},
            has_mental_models=True,
            include_observations=True,
            include_recall=True,
            budget="low",
            max_iterations=6,
            **functions,
        )

        assert result.text, "agent must return a non-empty answer"
        # The core behavioural win: the forced lower-level path was released, and a
        # real model answered off the fresh mental model instead of digging deeper.
        functions["search_observations_fn"].assert_not_called()
        functions["recall_fn"].assert_not_called()
        await assert_meets_criteria(
            response=result.text,
            criteria=(
                "The answer states the team prefers asynchronous written communication for architecture "
                "decisions (e.g. ADRs) rather than live meetings."
            ),
            context="The only knowledge available was a mental model describing the team's async/ADR preference.",
        )

    @pytest.mark.asyncio
    async def test_real_stale_mental_model_forces_and_grounds_in_deeper_evidence(self, llm_config):
        """A stale mental model must NOT short-circuit: the agent is forced deeper and grounds its answer there.

        This is the safety side of the guard. Forcing the lower layers is
        deterministic (stale fails the freshness check), so observations/recall
        run regardless of model discretion; the real-LLM value is confirming the
        agent corrects the stale summary using the freshly retrieved raw fact.
        """
        functions = self._stub_functions(
            mental_models=[
                {
                    "id": "mm-aurora",
                    "name": "Project Aurora Launch Plan",
                    "content": "Project Aurora's launch is still pending and has not happened yet.",
                    "relevance": 0.91,
                    "is_stale": True,
                    "staleness_reason": "newer deployment facts exist",
                }
            ],
            recall_memories=[
                {
                    "id": "mem-deploy",
                    "content": "Project Aurora shipped to production on Friday at 16:00 UTC (deploy log A-1029).",
                }
            ],
            observations=[
                {
                    "id": "obs-deploy",
                    "content": "Aurora production deployment confirmed Friday; see deploy log A-1029.",
                }
            ],
        )

        result = await run_reflect_agent(
            llm_config=llm_config,
            bank_id="test-bank",
            query="What is the current status of Project Aurora, and what concrete fact supports it?",
            bank_profile={"name": "Test", "mission": "Answer with verifiable detail"},
            has_mental_models=True,
            include_observations=True,
            include_recall=True,
            budget="low",
            max_iterations=6,
            **functions,
        )

        assert result.text, "agent must return a non-empty answer"
        # Stale mental model → no short-circuit → lower layers are still forced.
        assert functions["search_observations_fn"].await_count > 0
        assert functions["recall_fn"].await_count > 0
        await assert_meets_criteria(
            response=result.text,
            criteria=(
                "The answer states Project Aurora has launched/shipped to production (NOT that it is still "
                "pending) and cites the Friday production deployment (e.g. deploy log A-1029) as support."
            ),
            context="A stale mental model claimed the launch was still pending, but the freshly retrieved raw "
            "fact (deploy log A-1029) shows it shipped on Friday. The agent should correct the stale summary.",
        )

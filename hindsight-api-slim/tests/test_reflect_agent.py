"""
Tests for the reflect agent with mocked LLM outputs.

These tests verify:
1. Tool name normalization for various LLM output formats
2. Recovery from unknown tool calls
3. Recovery from tool execution errors
4. Wall-clock timeout enforcement
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.cancellation import OperationCancelledError
from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_AUTO, LLMToolChoice
from hindsight_api.engine.reflect.agent import (
    ReflectNoAnswerError,
    ReflectToolCallError,
    ReflectToolExecutionError,
    _all_mental_models_are_usable_and_fresh,
    _cache_cleanup_tasks,
    _count_messages_tokens,
    _generate_structured_output,
    _is_context_overflow_error,
    _is_done_tool,
    _normalize_tool_name,
    run_reflect_agent,
)
from hindsight_api.engine.reflect.structured_doc import DocumentSectionsInvalidError
from hindsight_api.engine.response_models import LLMCallResult, LLMToolCall, LLMToolCallResult, TokenUsage
from tests.llm_judge import assert_meets_criteria


@pytest.fixture
def mock_functions():
    """The agent's tool callbacks, all stubbed.

    Module-level and shared by every class in this file on purpose. These are
    REQUIRED keyword arguments of run_reflect_agent, so a per-class copy that
    misses one is invisible until that class runs: three copies had drifted when
    read_mental_models_fn was added, and the class whose copy went unupdated died
    on a TypeError in every one of its tests (#4774). One definition, so a new
    callback is added once and reaches every class. A class or test that needs a
    different return value overrides that one key on top of this dict instead of
    writing out the whole set again.
    """
    return {
        "search_mental_models_fn": AsyncMock(return_value={"mental_models": []}),
        "read_mental_models_fn": AsyncMock(return_value={"mental_models": []}),
        "search_observations_fn": AsyncMock(return_value={"observations": []}),
        "recall_fn": AsyncMock(return_value={"memories": [{"id": "mem-1", "content": "test memory"}]}),
        "expand_fn": AsyncMock(return_value={"memories": []}),
    }


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

    def test_a_snippet_counts_as_usable_content(self):
        """A search now returns the runner-up pages as a snippet, not as `content`.

        Reading only `content` made every result but the top hit look empty, which
        kept the forced ladder running past a set of fresh, relevant pages.
        """
        output = {
            "mental_models": [
                {"id": "mm-1", "content": "The best hit, whole.", "is_stale": False},
                {"id": "mm-2", "snippet": "The runner-up, truncated", "content_chars": 900, "is_stale": False},
            ]
        }
        assert _all_mental_models_are_usable_and_fresh(output) is True

    def test_a_blank_snippet_is_not_usable(self):
        output = {"mental_models": [{"id": "mm-1", "snippet": "  ", "is_stale": False}]}
        assert _all_mental_models_are_usable_and_fresh(output) is False

    def test_empty_list_is_vacuously_usable(self):
        # The caller gates on a non-empty list separately; the helper itself is
        # only responsible for freshness/content of the models it is given.
        assert _all_mental_models_are_usable_and_fresh({"mental_models": []}) is True
        assert _all_mental_models_are_usable_and_fresh({}) is True


class TestReflectStructuredOutput:
    """Tests for the second-pass structured-output extraction."""

    @pytest.mark.asyncio
    async def test_structured_output_uses_short_retry_budget(self):
        """A provider-specific structured-output failure must not consume the full reflect timeout."""
        llm = MagicMock()
        llm.call = AsyncMock(side_effect=RuntimeError("empty message content: finish_reason=length"))

        result = await _generate_structured_output(
            answer="Alice prefers concise engineering updates.",
            response_schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                },
                "required": ["summary"],
            },
            llm_config=llm,
            reflect_id="test-reflect",
        )

        assert result.structured_output is None
        assert result.error is not None
        call_kwargs = llm.call.await_args.kwargs
        assert call_kwargs["scope"] == "reflect_structured"
        assert call_kwargs["max_retries"] == 1
        assert call_kwargs["initial_backoff"] == 0.25
        assert call_kwargs["max_backoff"] == 1.0

    @pytest.mark.asyncio
    async def test_structured_output_forwards_max_tokens(self):
        """Structured extraction must receive the reflect output-token budget so
        reasoning / preamble models do not exhaust the provider default before
        emitting JSON (finish_reason=length, empty content -> issue #2431). The
        plain reflect calls already pass max_completion_tokens=max_tokens; the
        structured second pass must too."""
        llm = MagicMock()
        llm.call = AsyncMock(side_effect=RuntimeError("empty message content: finish_reason=length"))

        await _generate_structured_output(
            answer="Alice prefers concise engineering updates.",
            response_schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
            llm_config=llm,
            reflect_id="test-reflect",
            max_tokens=4096,
        )

        call_kwargs = llm.call.await_args.kwargs
        assert call_kwargs["max_completion_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_structured_output_uses_deterministic_temperature(self, monkeypatch):
        """Structured extraction is deterministic even when reflect generation is not."""
        config = MagicMock(llm_temperature_reflect=0.17, llm_strict_schema_reflect=False)
        monkeypatch.setattr("hindsight_api.engine.reflect.agent.get_config", lambda: config)
        llm = MagicMock()
        llm.call = AsyncMock(side_effect=RuntimeError("stop after request capture"))

        await _generate_structured_output(
            answer="Alice prefers concise engineering updates.",
            response_schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
            llm_config=llm,
            reflect_id="test-reflect",
        )

        assert llm.call.await_args.kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_structured_output_omits_temperature_when_reflect_omits_it(self, monkeypatch):
        """Schema extraction must not reintroduce a parameter disabled for the model."""
        config = MagicMock(llm_temperature_reflect=None, llm_strict_schema_reflect=False)
        monkeypatch.setattr("hindsight_api.engine.reflect.agent.get_config", lambda: config)
        llm = MagicMock()
        llm.call = AsyncMock(side_effect=RuntimeError("stop after request capture"))

        await _generate_structured_output(
            answer="Alice prefers concise engineering updates.",
            response_schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
            llm_config=llm,
            reflect_id="test-reflect",
        )

        assert llm.call.await_args.kwargs["temperature"] is None

    @pytest.mark.asyncio
    async def test_structured_output_omits_budget_when_unset(self):
        """With no max_tokens (default), the structured call forwards
        max_completion_tokens=None -- which LLMProvider.call omits, exactly like
        the plain reflect calls -- so behavior is unchanged for callers that do
        not request a budget."""
        llm = MagicMock()
        llm.call = AsyncMock(side_effect=RuntimeError("boom"))

        await _generate_structured_output(
            answer="Alice prefers concise engineering updates.",
            response_schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
            llm_config=llm,
            reflect_id="test-reflect",
        )

        assert llm.call.await_args.kwargs.get("max_completion_tokens") is None

    @pytest.mark.asyncio
    async def test_failed_extraction_reports_why(self):
        """A failed extraction carries the reason, so a caller can tell it apart from
        an answer that simply held nothing matching the schema (issue #4230)."""
        llm = MagicMock()
        llm.call = AsyncMock(side_effect=RuntimeError("provider is down"))

        result = await _generate_structured_output(
            answer="Alice prefers concise engineering updates.",
            response_schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
            llm_config=llm,
            reflect_id="test-reflect",
        )

        assert result.structured_output is None
        assert result.error == "RuntimeError: provider is down"

    @pytest.mark.asyncio
    async def test_successful_extraction_reports_no_error(self):
        """The success path leaves ``error`` unset, so its presence alone means failure."""
        llm = MagicMock()
        llm.call = AsyncMock(
            return_value=LLMCallResult(
                content={"summary": "Alice likes short updates."},
                usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        )

        result = await _generate_structured_output(
            answer="Alice prefers concise engineering updates.",
            response_schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
            llm_config=llm,
            reflect_id="test-reflect",
        )

        assert result.structured_output == {"summary": "Alice likes short updates."}
        assert result.error is None


class TestReflectAgentMocked:
    """Test reflect agent with mocked LLM outputs."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        llm = MagicMock()
        llm.call_with_tools = AsyncMock()
        # Also mock call() for the final-iteration fallback.
        llm.call = AsyncMock(
            return_value=LLMCallResult(
                content="Fallback answer from final iteration",
                usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            )
        )
        return llm

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
        assert first_choice == LLMToolChoice.named("search_mental_models")
        assert mock_llm.call_with_tools.await_args_list[1].kwargs["tool_choice"] is LLM_TOOL_CHOICE_AUTO
        tool_result = mock_llm.call_with_tools.await_args_list[1].kwargs["messages"][-1]
        assert tool_result["role"] == "tool"
        assert tool_result["tool_call_id"] == "1"
        assert "name" not in tool_result
        # And the model is told why (#5272): releasing tool_choice alone left a
        # model that kept descending and padded the answer with raw facts.
        assert "call done now" in json.loads(tool_result["content"])["guidance"]

    @pytest.mark.asyncio
    async def test_stale_mental_model_gets_no_answer_now_guidance(self, mock_llm, mock_functions):
        mock_functions["search_mental_models_fn"].return_value = {
            "mental_models": [{"id": "mm-1", "name": "P", "content": "Old.", "is_stale": True}]
        }
        mock_llm.call_with_tools.side_effect = [
            self._mm_call(),
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="2", name="done", arguments={"answer": "Old.", "mental_model_ids": ["mm-1"]})
                ],
                finish_reason="tool_calls",
            ),
        ]

        await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            has_mental_models=True,
            budget="low",
            max_iterations=5,
            **mock_functions,
        )

        tool_result = mock_llm.call_with_tools.await_args_list[1].kwargs["messages"][-1]
        assert "guidance" not in json.loads(tool_result["content"])

    @pytest.mark.asyncio
    async def test_output_language_reaches_the_done_path(self, mock_llm, mock_functions):
        """The tool-calling model — the one that writes done() — sees the configured
        language: its system prompt drops the answer-in-the-question's-language rule and
        the directive closes the user message. Forced synthesis is never reached here, so
        fixing ``build_final_system_prompt`` alone would leave this call untouched (#3776).
        """
        mock_functions["search_mental_models_fn"].return_value = {
            "mental_models": [{"id": "mm-1", "name": "User prefs", "content": "Fresh content.", "is_stale": False}]
        }
        mock_llm.call_with_tools.side_effect = [
            self._mm_call(),
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="2", name="done", arguments={"answer": "Done.", "mental_model_ids": ["mm-1"]})
                ],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="张伟负责什么工作？",
            bank_profile={"name": "Test", "mission": ""},
            has_mental_models=True,
            budget="low",
            max_iterations=5,
            llm_output_language="English",
            **mock_functions,
        )

        assert result.text == "Done."
        mock_llm.call.assert_not_called()  # answered via done(), not forced synthesis
        for call in mock_llm.call_with_tools.await_args_list:
            messages = call.kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert "respond in that SAME language" not in messages[0]["content"]
            assert "Respond exclusively in English" not in messages[0]["content"]
            assert messages[1]["role"] == "user"
            assert messages[1]["content"].startswith("张伟负责什么工作？")
            assert "Respond exclusively in English" in messages[1]["content"]
            # The done tool schema is the last thing the model reads before writing the
            # answer; its own "SAME language" rule has to go too, or it wins from there.
            done_tool = next(t for t in call.kwargs["tools"] if t["function"]["name"] == "done")
            assert "SAME language" not in done_tool["function"]["parameters"]["properties"]["answer"]["description"]
            assert "exclusively in English" in done_tool["function"]["description"]

    @pytest.mark.asyncio
    async def test_done_tool_answer_respects_max_tokens(self, mock_llm, mock_functions, monkeypatch):
        config = MagicMock(
            reflect_prompt_cache_enabled=False,
            reflect_max_completion_tokens=None,
            llm_temperature_reflect=0.17,
        )
        monkeypatch.setattr("hindsight_api.engine.reflect.agent.get_config", lambda: config)
        mock_functions["search_mental_models_fn"].return_value = {
            "mental_models": [{"id": "mm-1", "name": "User prefs", "content": "Fresh content.", "is_stale": False}]
        }
        mock_llm.call_with_tools.side_effect = [
            self._mm_call(),
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(
                        id="2",
                        name="done",
                        arguments={"answer": "important detail " * 100, "mental_model_ids": ["mm-1"]},
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
            has_mental_models=True,
            budget="low",
            max_iterations=5,
            max_tokens=8,
            **mock_functions,
        )

        assert result.text == "Fallback answer from final iteration"
        # The page budget is now enforced via the rewrite PROMPT, not a hard
        # transport cap: max_completion_tokens is left uncapped (None) by default
        # so thinking models don't truncate the rewrite mid-word (#3365), while the
        # target still reaches the model through the prompt.
        assert mock_llm.call.await_args.kwargs["max_completion_tokens"] is None
        assert mock_llm.call.await_args.kwargs["temperature"] == 0.17
        assert all(call.kwargs["temperature"] == 0.17 for call in mock_llm.call_with_tools.await_args_list)
        rewrite_user_msg = mock_llm.call.await_args.kwargs["messages"][1]["content"]
        assert "Target budget: 8 tokens" in rewrite_user_msg
        assert result.usage.total_tokens == 150
        assert result.llm_trace[-1].scope == "final_rewrite"

    @pytest.mark.asyncio
    async def test_forced_synthesis_answer_respects_max_tokens(self, mock_llm, mock_functions, monkeypatch):
        """The rewrite runs on the forced path too, not only after ``done`` (#4156).

        ``max_tokens`` stopped being a transport cap on the forced path in #3389,
        and the rewrite that replaced it lived only in ``_process_done_tool`` --
        which left the forced path enforcing nothing at all, on exactly the
        long-running questions that end there.
        """
        config = MagicMock(
            reflect_prompt_cache_enabled=False,
            reflect_max_completion_tokens=None,
            llm_temperature_reflect=0.17,
        )
        monkeypatch.setattr("hindsight_api.engine.reflect.agent.get_config", lambda: config)
        mock_functions["search_mental_models_fn"].return_value = {
            "mental_models": [{"id": "mm-1", "name": "Prefs", "content": "Fresh content.", "is_stale": False}]
        }
        mock_llm.call_with_tools.side_effect = [
            self._mm_call(),
            # No tool call on turn 1 -> forced final synthesis.
            LLMToolCallResult(tool_calls=[], content="I have enough to answer.", finish_reason="stop"),
        ]
        mock_llm.call = AsyncMock(
            side_effect=[
                # The synthesis honours the prompt directive only as far as it likes.
                LLMCallResult(
                    content="important detail " * 100,
                    usage=TokenUsage(input_tokens=40, output_tokens=12, total_tokens=52),
                ),
                LLMCallResult(
                    content="Short capped answer.",
                    usage=TokenUsage(input_tokens=30, output_tokens=6, total_tokens=36),
                ),
            ]
        )

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            has_mental_models=True,
            budget="low",
            max_tokens=8,
            **mock_functions,
        )

        # The returned answer is the rewritten one, and the rewrite is traced and billed.
        assert result.text == "Short capped answer."
        assert mock_llm.call.await_count == 2
        rewrite_user_msg = mock_llm.call.await_args.kwargs["messages"][1]["content"]
        assert "Target budget: 8 tokens" in rewrite_user_msg
        # Uncapped transport on the rewrite as well -- the budget is a prompt target (#3365).
        assert mock_llm.call.await_args.kwargs["max_completion_tokens"] is None
        assert mock_llm.call.await_args.kwargs["temperature"] == 0.17
        rewrite_trace = result.llm_trace[-1]
        assert rewrite_trace.scope == "final_rewrite"
        assert rewrite_trace.input_tokens == 30
        assert rewrite_trace.output_tokens == 6

    @pytest.mark.asyncio
    async def test_forced_synthesis_skips_rewrite_within_budget(self, mock_llm, mock_functions):
        """An answer already inside the budget must not pay for a second call."""
        mock_functions["search_mental_models_fn"].return_value = {
            "mental_models": [{"id": "mm-1", "name": "Prefs", "content": "Fresh content.", "is_stale": False}]
        }
        mock_llm.call_with_tools.side_effect = [
            self._mm_call(),
            LLMToolCallResult(tool_calls=[], content="I have enough to answer.", finish_reason="stop"),
        ]
        mock_llm.call = AsyncMock(
            return_value=LLMCallResult(
                content="Synthesized final answer.",
                usage=TokenUsage(input_tokens=40, output_tokens=12, total_tokens=52),
            )
        )

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            has_mental_models=True,
            budget="low",
            max_tokens=64,
            **mock_functions,
        )

        assert result.text == "Synthesized final answer."
        assert mock_llm.call.await_count == 1
        assert not any(call.scope == "final_rewrite" for call in result.llm_trace)

    @pytest.mark.asyncio
    async def test_empty_rewrite_keeps_the_original_answer(self, mock_llm, mock_functions):
        """A blank rewrite must not blank the answer.

        The rewrite runs *past* the ReflectNoAnswerError guard, so returning the
        model's empty string would hand back a blank result from a run that had a
        complete synthesis -- the exact failure #2959 made loud. The document
        branch already falls back in _document_from_rewrite; prose must too.
        """
        mock_functions["search_mental_models_fn"].return_value = {
            "mental_models": [{"id": "mm-1", "name": "Prefs", "content": "Fresh content.", "is_stale": False}]
        }
        mock_llm.call_with_tools.side_effect = [
            self._mm_call(),
            LLMToolCallResult(tool_calls=[], content="I have enough to answer.", finish_reason="stop"),
        ]
        long_answer = "important detail " * 100
        mock_llm.call = AsyncMock(
            side_effect=[
                LLMCallResult(
                    content=long_answer,
                    usage=TokenUsage(input_tokens=40, output_tokens=12, total_tokens=52),
                ),
                # The rewrite model returns nothing usable.
                LLMCallResult(
                    content="   ",
                    usage=TokenUsage(input_tokens=30, output_tokens=0, total_tokens=30),
                ),
            ]
        )

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            has_mental_models=True,
            budget="low",
            max_tokens=8,
            **mock_functions,
        )

        # Over budget, but a real answer beats a blank one. (The synthesis call
        # strips its response, so compare against the stripped form.)
        assert result.text == long_answer.strip()
        assert mock_llm.call.await_count == 2

    @pytest.mark.asyncio
    async def test_no_tool_call_ever_raises_tool_call_error(self, mock_llm, mock_functions):
        """A transport that strips tool support (never yields a tool call) fails loudly.

        This is the harmony/gpt-oss-via-Vertex-MaaS case: the model returns free text
        that mimics a done() payload with sibling id fields. We must NOT salvage it as
        the answer -- reflect raises ReflectToolCallError instead.
        """
        mock_llm.provider = "litellm"
        mock_llm.model = "vertex_ai/openai/gpt-oss-120b-maas"
        leaked = '{"answer": "The user has a cat named Luna.", "memory_ids": ["mem-1"], "observation_ids": []}'
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(content=leaked, tool_calls=[], finish_reason="stop"),
        ]

        with pytest.raises(ReflectToolCallError) as exc_info:
            await run_reflect_agent(
                llm_config=mock_llm,
                bank_id="test-bank",
                query="what pets does the user have?",
                bank_profile={"name": "Test", "mission": "Testing"},
                has_mental_models=True,
                budget="low",
                max_iterations=5,
                **mock_functions,
            )

        msg = str(exc_info.value)
        assert "vertex_ai/openai/gpt-oss-120b-maas" in msg
        assert "no usable tool call" in msg
        # The forced-final fallback (mock_llm.call) must NOT have run: we fail fast
        # rather than synthesizing a hollow answer from zero evidence.
        mock_llm.call.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "has_mental_models,include_observations,include_recall,expected_choice",
        [
            (True, True, True, "search_mental_models"),
            (False, True, True, "search_observations"),
            (False, False, True, "recall"),
            (False, False, False, "auto"),
        ],
    )
    @pytest.mark.parametrize(
        "content,finish_reason",
        [("Hello!", "stop"), (None, "length"), ("x" * 600, None)],
        ids=["text", "empty-truncated", "long-unknown"],
    )
    async def test_no_tool_call_reports_request_and_response(
        self,
        mock_llm: MagicMock,
        mock_functions: dict[str, AsyncMock],
        has_mental_models: bool,
        include_observations: bool,
        include_recall: bool,
        expected_choice: str,
        content: str | None,
        finish_reason: str | None,
    ) -> None:
        mock_llm.provider = "openai"
        mock_llm.model = "test-model"
        mock_llm.call_with_tools.return_value = LLMToolCallResult(
            content=content,
            tool_calls=[],
            finish_reason=finish_reason,
        )
        with pytest.raises(ReflectToolCallError) as exc_info:
            await run_reflect_agent(
                llm_config=mock_llm,
                bank_id="test-bank",
                query="hi",
                bank_profile={"name": "Test", "mission": "Testing"},
                has_mental_models=has_mental_models,
                include_observations=include_observations,
                include_recall=include_recall,
                budget="low",
                max_iterations=5,
                **mock_functions,
            )
        message = str(exc_info.value)
        assert "during initial tool selection" in message
        assert f"tool_choice={expected_choice!r}" in message
        assert f"finish_reason={finish_reason!r}" in message
        assert "openai/test-model" in message
        if content:
            preview = content[:500] + ("..." if len(content) > 500 else "")
            assert f"Response: {preview!r}" in message
        else:
            assert "returned no content" in message
        mock_llm.call_with_tools.assert_awaited_once()
        mock_llm.call.assert_not_called()
        for function in mock_functions.values():
            function.assert_not_called()

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
        assert mock_llm.call_with_tools.await_args_list[1].kwargs["tool_choice"] is LLM_TOOL_CHOICE_AUTO
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
            LLMToolChoice.named("search_mental_models"),
            LLMToolChoice.named("search_observations"),
            LLMToolChoice.named("recall"),
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
        assert mock_llm.call_with_tools.await_args_list[1].kwargs["tool_choice"] == LLMToolChoice.named(
            "search_observations"
        )

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
    async def test_empty_reply_on_forced_step_retries_that_step(self, mock_llm, mock_functions):
        """A forced step that comes back with no tool call is asked for again, not skipped (#4564)."""
        mock_functions["search_mental_models_fn"].return_value = {"query": "test query", "mental_models": []}
        mock_llm.call_with_tools.side_effect = [
            self._mm_call(),
            # Endpoint ignores the forced ``search_observations`` once.
            LLMToolCallResult(tool_calls=[], finish_reason="tool_calls"),
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
            max_iterations=10,
            **mock_functions,
        )

        assert result.text == "Done."
        choices = [c.kwargs["tool_choice"] for c in mock_llm.call_with_tools.call_args_list]
        assert choices[:4] == [
            LLMToolChoice.named("search_mental_models"),
            LLMToolChoice.named("search_observations"),
            LLMToolChoice.named("search_observations"),
            LLMToolChoice.named("recall"),
        ]
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
    async def test_tool_execution_error_fails_the_run(self, mock_llm, mock_functions):
        """A tool that raises fails the whole run — the loop must not answer around it.

        Reflect used to feed the exception back as a tool result and let the model
        try again. Whatever it answered was then indistinguishable from a run over
        a bank that genuinely holds nothing, and a mental-model refresh wrote that
        answer over a real document (#2894). A raising tool is a broken dependency,
        so the run fails and the caller never reaches its write.
        """
        mock_functions["recall_fn"].side_effect = Exception("Database connection failed")

        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="recall", arguments={"query": "test"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(
                        id="2",
                        name="done",
                        arguments={"answer": "I don't have information about that.", "memory_ids": []},
                    )
                ],
                finish_reason="tool_calls",
            ),
        ]

        with pytest.raises(ReflectToolExecutionError) as exc_info:
            await run_reflect_agent(
                llm_config=mock_llm,
                bank_id="test-bank",
                query="test query",
                bank_profile={"name": "Test", "mission": "Testing"},
                **mock_functions,
            )

        assert "recall" in str(exc_info.value)
        assert "Database connection failed" in str(exc_info.value)
        # The failure is immediate: the model is never asked to carry on without it.
        assert mock_llm.call_with_tools.call_count == 1

    @pytest.mark.asyncio
    async def test_one_failing_tool_fails_a_parallel_batch(self, mock_llm, mock_functions):
        """Even one failure in a parallel batch fails the run, however much the others returned."""
        mock_functions["search_observations_fn"].side_effect = Exception("pgroonga index unavailable")

        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="1", name="recall", arguments={"query": "test"}),
                    LLMToolCall(id="2", name="search_observations", arguments={"query": "test"}),
                ],
                finish_reason="tool_calls",
            ),
        ]

        with pytest.raises(ReflectToolExecutionError) as exc_info:
            await run_reflect_agent(
                llm_config=mock_llm,
                bank_id="test-bank",
                query="test query",
                bank_profile={"name": "Test", "mission": "Testing"},
                **mock_functions,
            )

        assert "search_observations" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tool_argument_errors_are_still_fed_back(self, mock_llm, mock_functions):
        """A malformed call is the model's mistake, not a broken dependency.

        ``_execute_tool`` returns ``{"error": ...}`` for a missing argument or a
        hallucinated tool name. Those are fixable by calling again with different
        arguments, so they keep going back to the model — only tools that *raise*
        fail the run.
        """
        mock_llm.call_with_tools.side_effect = [
            # No query argument: the dispatcher answers with an error dict.
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="recall", arguments={})],
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
                        arguments={"answer": "Recovered from a bad call", "memory_ids": ["mem-1"]},
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

        assert result.text == "Recovered from a bad call"
        assert mock_llm.call_with_tools.call_count == 3

    @pytest.mark.asyncio
    async def test_persistent_llm_error_fails_instead_of_synthesizing(self, mock_llm, mock_functions):
        """A provider that keeps failing fails the run, rather than answering around it.

        The loop used to fall through to a forced final synthesis on any LLM error.
        With retrieval never completed, that synthesis had nothing to work from and
        produced a confident "no information" answer that callers stored (#2894).
        The provider's own 429/5xx retries already ran inside the call, so reaching
        here twice means the failure is not transient.
        """
        mock_llm.call_with_tools.side_effect = RuntimeError("provider unavailable")

        with pytest.raises(RuntimeError, match="provider unavailable"):
            await run_reflect_agent(
                llm_config=mock_llm,
                bank_id="test-bank",
                query="test query",
                bank_profile={"name": "Test", "mission": "Testing"},
                **mock_functions,
            )

        # Retried once (capped at 2 consecutive errors), then gave up.
        assert mock_llm.call_with_tools.call_count == 2
        mock_llm.call.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_transient_llm_error_is_still_retried(self, mock_llm, mock_functions):
        """One failed turn followed by a good one is a normal run, not a failure."""
        mock_llm.call_with_tools.side_effect = [
            RuntimeError("connection reset"),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="recall", arguments={"query": "test"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(
                        id="2",
                        name="done",
                        arguments={"answer": "Recovered after a blip", "memory_ids": ["mem-1"]},
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

        assert result.text == "Recovered after a blip"
        # The retry asks for the step that failed, not the next one (#4564).
        choices = [c.kwargs["tool_choice"] for c in mock_llm.call_with_tools.call_args_list]
        assert choices[0] == choices[1] == LLMToolChoice.named("search_observations")

    @pytest.mark.asyncio
    async def test_cancellation_from_a_tool_is_not_wrapped(self, mock_llm, mock_functions):
        """A client disconnect must stay a cancellation, not become a tool failure.

        ``recall`` re-raises ``OperationCancelledError`` so the HTTP layer can
        answer 499 (issue #2122). Wrapping it in ``ReflectToolExecutionError``
        would report a client that went away as a server-side retrieval failure.
        """
        mock_functions["recall_fn"].side_effect = OperationCancelledError("client disconnected")

        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="recall", arguments={"query": "test"})],
                finish_reason="tool_calls",
            ),
        ]

        with pytest.raises(OperationCancelledError):
            await run_reflect_agent(
                llm_config=mock_llm,
                bank_id="test-bank",
                query="test query",
                bank_profile={"name": "Test", "mission": "Testing"},
                **mock_functions,
            )

    @pytest.mark.asyncio
    async def test_cancellation_from_the_llm_call_is_not_retried(self, mock_llm, mock_functions):
        """A cancellation raised by the LLM call is not a provider failure.

        It must not be retried and must not be synthesized around — it reaches the
        HTTP layer as itself, so a client disconnect stays a 499 (issue #2122).
        """
        mock_llm.call_with_tools.side_effect = OperationCancelledError("client disconnected")

        with pytest.raises(OperationCancelledError):
            await run_reflect_agent(
                llm_config=mock_llm,
                bank_id="test-bank",
                query="test query",
                bank_profile={"name": "Test", "mission": "Testing"},
                **mock_functions,
            )

        assert mock_llm.call_with_tools.call_count == 1
        mock_llm.call.assert_not_awaited()

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
    async def test_stop_after_evidence_uses_forced_final_synthesis(self, mock_llm, mock_functions):
        """A model that tool-called at least once and then stops (no tool call) is a
        legitimate completion: reflect does a clean forced final-synthesis call (tools
        disabled) rather than salvaging free text or raising ReflectToolCallError.
        """
        mock_functions["search_mental_models_fn"].return_value = {
            "mental_models": [{"id": "mm-1", "name": "Prefs", "content": "Fresh content.", "is_stale": False}]
        }
        mock_llm.call_with_tools.side_effect = [
            # Turn 0: a real tool call -> saw_tool_call becomes True.
            self._mm_call(),
            # Turn 1: model stops with plain text and no tool call.
            LLMToolCallResult(tool_calls=[], content="I have enough to answer.", finish_reason="stop"),
        ]
        mock_llm.call = AsyncMock(
            return_value=LLMCallResult(
                content="Synthesized final answer.",
                usage=TokenUsage(input_tokens=40, output_tokens=12, total_tokens=52),
            )
        )

        cap = 64
        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            has_mental_models=True,
            budget="low",
            max_tokens=cap,
            **mock_functions,
        )

        # Answer comes from the clean forced-final call, not the turn-1 free text.
        assert result.text == "Synthesized final answer."
        assert mock_llm.call.await_count == 1
        # The forced-final synthesis no longer hard-caps the transport at the page
        # budget (that truncates thinking models mid-word, #3365): the call is
        # uncapped by default and the page length reaches the model as a prompt
        # directive instead.
        assert mock_llm.call.await_args.kwargs["max_completion_tokens"] is None
        final_prompt = mock_llm.call.await_args.kwargs["messages"][1]["content"]
        assert f"approximately {cap} tokens" in final_prompt

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

    @pytest.mark.asyncio
    async def test_tool_results_follow_the_models_tool_call_order(
        self, mock_llm: MagicMock, mock_functions: dict[str, AsyncMock]
    ) -> None:
        """A mixed [allowed, hallucinated, allowed] batch serializes in the model's order.

        Anthropic requires the tool_result blocks to line up with the assistant
        tool_use blocks. Rejections for hallucinated tools used to be written
        before the executed tools' results, so a hallucinated call in the middle
        of a batch produced an out-of-order history the API rejects.
        """
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="a", name="recall", arguments={"query": "q1"}),
                    LLMToolCall(id="b", name="totally_made_up", arguments={}),
                    LLMToolCall(id="c", name="search_observations", arguments={"query": "q2"}),
                ],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="d", name="done", arguments={"answer": "A", "memory_ids": ["mem-1"]})],
                finish_reason="tool_calls",
            ),
        ]

        await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            **mock_functions,
        )

        messages = mock_llm.call_with_tools.call_args_list[-1].kwargs["messages"]
        assistant = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")][-1]
        assert [tc["id"] for tc in assistant["tool_calls"]] == ["a", "b", "c"]
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_messages] == ["a", "b", "c"]
        # Each slot carries its OWN payload rather than a neighbour's.
        assert "memories" in tool_messages[0]["content"]
        assert "totally_made_up" in tool_messages[1]["content"]
        assert "observations" in tool_messages[2]["content"]

    @pytest.mark.asyncio
    async def test_duplicate_tool_call_ids_keep_their_own_results(
        self, mock_llm: MagicMock, mock_functions: dict[str, AsyncMock]
    ) -> None:
        """Results are slotted by position, so a reused tool_call_id loses nothing.

        Every first-party provider mints unique ids, but a non-conforming
        OpenAI-compatible gateway can repeat one (or send an empty string) across
        a parallel batch. Keying the results by id would drop one tool's evidence
        and duplicate the other's.
        """
        mock_functions["recall_fn"].side_effect = [
            {"memories": [{"id": "mem-1", "content": "first"}]},
            {"memories": [{"id": "mem-2", "content": "second"}]},
        ]
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="dup", name="recall", arguments={"query": "q1"}),
                    LLMToolCall(id="dup", name="recall", arguments={"query": "q2"}),
                ],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="d", name="done", arguments={"answer": "A", "memory_ids": ["mem-1"]})],
                finish_reason="tool_calls",
            ),
        ]

        await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            **mock_functions,
        )

        messages = mock_llm.call_with_tools.call_args_list[-1].kwargs["messages"]
        contents = [m["content"] for m in messages if m.get("role") == "tool"]
        assert len(contents) == 2
        assert "first" in contents[0], contents
        assert "second" in contents[1], contents

    @pytest.mark.asyncio
    async def test_duplicate_tool_call_ids_are_uniquified_on_the_wire(
        self, mock_llm: MagicMock, mock_functions: dict[str, AsyncMock]
    ) -> None:
        """Repeated or empty ids get unique wire ids so a strict Anthropic API accepts the turn.

        Anthropic rejects a turn where two tool_use blocks share an id
        ("each tool_use must have a single result"), which used to make the whole
        reflect loop fail on the first tool round-trip against a strict endpoint.
        """
        mock_functions["recall_fn"].side_effect = [
            {"memories": [{"id": "mem-1", "content": "first"}]},
            {"memories": [{"id": "mem-2", "content": "second"}]},
            {"memories": [{"id": "mem-3", "content": "third"}]},
        ]
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="dup", name="recall", arguments={"query": "q1"}),
                    LLMToolCall(id="dup", name="recall", arguments={"query": "q2"}),
                    LLMToolCall(id="", name="recall", arguments={"query": "q3"}),
                ],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="d", name="done", arguments={"answer": "A", "memory_ids": ["mem-1"]})],
                finish_reason="tool_calls",
            ),
        ]

        await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            **mock_functions,
        )

        messages = mock_llm.call_with_tools.call_args_list[-1].kwargs["messages"]
        assistant = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")][-1]
        tool_use_ids = [tc["id"] for tc in assistant["tool_calls"]]
        assert len(tool_use_ids) == 3
        assert all(tool_use_id for tool_use_id in tool_use_ids), tool_use_ids
        assert len(set(tool_use_ids)) == 3, tool_use_ids
        # Every tool_result maps to exactly one tool_use, in the model's order.
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_messages] == tool_use_ids
        # ...and each slot still carries its OWN payload.
        assert "first" in tool_messages[0]["content"], tool_messages
        assert "second" in tool_messages[1]["content"], tool_messages
        assert "third" in tool_messages[2]["content"], tool_messages

    @pytest.mark.asyncio
    async def test_wire_ids_stay_unique_across_iterations(
        self, mock_llm: MagicMock, mock_functions: dict[str, AsyncMock]
    ) -> None:
        """A gateway that blanks ids blanks them EVERY turn; the replacements must still differ.

        The whole reflect loop serializes into one request, so deduping per batch
        would mint the same replacement id on turn 1 and turn 2 and hand the
        strict API back the collision it was supposed to remove.
        """
        mock_functions["recall_fn"].side_effect = [
            {"memories": [{"id": "mem-1", "content": "first"}]},
            {"memories": [{"id": "mem-2", "content": "second"}]},
        ]
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="", name="recall", arguments={"query": "q1"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="", name="recall", arguments={"query": "q2"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="d", name="done", arguments={"answer": "A", "memory_ids": ["mem-1"]})],
                finish_reason="tool_calls",
            ),
        ]

        await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            **mock_functions,
        )

        messages = mock_llm.call_with_tools.call_args_list[-1].kwargs["messages"]
        tool_use_ids = [
            tc["id"] for m in messages if m.get("role") == "assistant" and m.get("tool_calls") for tc in m["tool_calls"]
        ]
        assert len(tool_use_ids) == 2, tool_use_ids
        assert len(set(tool_use_ids)) == 2, tool_use_ids
        # Each tool_use is still answered by exactly one tool_result.
        assert [m["tool_call_id"] for m in messages if m.get("role") == "tool"] == tool_use_ids

    @pytest.mark.asyncio
    async def test_premature_done_guardrail_emits_a_usable_wire_id(
        self, mock_llm: MagicMock, mock_functions: dict[str, AsyncMock]
    ) -> None:
        """The "search first" guardrail loops, so its tool_use needs a deduped id too.

        An empty id reaches Anthropic verbatim as ``tool_use.id: ""`` and is
        rejected, taking the whole reflect run with it.
        """
        mock_functions["recall_fn"].side_effect = [{"memories": [{"id": "mem-1", "content": "first"}]}]
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="", name="done", arguments={"answer": "premature"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="", name="recall", arguments={"query": "q"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="d", name="done", arguments={"answer": "A", "memory_ids": ["mem-1"]})],
                finish_reason="tool_calls",
            ),
        ]

        await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            **mock_functions,
        )

        messages = mock_llm.call_with_tools.call_args_list[-1].kwargs["messages"]
        tool_use_ids = [
            tc["id"] for m in messages if m.get("role") == "assistant" and m.get("tool_calls") for tc in m["tool_calls"]
        ]
        assert len(tool_use_ids) == 2, tool_use_ids
        assert all(tool_use_id for tool_use_id in tool_use_ids), tool_use_ids
        assert len(set(tool_use_ids)) == 2, tool_use_ids
        assert [m["tool_call_id"] for m in messages if m.get("role") == "tool"] == tool_use_ids


class TestMalformedDoneDocumentIsFedBack:
    """A ``document`` whose shape the schema refuses is re-asked, never guessed (#4910).

    ``str()`` of a block object is its Python repr, so a block emitted as
    ``{"text": ...}`` used to be stored as the literal ``{'text': '...'}`` and
    rendered into ``mental_models.content``. The payload is refused instead, and
    the per-field errors go back as the done call's tool result so the model can
    re-emit the same content in the declared shape.
    """

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.call_with_tools = AsyncMock()
        llm.call = AsyncMock(
            return_value=LLMCallResult(
                content="Fallback answer",
                usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        )
        return llm

    @pytest.fixture
    def mock_functions(self, mock_functions):
        """The shared callbacks, with a fresh mental model to find and nothing to recall."""
        return {
            **mock_functions,
            "search_mental_models_fn": AsyncMock(
                return_value={
                    "mental_models": [{"id": "mm-1", "name": "P", "content": "Fresh.", "is_stale": False}],
                }
            ),
            "recall_fn": AsyncMock(return_value={"memories": []}),
        }

    @staticmethod
    def _done(call_id: str, blocks: list) -> LLMToolCallResult:
        return LLMToolCallResult(
            tool_calls=[
                LLMToolCall(
                    id=call_id,
                    name="done",
                    arguments={
                        "document": {"sections": [{"heading": "Ops", "level": 2, "blocks": blocks}]},
                        "mental_model_ids": ["mm-1"],
                    },
                )
            ],
            finish_reason="tool_calls",
        )

    @pytest.mark.asyncio
    async def test_object_blocks_are_rejected_and_the_retry_answers(self, mock_llm, mock_functions):
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="1", name="search_mental_models", arguments={"reason": "curated", "query": "q"})
                ],
                finish_reason="tool_calls",
            ),
            self._done("2", [{"text": "Intro."}]),
            self._done("3", ["Intro."]),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="q",
            bank_profile={"name": "Test", "mission": ""},
            has_mental_models=True,
            budget="low",
            max_iterations=5,
            answer_as_document=True,
            **mock_functions,
        )

        assert result.text == "## Ops\n\nIntro."
        assert "{'text'" not in result.text
        # The rejection went back as the done call's own tool result.
        messages = mock_llm.call_with_tools.await_args_list[2].kwargs["messages"]
        rejection = messages[-1]
        assert rejection["role"] == "tool"
        assert "sections[0].blocks[0]" in rejection["content"]

    @pytest.mark.asyncio
    async def test_a_model_that_keeps_the_bad_shape_fails_rather_than_storing_it(self, mock_llm, mock_functions):
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="1", name="search_mental_models", arguments={"reason": "curated", "query": "q"})
                ],
                finish_reason="tool_calls",
            ),
            self._done("2", [{"text": "Intro."}]),
            self._done("3", [{"text": "Intro."}]),
        ]

        with pytest.raises(DocumentSectionsInvalidError):
            await run_reflect_agent(
                llm_config=mock_llm,
                bank_id="test-bank",
                query="q",
                bank_profile={"name": "Test", "mission": ""},
                has_mental_models=True,
                budget="low",
                max_iterations=5,
                answer_as_document=True,
                **mock_functions,
            )

    @pytest.mark.asyncio
    async def test_the_closing_done_call_is_re_asked_not_answered_as_prose(self, mock_llm, mock_functions):
        """The model stops with prose, the closing done() is malformed, and the re-ask lands.

        The standalone prose synthesis (``llm.call``) must not be reached: it
        would answer by re-reading markdown the model wrote, which is the one
        thing document mode exists to avoid.
        """
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="1", name="search_mental_models", arguments={"reason": "curated", "query": "q"})
                ],
                finish_reason="tool_calls",
            ),
            # Stops with prose -> the closing done() call is asked for.
            LLMToolCallResult(tool_calls=[], content="I have enough to answer.", finish_reason="stop"),
            self._done("2", [{"text": "Intro."}]),
            self._done("3", ["Intro."]),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="q",
            bank_profile={"name": "Test", "mission": ""},
            has_mental_models=True,
            budget="low",
            max_iterations=5,
            answer_as_document=True,
            **mock_functions,
        )

        assert result.text == "## Ops\n\nIntro."
        mock_llm.call.assert_not_called()
        # The rejection was fed back on the same prefix.
        retry_messages = mock_llm.call_with_tools.await_args_list[3].kwargs["messages"]
        assert "sections[0].blocks[0]" in retry_messages[-1]["content"]

    @pytest.mark.asyncio
    async def test_a_closing_call_that_stays_malformed_never_answers_as_prose(self, mock_llm, mock_functions):
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="1", name="search_mental_models", arguments={"reason": "curated", "query": "q"})
                ],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(tool_calls=[], content="I have enough to answer.", finish_reason="stop"),
            self._done("2", [{"text": "Intro."}]),
            self._done("3", [{"text": "Intro."}]),
        ]

        with pytest.raises(DocumentSectionsInvalidError):
            await run_reflect_agent(
                llm_config=mock_llm,
                bank_id="test-bank",
                query="q",
                bank_profile={"name": "Test", "mission": ""},
                has_mental_models=True,
                budget="low",
                max_iterations=5,
                answer_as_document=True,
                **mock_functions,
            )
        mock_llm.call.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_over_budget_document_keeps_its_structure_when_the_trim_is_malformed(
        self, mock_llm, mock_functions
    ):
        """A shortening response that is not a document is re-asked, then dropped.

        The full document stands -- over the length budget is a reported outcome.
        What must not happen is the trim being read back as prose, which is how
        unvalidated model markdown would become the stored document.
        """
        long_blocks = ["important detail " * 40, "second detail " * 40]
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="1", name="search_mental_models", arguments={"reason": "curated", "query": "q"})
                ],
                finish_reason="tool_calls",
            ),
            self._done("2", long_blocks),
        ]
        mock_llm.call = AsyncMock(
            return_value=LLMCallResult(
                # Prose where a document was asked for, every attempt.
                content="## Ops\n\nshortened prose\n",
                usage=TokenUsage(input_tokens=30, output_tokens=10, total_tokens=40),
            )
        )

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="q",
            bank_profile={"name": "Test", "mission": ""},
            has_mental_models=True,
            budget="low",
            max_iterations=5,
            answer_as_document=True,
            max_tokens=32,
            **mock_functions,
        )

        assert "shortened prose" not in result.text
        assert long_blocks[0].strip() in result.text
        # Re-asked once before giving up, and both calls are billed.
        assert mock_llm.call.await_count == 2
        rewrite_trace = [call for call in result.llm_trace if call.scope == "final_rewrite"]
        assert rewrite_trace and rewrite_trace[0].input_tokens == 60

    @pytest.mark.asyncio
    async def test_a_valid_document_trim_is_applied(self, mock_llm, mock_functions):
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[
                    LLMToolCall(id="1", name="search_mental_models", arguments={"reason": "curated", "query": "q"})
                ],
                finish_reason="tool_calls",
            ),
            self._done("2", ["important detail " * 40]),
        ]
        mock_llm.call = AsyncMock(
            return_value=LLMCallResult(
                content='{"sections": [{"heading": "Ops", "level": 2, "blocks": ["Short."]}]}',
                usage=TokenUsage(input_tokens=30, output_tokens=10, total_tokens=40),
            )
        )

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="q",
            bank_profile={"name": "Test", "mission": ""},
            has_mental_models=True,
            budget="low",
            max_iterations=5,
            answer_as_document=True,
            max_tokens=32,
            **mock_functions,
        )

        assert result.text == "## Ops\n\nShort."
        assert mock_llm.call.await_count == 1


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
            return_value=LLMCallResult(
                content="Synthesized answer from gathered evidence.",
                usage=TokenUsage(input_tokens=50, output_tokens=20, total_tokens=70),
            )
        )
        return llm

    @pytest.fixture
    def mock_functions_with_large_output(self, mock_functions):
        """The shared callbacks, with a recall payload big enough to exceed a tiny token budget."""
        large_memories = [{"id": f"mem-{i}", "content": f"Memory fact number {i}: " + "A" * 200} for i in range(20)]
        return {**mock_functions, "recall_fn": AsyncMock(return_value={"memories": large_memories})}

    @pytest.mark.asyncio
    async def test_proactive_guard_fires_when_budget_exceeded(self, mock_llm, mock_functions_with_large_output):
        """When token count exceeds max_context_tokens after a tool call, the agent
        should immediately synthesize from gathered evidence instead of making
        another LLM call that would overflow. Evidence beyond the prompt budget
        is split-synthesized (parallel claim extraction + reduce), never dropped."""
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
        # The synthesis ran: at least one no-tools call, ending with the final
        # (single-shot or reduce) call. Whether the history split depends on its
        # rendered size vs the floored chunk budget — both shapes are valid here.
        assert mock_llm.call.call_count >= 1
        scopes = [c.scope for c in result.llm_trace]
        assert scopes[-1] == "final"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("max_iterations", [3, 10], ids=["recall-on-last-turn", "turns-to-spare"])
    async def test_big_observations_do_not_skip_forced_recall(self, mock_llm, mock_functions, max_iterations):
        """#4563: observations alone fill the budget before the forced recall turn.
        The agent blanks the earlier result so recall still runs, then answers from
        the full evidence -- both the observations and the recalled facts -- whether
        the loop ends on its iteration limit or stops early."""
        observations = [{"id": f"obs-{i}", "text": f"Observation {i}: " + "O" * 200} for i in range(20)]
        functions = {
            **mock_functions,
            "search_observations_fn": AsyncMock(return_value={"observations": observations}),
            "recall_fn": AsyncMock(return_value={"memories": [{"id": "mem-1", "content": "Newest correction"}]}),
        }
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="search_observations", arguments={"query": "q"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="2", name="recall", arguments={"query": "q"})],
                finish_reason="tool_calls",
            ),
        ]

        result = await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="What do you know?",
            bank_profile={"name": "Test", "mission": "Testing"},
            include_observations=True,
            max_iterations=max_iterations,
            # ~3.5k tokens with the observations, ~2.1k once they are blanked.
            max_context_tokens=2800,
            **functions,
        )

        assert result.text == "Synthesized answer from gathered evidence."
        functions["recall_fn"].assert_awaited_once()
        assert mock_llm.call_with_tools.call_count == 2
        recall_turn = mock_llm.call_with_tools.call_args_list[1].kwargs
        assert recall_turn["tool_choice"].selected_function_name == "recall"
        assert "Observation 0" not in str(recall_turn["messages"])
        synthesis_prompts = str(mock_llm.call.call_args_list)
        assert "Observation 0" in synthesis_prompts
        assert "Newest correction" in synthesis_prompts

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


class TestNoAnswerFailsHard:
    """A run that produces no answer raises instead of inventing placeholder text (#2959).

    Reflect used to substitute "No answer provided." (or an iteration-limit
    sentence) whenever a terminal path came up empty. Both are non-empty strings,
    so every downstream emptiness guard read them as a real answer: a mental-model
    refresh persisted the stub over a document built across months of refreshes and
    recorded the operation as ``completed`` with no error. There is no answer to
    return in these cases, so the run fails.
    """

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.provider = "openai"
        llm.model = "gpt-test"
        llm.call_with_tools = AsyncMock()
        llm.call = AsyncMock(
            return_value=LLMCallResult(
                content="Synthesized answer from gathered evidence.",
                usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        )
        return llm

    @staticmethod
    def _recall_then(done_arguments: dict) -> list[LLMToolCallResult]:
        """Gather evidence first, so ``done`` is not rejected by the evidence guardrail."""
        return [
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="recall", arguments={"query": "test query"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="2", name="done", arguments=done_arguments)],
                finish_reason="tool_calls",
            ),
        ]

    async def _run(self, mock_llm, mock_functions):
        return await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            has_mental_models=False,
            budget="low",
            max_iterations=5,
            **mock_functions,
        )

    @pytest.mark.asyncio
    async def test_blank_done_answer_raises(self, mock_llm, mock_functions):
        """The production trigger: output truncated mid-tool-call, so ``answer`` arrives empty."""
        mock_llm.call_with_tools.side_effect = self._recall_then({"answer": "   ", "memory_ids": ["mem-1"]})

        with pytest.raises(ReflectNoAnswerError) as exc_info:
            await self._run(mock_llm, mock_functions)

        assert "no answer" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_done_answer_raises(self, mock_llm, mock_functions):
        """``answer`` absent entirely, not just blank."""
        mock_llm.call_with_tools.side_effect = self._recall_then({"memory_ids": ["mem-1"]})

        with pytest.raises(ReflectNoAnswerError):
            await self._run(mock_llm, mock_functions)

    @pytest.mark.asyncio
    async def test_empty_document_mode_answer_raises(self, mock_llm, mock_functions):
        """Document mode that renders to nothing is the same failure as a blank answer."""
        mock_llm.call_with_tools.side_effect = self._recall_then(
            {"document": {"sections": []}, "memory_ids": ["mem-1"]}
        )

        with pytest.raises(ReflectNoAnswerError):
            await self._run(mock_llm, mock_functions)

    @pytest.mark.asyncio
    async def test_empty_final_synthesis_raises(self, mock_llm, mock_functions):
        """The forced final synthesis (tools disabled) returning nothing also fails."""
        mock_llm.call.return_value = LLMCallResult(
            content="   ",
            usage=TokenUsage(input_tokens=10, output_tokens=0, total_tokens=10),
        )
        # Gather evidence, then stop tool-calling: the agent falls through to the
        # forced synthesis, which is where the empty text comes from.
        mock_llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="recall", arguments={"query": "test query"})],
                finish_reason="tool_calls",
            ),
            # Empty while a forced step is still pending: retried once, then given up.
            LLMToolCallResult(content="", tool_calls=[], finish_reason="stop"),
            LLMToolCallResult(content="", tool_calls=[], finish_reason="stop"),
        ]

        with pytest.raises(ReflectNoAnswerError) as exc_info:
            await self._run(mock_llm, mock_functions)

        assert "final synthesis" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_real_answer_still_returned(self, mock_llm, mock_functions):
        """The guard must not touch a run that did produce an answer."""
        mock_llm.call_with_tools.side_effect = self._recall_then(
            {"answer": "The user has a cat named Luna.", "memory_ids": ["mem-1"]}
        )

        result = await self._run(mock_llm, mock_functions)

        assert result.text == "The user has a cat named Luna."


_DOCUMENT = {
    "sections": [
        {
            "heading": "Ops",
            "level": 2,
            "blocks": ["Quarterly planning is owned by the platform team.", "- one\n- two"],
        }
    ]
}


class TestDoneToolStringDocument:
    """A string-encoded ``document`` argument in document mode must render like an object.

    Models on OpenAI-compatible transports sometimes double-encode nested tool
    arguments: the ``document`` arrives as a JSON *string* instead of an object.
    In document mode the done tool has no ``answer`` field, so the dict-only
    branch fell through to ``args.get("answer", "")`` -- always empty -- and
    every well-formed-but-string-encoded completion was a deterministic
    ReflectNoAnswerError misread as truncation. Same transport-artifact class
    PR #899 fixed for MCP tool arguments.
    """

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.provider = "openai"
        llm.model = "gpt-test"
        llm.call_with_tools = AsyncMock()
        llm.call = AsyncMock(
            return_value=LLMCallResult(
                content="Synthesized answer from gathered evidence.",
                usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        )
        return llm

    @staticmethod
    def _recall_then(done_arguments: dict) -> list[LLMToolCallResult]:
        """Gather evidence first, so ``done`` is not rejected by the evidence guardrail."""
        return [
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="recall", arguments={"query": "test query"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="2", name="done", arguments=done_arguments)],
                finish_reason="tool_calls",
            ),
        ]

    async def _run(self, mock_llm, mock_functions):
        return await run_reflect_agent(
            llm_config=mock_llm,
            bank_id="test-bank",
            query="test query",
            bank_profile={"name": "Test", "mission": "Testing"},
            has_mental_models=False,
            budget="low",
            max_iterations=5,
            **mock_functions,
        )

    @pytest.mark.asyncio
    async def test_object_document_renders(self, mock_llm, mock_functions):
        """Object form: the render path runs and returns the document markdown."""
        mock_llm.call_with_tools.side_effect = self._recall_then({"document": _DOCUMENT, "memory_ids": ["mem-1"]})

        result = await self._run(mock_llm, mock_functions)

        assert "## Ops" in result.text
        assert "Quarterly planning is owned by the platform team." in result.text

    @pytest.mark.asyncio
    async def test_string_document_renders_identically(self, mock_llm, mock_functions):
        """The bug: a JSON-string-encoded ``document`` must render exactly like the object form."""
        mock_llm.call_with_tools.side_effect = self._recall_then(
            {"document": json.dumps(_DOCUMENT), "memory_ids": ["mem-1"]}
        )

        result = await self._run(mock_llm, mock_functions)

        mock_llm.call_with_tools.side_effect = self._recall_then({"document": _DOCUMENT, "memory_ids": ["mem-1"]})
        expected = await self._run(mock_llm, mock_functions)
        assert result.text == expected.text

    @pytest.mark.asyncio
    async def test_empty_document_still_raises(self, mock_llm, mock_functions):
        """Document mode that renders to nothing is unchanged: it raises."""
        mock_llm.call_with_tools.side_effect = self._recall_then(
            {"document": {"sections": []}, "memory_ids": ["mem-1"]}
        )

        with pytest.raises(ReflectNoAnswerError):
            await self._run(mock_llm, mock_functions)

    @staticmethod
    def _swapped_closers() -> str:
        """The #5272 Qwen emission: complete, but the final ``"]}]}`` written as ``"}]}]``."""
        good = json.dumps(_DOCUMENT)
        assert good.endswith('"]}]}')
        return good[:-5] + '"}]}]'

    def _done(self, call_id: str, document: str, finish_reason: str | None) -> LLMToolCallResult:
        return LLMToolCallResult(
            tool_calls=[
                LLMToolCall(id=call_id, name="done", arguments={"document": document, "memory_ids": ["mem-1"]})
            ],
            finish_reason=finish_reason,
        )

    @pytest.mark.asyncio
    async def test_swapped_closers_from_a_completed_call_are_repaired(self, mock_llm, mock_functions):
        """#5272: a finished generation with mis-ordered closing brackets renders, no re-ask."""
        mock_llm.call_with_tools.side_effect = [
            self._recall_then({})[0],
            self._done("2", self._swapped_closers(), "tool_calls"),
        ]

        result = await self._run(mock_llm, mock_functions)

        mock_llm.call_with_tools.side_effect = self._recall_then({"document": _DOCUMENT, "memory_ids": ["mem-1"]})
        expected = await self._run(mock_llm, mock_functions)
        assert result.text == expected.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("finish_reason", [None, "length"])
    async def test_swapped_closers_without_a_completion_signal_are_re_asked(
        self, mock_llm, mock_functions, finish_reason
    ):
        """No completed finish_reason, no repair (it could be a cut-off body): re-ask with the parse error."""
        mock_llm.call_with_tools.side_effect = [
            self._recall_then({})[0],
            self._done("2", self._swapped_closers(), finish_reason),
            self._done("3", json.dumps(_DOCUMENT), "tool_calls"),
        ]

        result = await self._run(mock_llm, mock_functions)

        assert "Quarterly planning is owned by the platform team." in result.text
        rejection = mock_llm.call_with_tools.await_args_list[2].kwargs["messages"][-1]
        assert rejection["role"] == "tool"
        assert "not valid JSON" in rejection["content"]

    @staticmethod
    def _raw_with_dropped_bracket() -> str:
        """The #5272 qwen3.8-flash emission: the whole payload unparseable, a block list closed ``"}``."""
        good = json.dumps({"document": _DOCUMENT, "memory_ids": ["mem-1"]})
        assert '"]}' in good
        return good.replace('"]}', '"}', 1)

    def _raw_done(self, call_id: str, finish_reason: str | None) -> LLMToolCallResult:
        return LLMToolCallResult(
            tool_calls=[LLMToolCall(id=call_id, name="done", arguments={"_raw": self._raw_with_dropped_bracket()})],
            finish_reason=finish_reason,
        )

    @pytest.mark.asyncio
    async def test_unparseable_payload_from_a_completed_call_is_repaired(self, mock_llm, mock_functions):
        """#5272: the provider could not parse done's arguments at all; repair them, no re-ask."""
        mock_llm.call_with_tools.side_effect = [self._recall_then({})[0], self._raw_done("2", "tool_calls")]

        result = await self._run(mock_llm, mock_functions)

        assert "Quarterly planning is owned by the platform team." in result.text
        assert mock_llm.call_with_tools.await_count == 2

    @pytest.mark.asyncio
    async def test_unparseable_payload_without_a_completion_signal_is_re_asked(self, mock_llm, mock_functions):
        mock_llm.call_with_tools.side_effect = [
            self._recall_then({})[0],
            self._raw_done("2", None),
            self._done("3", json.dumps(_DOCUMENT), "tool_calls"),
        ]

        result = await self._run(mock_llm, mock_functions)

        assert "Quarterly planning is owned by the platform team." in result.text
        rejection = mock_llm.call_with_tools.await_args_list[2].kwargs["messages"][-1]
        assert "arguments: not valid JSON" in rejection["content"]

    @pytest.mark.asyncio
    async def test_unparseable_string_document_that_is_never_fixed_raises(self, mock_llm, mock_functions):
        """A model that keeps sending garbage still fails the run loudly."""
        mock_llm.call_with_tools.side_effect = [
            self._recall_then({})[0],
            self._done("2", "not json", "tool_calls"),
            self._done("3", "not json", "tool_calls"),
        ]

        with pytest.raises(DocumentSectionsInvalidError, match="not valid JSON"):
            await self._run(mock_llm, mock_functions)


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
            await memory.ensure_bank_profile(bank_id, request_context=request_context)

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
    sufficient mental model — and, when the model is fresh but does not cover the
    question, that it retrieves deeper itself rather than reporting an absence
    (#4567).

    The search functions are stubbed so the mental-model content is controlled,
    but ``llm_config`` drives the real agent loop.
    """

    @staticmethod
    def _stub_functions(base, mental_models, recall_memories=None, observations=None):
        async def search_mental_models_fn(query, max_results):
            return {"query": query, "count": len(mental_models), "mental_models": mental_models}

        return {
            **base,
            "search_mental_models_fn": AsyncMock(side_effect=search_mental_models_fn),
            "search_observations_fn": AsyncMock(return_value={"observations": observations or []}),
            "recall_fn": AsyncMock(return_value={"memories": recall_memories or []}),
        }

    @pytest.mark.asyncio
    async def test_real_fresh_mental_model_answers_without_lower_retrieval(self, llm_config, mock_functions):
        """A fresh, sufficient mental model lets the real agent answer without obs/recall."""
        functions = self._stub_functions(
            mock_functions,
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
            ],
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
    async def test_real_stale_mental_model_forces_and_grounds_in_deeper_evidence(self, llm_config, mock_functions):
        """A stale mental model must NOT short-circuit: the agent is forced deeper and grounds its answer there.

        This is the safety side of the guard. Forcing the lower layers is
        deterministic (stale fails the freshness check), so observations/recall
        run regardless of model discretion; the real-LLM value is confirming the
        agent corrects the stale summary using the freshly retrieved raw fact.
        """
        functions = self._stub_functions(
            mock_functions,
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

    @pytest.mark.asyncio
    async def test_real_fresh_but_uncovering_mental_model_digs_instead_of_denying(self, llm_config, mock_functions):
        """Fresh pages that do not COVER the question must not become a negative answer.

        The regression from #4567: the short-circuit releases the forced lower
        layers whenever the page search comes back fresh and non-empty, and the
        "you MUST call recall() before giving up" rule only fires on a search that
        returns *zero* results. A page that is on the same topic but silent on the
        question is neither, so nothing made the agent look deeper -- it answered
        "the bank holds no decision, record or history about X" while recall() on
        the same bank held the facts. The low/mid depth guidance said as much:
        stop once the pages give "a reasonable answer".

        **Scope, stated honestly.** This does NOT reproduce the reported
        incident, and it is not evidence that the guidance works. Measured on two
        models: on gemini-3.1-flash-lite (what the core-LLM job runs) it passes
        against the pre-change prompt too, 3 runs out of 3; on 2.5-flash-lite it
        fails with the change and without it, across three different wordings of
        the guidance. The incident itself was a capable model (gpt-5.6-luna at
        effort low) over a 546k-char page set — the size and plausibility of the
        page payload is the part this five-line fixture cannot stage.

        So this guards the *contract*: that a page which does not state the answer
        does not end the search. Whether the wording moves a real bank is measured
        on the reporter's banks in #4567, not here.
        """
        functions = self._stub_functions(
            mock_functions,
            mental_models=[
                {
                    "id": "mm-process",
                    "name": "Release Process",
                    "content": (
                        "Issues in the AURORA project are tracked with keys like AURORA-412. Every change ships "
                        "behind a feature flag, is reviewed by two engineers, and is released on Thursdays. This "
                        "page documents the process only and records no status, decision or history for any "
                        "individual issue."
                    ),
                    "relevance": 0.88,
                    "is_stale": False,
                }
            ],
            recall_memories=[
                {
                    "id": "mem-412",
                    "content": (
                        "AURORA-412 was closed as fixed on 12 March; the flag was removed in the same release."
                    ),
                }
            ],
        )

        result = await run_reflect_agent(
            llm_config=llm_config,
            bank_id="test-bank",
            query="Where do we stand on AURORA-412?",
            bank_profile={"name": "Test", "mission": "Answer from what the bank holds"},
            has_mental_models=True,
            include_observations=True,
            include_recall=True,
            budget="low",
            max_iterations=6,
            **functions,
        )

        assert result.text, "agent must return a non-empty answer"
        # The behavioural fix: a page that does not cover the question is not an
        # answer, so the agent goes to the layer that does instead of denying.
        assert functions["recall_fn"].await_count > 0, (
            "the agent answered without calling recall, from a page that holds no status for the issue"
        )
        await assert_meets_criteria(
            response=result.text,
            criteria=(
                "The answer reports that AURORA-412 was closed/fixed (on 12 March, and/or that its feature flag "
                "was removed). It does NOT claim the bank holds nothing, no record, no status or no history for "
                "AURORA-412."
            ),
            context=(
                "The first search returned only a fresh page about the AURORA release process, which records no "
                "status for any individual issue. The raw-fact layer holds that AURORA-412 was closed as fixed "
                "on 12 March."
            ),
        )


class _StepCacheProvider:
    """Fake provider that records the step-by-step incremental-cache protocol.

    Serves as both ``llm_config`` and its own ``_provider_impl``: the reflect loop
    reaches the cache methods via ``llm_config._provider_impl`` and issues LLM
    turns via ``llm_config.call_with_tools``. Every ``call_with_tools`` records the
    ``cached_prefix`` / ``cached_prefix_message_count`` it was handed, so a test
    can assert that each ``auto`` turn reuses exactly the previous turn's full
    input and that the caches are torn down at the end.
    """

    def __init__(self, scripted: list[LLMToolCallResult]):
        self._scripted = scripted
        self._i = 0
        self._provider_impl = self
        self.cache_counter = 0
        self.created: list[tuple[str, int]] = []  # (session_id, #messages covered)
        self.deleted_sessions: list[str] = []
        self.calls: list[dict] = []  # per call_with_tools: tool_choice / cached_prefix / count / #messages

    # -- incremental cache capability --
    def supports_incremental_prompt_cache(self) -> bool:
        return True

    async def create_incremental_cache(self, *, session_id, messages, tools=None):
        self.cache_counter += 1
        self.created.append((session_id, len(messages)))
        return f"cache-{self.cache_counter}"

    async def delete_cached_prefix(self, name):  # pragma: no cover - not exercised here
        pass

    async def delete_cache_session(self, session_id):
        self.deleted_sessions.append(session_id)

    # -- llm surface --
    async def call_with_tools(
        self,
        *,
        messages,
        tools,
        scope="tools",
        tool_choice="auto",
        cached_prefix=None,
        cached_prefix_message_count=0,
        **_,
    ):
        self.calls.append(
            {
                "tool_choice": tool_choice,
                "cached_prefix": cached_prefix,
                "cached_prefix_message_count": cached_prefix_message_count,
                "n_messages": len(messages),
            }
        )
        res = self._scripted[self._i]
        self._i += 1
        return res

    async def call(self, *args, **kwargs):  # final-synthesis fallback (unused on the happy path)
        return LLMCallResult(content="final", usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2))


class TestReflectIncrementalCache:
    """The step-by-step Gemini context cache: each auto turn reuses the previous
    turn's full input, and every per-reflect cache is deleted at the end."""

    @pytest.mark.asyncio
    async def test_each_auto_turn_reuses_previous_step_cache_and_cleans_up(self, mock_functions):
        functions = {
            **mock_functions,
            "search_observations_fn": AsyncMock(return_value={"observations": [{"id": "obs-1"}]}),
            "recall_fn": AsyncMock(return_value={"memories": [{"id": "mem-1", "content": "x"}]}),
        }

        def _tc(cid, name):
            # A real query arg so the stubbed tools return evidence (not an
            # error), which lets the terminal ``done`` call be accepted.
            return LLMToolCallResult(
                tool_calls=[LLMToolCall(id=cid, name=name, arguments={"query": "q"})], finish_reason="tool_calls"
            )

        # Forced obs -> forced recall -> two auto recalls -> done.
        provider = _StepCacheProvider(
            scripted=[
                _tc("0", "search_observations"),
                _tc("1", "recall"),
                _tc("2", "recall"),
                _tc("3", "recall"),
                LLMToolCallResult(
                    tool_calls=[LLMToolCall(id="4", name="done", arguments={"answer": "A", "memory_ids": ["mem-1"]})],
                    finish_reason="tool_calls",
                ),
            ]
        )

        result = await run_reflect_agent(
            llm_config=provider,
            bank_id="cache-bank",
            query="q",
            bank_profile={"name": "T", "mission": "M"},
            has_mental_models=False,
            include_observations=True,
            include_recall=True,
            budget="high",  # keep the full forced path (no early release) so counts are deterministic
            max_iterations=8,
            **functions,
        )
        assert result.text == "A"

        calls = provider.calls
        # 2 forced turns (obs, recall) then 3 auto turns (recall, recall, done).
        assert [c["tool_choice"] for c in calls[:2]] == [
            LLMToolChoice.named("search_observations"),
            LLMToolChoice.named("recall"),
        ]
        assert all(c["tool_choice"] is LLM_TOOL_CHOICE_AUTO for c in calls[2:])

        # Forced turns never reference a cache (Gemini forbids cache + tool_config).
        assert calls[0]["cached_prefix"] is None
        assert calls[1]["cached_prefix"] is None

        # Every auto turn references a cache, and the count it was handed equals the
        # PREVIOUS turn's full input length — i.e. it reuses the previous step entirely.
        for i in range(2, len(calls)):
            assert calls[i]["cached_prefix"] is not None, f"auto call {i} should use the cache"
            assert calls[i]["cached_prefix_message_count"] == calls[i - 1]["n_messages"], (
                f"auto call {i} must cache exactly the previous step's input"
            )

        # Caches are created covering a strictly growing prefix, one per auto turn.
        assert [n for _, n in provider.created] == [
            calls[1]["n_messages"],
            calls[2]["n_messages"],
            calls[3]["n_messages"],
        ]
        assert all(sid.startswith("reflect:") for sid, _ in provider.created)

        # Ephemeral: the session is torn down exactly once when the reflect ends.
        # Teardown is deliberately detached (the caller must not wait on deletes),
        # so drain the background task before asserting it ran.
        await asyncio.gather(*list(_cache_cleanup_tasks), return_exceptions=True)
        assert len(provider.deleted_sessions) == 1
        assert provider.deleted_sessions[0].startswith("reflect:")
        assert provider.deleted_sessions[0] == provider.created[0][0]


class TestReflectShortIdAliases:
    """The model reads short aliases (presentation.py); what it cites must come back as real ids."""

    @pytest.mark.asyncio
    async def test_done_citations_written_as_aliases_resolve_to_real_ids(self, mock_functions):
        functions = {
            **mock_functions,
            "search_observations_fn": AsyncMock(
                return_value={"observations": [{"id": "obs-uuid-1", "text": "an observation"}]}
            ),
            "recall_fn": AsyncMock(return_value={"memories": [{"id": "mem-uuid-1", "text": "a fact"}]}),
        }

        def _tc(cid, name):
            return LLMToolCallResult(
                tool_calls=[LLMToolCall(id=cid, name=name, arguments={"query": "q"})], finish_reason="tool_calls"
            )

        provider = _StepCacheProvider(
            scripted=[
                _tc("0", "search_observations"),
                _tc("1", "recall"),
                LLMToolCallResult(
                    tool_calls=[
                        LLMToolCall(
                            id="2",
                            name="done",
                            arguments={"answer": "A", "memory_ids": ["f1"], "observation_ids": ["o1"]},
                        )
                    ],
                    finish_reason="tool_calls",
                ),
            ]
        )

        result = await run_reflect_agent(
            llm_config=provider,
            bank_id="alias-bank",
            query="q",
            bank_profile={"name": "T", "mission": "M"},
            has_mental_models=False,
            include_observations=True,
            include_recall=True,
            budget="high",
            max_iterations=8,
            **functions,
        )

        assert result.used_memory_ids == ["mem-uuid-1"]
        assert result.used_observation_ids == ["obs-uuid-1"]
        # The raw record stays in the trace; only the prompt carried the alias.
        assert result.tool_trace[1].output["memories"][0]["id"] == "mem-uuid-1"


def _done(arguments: dict) -> LLMToolCallResult:
    return LLMToolCallResult(
        tool_calls=[LLMToolCall(id="2", name="done", arguments=arguments)], finish_reason="tool_calls"
    )


class TestReflectDropsUnseenIds:
    """A UUID in the answer that the model never read is not passed off as a citation (#5166)."""

    REAL = "3f2a9c1e-0b4d-4e6f-8a1b-2c3d4e5f6a7b"
    # Same first eight characters as REAL, different after: the shape the report found.
    FAKE = "3f2a9c1e-9999-4e6f-8a1b-000000000000"
    # A request id quoted inside a memory's text: legitimate, not a citation.
    REQUEST = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"

    @pytest.fixture
    def functions(self):
        return {
            "search_mental_models_fn": AsyncMock(return_value={"mental_models": []}),
            "read_mental_models_fn": AsyncMock(return_value={"mental_models": []}),
            "search_observations_fn": AsyncMock(return_value={"observations": []}),
            "recall_fn": AsyncMock(
                return_value={"memories": [{"id": self.REAL, "text": f"Deploy failed, request {self.REQUEST}."}]}
            ),
            "expand_fn": AsyncMock(return_value={"memories": []}),
        }

    def _llm(self, *after_recall: LLMToolCallResult, rewrite: str | None = None):
        llm = MagicMock()
        recall = LLMToolCallResult(
            tool_calls=[LLMToolCall(id="1", name="recall", arguments={"reason": "r", "query": "deploy"})],
            finish_reason="tool_calls",
        )
        llm.call_with_tools = AsyncMock(side_effect=[recall, *after_recall])
        llm.call = AsyncMock(
            return_value=LLMCallResult(
                content=rewrite or "", usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)
            )
        )
        return llm

    async def _run(self, llm, functions, max_tokens=None):
        return await run_reflect_agent(
            llm_config=llm,
            bank_id="b",
            query="why did the deploy fail?",
            bank_profile={"name": "T", "mission": "M"},
            has_mental_models=False,
            include_recall=True,
            budget="low",
            max_iterations=5,
            max_tokens=max_tokens,
            **functions,
        )

    @pytest.mark.asyncio
    async def test_done_answer_with_made_up_uuid(self, functions):
        llm = self._llm(
            _done({"answer": f"Deploy failed ({self.REQUEST}). Sources: f1, {self.FAKE}", "memory_ids": ["f1"]})
        )

        result = await self._run(llm, functions)

        assert result.text == f"Deploy failed ({self.REQUEST}). Sources: {self.REAL}, [unverified id]"
        assert result.used_memory_ids == [self.REAL]

    @pytest.mark.asyncio
    async def test_length_rewrite_that_garbles_a_document_citation(self, functions, monkeypatch):
        config = MagicMock(
            reflect_prompt_cache_enabled=False, reflect_max_completion_tokens=None, llm_temperature_reflect=0.1
        )
        monkeypatch.setattr("hindsight_api.engine.reflect.agent.get_config", lambda: config)
        document = {"sections": [{"heading": "Deploy", "blocks": ["detail " * 50 + "[f1]"]}]}
        rewrite = json.dumps({"sections": [{"heading": "Deploy", "blocks": [f"It failed [{self.FAKE}]."]}]})
        llm = self._llm(_done({"document": document, "memory_ids": ["f1"]}), rewrite=rewrite)

        result = await self._run(llm, functions, max_tokens=8)

        assert self.FAKE not in result.text
        assert "[[unverified id]]" in result.text
        assert all(self.FAKE not in b.text for s in result.document.sections for b in s.blocks)

    @pytest.mark.asyncio
    async def test_forced_synthesis_with_made_up_uuid(self, functions):
        # The model stops with prose and declines the closing ask for ``done`` too,
        # so the run ends in forced synthesis.
        prose = LLMToolCallResult(tool_calls=[], content="Enough.", finish_reason="stop")
        llm = self._llm(prose, prose)
        llm.call.return_value = LLMCallResult(
            content=f"Failed. Source: {self.REAL}, {self.FAKE}",
            usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        )

        result = await self._run(llm, functions)

        assert result.text == f"Failed. Source: {self.REAL}, [unverified id]"


class TestReflectFinishesThroughDone:
    """A model that stops with prose is asked for ``done`` in the same conversation.

    Measured on the refresh-cost eval: only ~29% of refreshes ever called ``done``
    on their own, so the answer was written twice — once as prose that reflect
    drops, then again from a standalone synthesis prompt that re-renders every
    tool result and re-pays for it. Asking here reuses the prefix the provider
    already holds and returns the structured document a page wants.
    """

    @staticmethod
    def _functions(base):
        return {
            **base,
            "search_observations_fn": AsyncMock(return_value={"observations": [{"id": "obs-1", "text": "o"}]}),
            "recall_fn": AsyncMock(return_value={"memories": [{"id": "mem-1", "text": "m"}]}),
        }

    @staticmethod
    def _searches():
        return [
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id=str(i), name=name, arguments={"query": "q"})], finish_reason="tool_calls"
            )
            for i, name in enumerate(("search_observations", "recall"))
        ]

    @pytest.mark.asyncio
    async def test_prose_stop_is_asked_to_say_it_through_done(self, mock_functions):
        provider = _StepCacheProvider(
            scripted=[
                *self._searches(),
                # The model stops with prose instead of calling done.
                LLMToolCallResult(content="here is the answer", finish_reason="stop"),
                LLMToolCallResult(
                    tool_calls=[LLMToolCall(id="9", name="done", arguments={"answer": "A", "memory_ids": ["f1"]})],
                    finish_reason="tool_calls",
                ),
            ]
        )

        result = await run_reflect_agent(
            llm_config=provider,
            bank_id="b",
            query="q",
            bank_profile={"name": "T", "mission": "M"},
            has_mental_models=False,
            budget="high",
            max_iterations=8,
            **self._functions(mock_functions),
        )

        assert result.text == "A"
        assert result.used_memory_ids == ["mem-1"], "aliases in the closing done call must resolve"
        # The closing call continues the same conversation (one message more than
        # the turn before it) and pins ``done``, rather than rebuilding a synthesis
        # prompt that would re-send every tool result as text.
        closing, previous = provider.calls[-1], provider.calls[-2]
        assert closing["tool_choice"].function_name == "done"
        # Its own trace scope: the standalone synthesis records "final", and the two
        # paths cost differently, so a reader must be able to tell them apart.
        assert [c.scope for c in result.llm_trace][-1] == "closing_done"
        assert not any(c.scope == "final" for c in result.llm_trace), "the standalone prompt must not have run"
        assert closing["n_messages"] == previous["n_messages"] + 1

    @pytest.mark.asyncio
    async def test_a_provider_that_will_not_call_done_falls_back_to_the_standalone_prompt(self, mock_functions):
        provider = _StepCacheProvider(
            scripted=[
                *self._searches(),
                LLMToolCallResult(content="here is the answer", finish_reason="stop"),
                # Asked for done, the provider returns prose again.
                LLMToolCallResult(content="still prose", finish_reason="stop"),
            ]
        )
        provider.call = AsyncMock(
            return_value=LLMCallResult(
                content="synthesized", usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)
            )
        )

        result = await run_reflect_agent(
            llm_config=provider,
            bank_id="b",
            query="q",
            bank_profile={"name": "T", "mission": "M"},
            has_mental_models=False,
            budget="high",
            max_iterations=8,
            **self._functions(mock_functions),
        )

        assert result.text == "synthesized"
        provider.call.assert_awaited()


class TestReflectKeepsItsPrefixStable:
    """Everything reflect re-sends each turn must be byte-identical to the last turn.

    A prompt that only GROWS is what a cached prefix needs — and on hybrid/local
    models (Ollama, llama.cpp), a prompt that diverges early forces a full
    re-prefill of the whole accumulated conversation instead of resuming, which
    is #3865. reflect controls two of the three inputs: the system prompt and the
    tools array. (``tool_choice`` still varies per turn while the retrieval
    sequence is forced — that is the part #4469 tracks, and it is measured but
    not changed here.)
    """

    @pytest.mark.asyncio
    async def test_the_system_prompt_and_tools_never_change_within_one_reflect(self, mock_functions):
        def _tc(cid, name):
            return LLMToolCallResult(
                tool_calls=[LLMToolCall(id=cid, name=name, arguments={"query": "q"})], finish_reason="tool_calls"
            )

        provider = _StepCacheProvider(
            scripted=[
                _tc("0", "search_mental_models"),
                _tc("1", "search_observations"),
                _tc("2", "recall"),
                LLMToolCallResult(
                    tool_calls=[LLMToolCall(id="3", name="done", arguments={"answer": "A"})],
                    finish_reason="tool_calls",
                ),
            ]
        )
        sent: list[tuple[str, str]] = []
        original = provider.call_with_tools

        async def _record(*, messages, tools, **kwargs):
            system = next((m["content"] for m in messages if m.get("role") == "system"), "")
            sent.append((system, json.dumps(tools, sort_keys=True)))
            return await original(messages=messages, tools=tools, **kwargs)

        provider.call_with_tools = _record

        functions = {
            **mock_functions,
            "search_mental_models_fn": AsyncMock(return_value={"mental_models": [{"id": "mm-1", "content": "c"}]}),
            "search_observations_fn": AsyncMock(return_value={"observations": [{"id": "obs-1", "text": "o"}]}),
            "recall_fn": AsyncMock(return_value={"memories": [{"id": "mem-1", "text": "m"}]}),
        }

        await run_reflect_agent(
            llm_config=provider,
            bank_id="b",
            query="q",
            bank_profile={"name": "T", "mission": "M"},
            **functions,
            has_mental_models=True,
            budget="high",
            max_iterations=8,
        )

        assert len(sent) >= 3, f"expected several turns, got {len(sent)}"
        assert len({s for s, _ in sent}) == 1, "the system prompt changed mid-reflect"
        assert len({t for _, t in sent}) == 1, "the tools array changed mid-reflect"

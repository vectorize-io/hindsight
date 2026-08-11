"""Split synthesis: forced final synthesis must not drop retrieved evidence.

When the reflect agent is forced to answer without tools (context guard, last
iteration, LLM error, or a clean stop) and the accumulated tool results exceed
the prompt budget, the old ``build_final_prompt`` dropped any over-budget block
whole — plus every older one — so the synthesis model could see an empty
Retrieved Data section while the response still attached hundreds of citations
(#3122). Split synthesis partitions the history into budget-sized chunks,
compresses each in parallel into dated, cited claims, and synthesizes the
answer from every chunk's claims: nothing retrieved is dropped.

The splitter tests are pure functions; the agent tests drive the map/reduce
flow with a mock LLM.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.reflect.agent import run_reflect_agent
from hindsight_api.engine.reflect.prompts import (
    _MIN_SPLIT_CHUNK_TOKENS,
    _render_history_block,
    build_chunk_claims_prompt,
    build_reduce_prompt,
    split_context_history,
)
from hindsight_api.engine.reflect.tokenization import count_cl100k_tokens
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage

# The splitter floors its per-chunk budget at _MIN_SPLIT_CHUNK_TOKENS, so tests
# use budgets above the floor to exercise the packing logic itself.
_BUDGET_TOKENS = 2048
_MAX_CONTEXT = int(_BUDGET_TOKENS / 0.8)


def _entry(tool: str, key: str, n_items: int, item_chars: int, id_prefix: str = "mem") -> dict:
    items = [{"id": f"{id_prefix}-{i}", "text": f"fact {i}: " + "x" * item_chars} for i in range(n_items)]
    return {"tool": tool, "output": {key: items, "query": "q"}}


def _ids_in(chunks: list[list[dict]]) -> list[str]:
    ids = []
    for chunk in chunks:
        for entry in chunk:
            output = entry["output"]
            for key in ("observations", "memories", "results"):
                for item in output.get(key, []) if isinstance(output, dict) else []:
                    ids.append(item["id"])
    return ids


class TestSplitContextHistory:
    def test_small_history_is_one_chunk(self):
        history = [_entry("recall", "memories", 3, 50)]
        chunks = split_context_history(history, _MAX_CONTEXT)
        assert chunks == [history]

    def test_empty_history_is_no_chunks(self):
        assert split_context_history([], _MAX_CONTEXT) == []

    def test_blocks_pack_greedily_in_order(self):
        """Several fitting blocks distribute across chunks without reordering
        and without losing a single result entry."""
        history = [_entry("recall", "memories", 8, 300, id_prefix=f"b{b}") for b in range(6)]
        chunks = split_context_history(history, _MAX_CONTEXT)

        assert len(chunks) > 1
        for chunk in chunks:
            rendered = "".join(_render_history_block(e) for e in chunk)
            assert count_cl100k_tokens(rendered) <= _BUDGET_TOKENS
        # Every entry survives, in original order.
        original_ids = [item["id"] for e in history for item in e["output"]["memories"]]
        assert _ids_in(chunks) == original_ids

    def test_oversized_block_splits_on_entry_boundaries(self):
        """One block bigger than the whole budget is split into partial blocks —
        the exact case the old code dropped entirely."""
        history = [_entry("search_observations", "observations", 40, 400)]
        chunks = split_context_history(history, _MAX_CONTEXT)

        assert len(chunks) > 1
        for chunk in chunks:
            rendered = "".join(_render_history_block(e) for e in chunk)
            assert count_cl100k_tokens(rendered) <= _BUDGET_TOKENS
            # Partial blocks keep the tool name and sibling keys.
            assert all(e["tool"] == "search_observations" for e in chunk)
            assert all(e["output"]["query"] == "q" for e in chunk)
        assert _ids_in(chunks) == [f"mem-{i}" for i in range(40)]

    def test_indivisible_oversized_entry_is_token_cut(self):
        """A single entry (or list-less output) bigger than the budget is the one
        case that cannot be split; it gets token-cut instead of dropped."""
        giant = {"tool": "expand", "output": {"full_text": "y" * 40_000}}
        chunks = split_context_history([giant], _MAX_CONTEXT)

        assert len(chunks) == 1 and len(chunks[0]) == 1
        cut = chunks[0][0]
        assert cut["output"]["truncated"] is True
        assert cut["output"]["content"]
        assert count_cl100k_tokens(_render_history_block(cut)) <= _BUDGET_TOKENS + 32

    def test_budget_floor_prevents_per_entry_fanout(self):
        """A tiny configured budget must not shred the history into one chunk
        per fact — the floor keeps chunks ~1k tokens."""
        history = [_entry("recall", "memories", 30, 60)]
        chunks = split_context_history(history, max_context_tokens=1)
        total_entries = sum(len(c) for c in chunks)
        assert len(chunks) <= 4
        assert _ids_in(chunks) == [f"mem-{i}" for i in range(30)]
        assert total_entries < 30, "history was shredded into per-entry chunks"


class TestSplitSynthesisPrompts:
    def test_chunk_claims_prompt_carries_evidence_and_question(self):
        chunk = [_entry("recall", "memories", 2, 30)]
        prompt = build_chunk_claims_prompt("what happened?", chunk)
        assert "mem-0" in prompt and "mem-1" in prompt
        assert "## Question\nwhat happened?" in prompt
        assert "Do not answer the question" in prompt

    def test_reduce_prompt_carries_all_sections_and_supersession_rule(self):
        sections = ["- claim A (mentioned_at: 2026-01-01; ...)", "- claim B (mentioned_at: 2026-02-01; ...)"]
        prompt = build_reduce_prompt("what happened?", sections, {"name": "Bank", "mission": "m"})
        assert "### Evidence pass 1:" in prompt and "### Evidence pass 2:" in prompt
        assert "claim A" in prompt and "claim B" in prompt
        assert "LATEST mentioned_at" in prompt
        assert "## Question\nwhat happened?" in prompt


class TestSplitSynthesisAgentFlow:
    """Drives the forced-synthesis path with a mock LLM and asserts the
    map/reduce call pattern."""

    @staticmethod
    def _mock_llm(final_answer: str = "Reduced answer."):
        llm = MagicMock()
        llm.call_with_tools = AsyncMock()

        async def _call(messages, **kwargs):
            # Map calls use the claims system prompt; the reduce/final call uses
            # the final system prompt. Answer accordingly so the test can tell
            # which output made it into the result.
            if "extract evidence" in messages[0]["content"]:
                return (
                    f"- claim from prompt of {count_cl100k_tokens(messages[1]['content'])} tokens "
                    "(mentioned_at: 2026-01-01; occurred: unknown; memory_ids: mem-1)",
                    TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
                )
            return (final_answer, TokenUsage(input_tokens=20, output_tokens=10, total_tokens=30))

        llm.call = AsyncMock(side_effect=_call)
        return llm

    @staticmethod
    def _functions(recall_payload: dict):
        return {
            "search_mental_models_fn": AsyncMock(return_value={"mental_models": []}),
            "search_observations_fn": AsyncMock(return_value={"observations": []}),
            "recall_fn": AsyncMock(return_value=recall_payload),
            "expand_fn": AsyncMock(return_value={"memories": []}),
        }

    @pytest.mark.asyncio
    async def test_overflowing_history_triggers_map_reduce(self):
        """History beyond the budget → one map call per chunk, then one reduce
        call whose output is the final answer."""
        big = {"memories": [{"id": f"mem-{i}", "text": "fact " + "z" * 600} for i in range(60)]}
        llm = self._mock_llm()
        llm.call_with_tools.return_value = LLMToolCallResult(
            tool_calls=[LLMToolCall(id="1", name="recall", arguments={"query": "q"})],
            finish_reason="tool_calls",
        )

        result = await run_reflect_agent(
            llm_config=llm,
            bank_id="b",
            query="what do you know?",
            bank_profile={"name": "Test", "mission": "Testing"},
            max_context_tokens=int(_MIN_SPLIT_CHUNK_TOKENS / 0.8) + 10,
            **self._functions(big),
        )

        scopes = [c.scope for c in result.llm_trace]
        map_scopes = [s for s in scopes if s.startswith("final_map_")]
        assert len(map_scopes) >= 2, f"expected parallel map calls, got scopes {scopes}"
        assert scopes[-1] == "final", "the reduce call must be the last LLM call"
        assert result.text == "Reduced answer."

        # The reduce prompt must carry every map output (one claim per chunk).
        reduce_call = llm.call.await_args_list[-1]
        reduce_prompt = reduce_call.kwargs.get("messages", reduce_call.args[0] if reduce_call.args else None)[1][
            "content"
        ]
        assert reduce_prompt.count("- claim from prompt of") == len(map_scopes)

    @pytest.mark.asyncio
    async def test_caller_max_tokens_is_a_directive_not_a_transport_cap(self):
        """The caller's max_tokens must not cap any synthesis call at the
        transport level (#3365 decoupling): it reaches the reduce prompt as a
        visible-length directive, while every map/reduce call carries the
        config transport cap (None by default)."""
        big = {"memories": [{"id": f"mem-{i}", "text": "fact " + "z" * 600} for i in range(60)]}
        llm = self._mock_llm()
        llm.call_with_tools.return_value = LLMToolCallResult(
            tool_calls=[LLMToolCall(id="1", name="recall", arguments={"query": "q"})],
            finish_reason="tool_calls",
        )

        await run_reflect_agent(
            llm_config=llm,
            bank_id="b",
            query="q?",
            bank_profile={"name": "Test", "mission": "Testing"},
            max_context_tokens=int(_MIN_SPLIT_CHUNK_TOKENS / 0.8) + 10,
            max_tokens=64,
            **self._functions(big),
        )

        caps = [c.kwargs["max_completion_tokens"] for c in llm.call.await_args_list]
        assert all(cap is None for cap in caps), (
            "no synthesis call may inherit the caller's answer budget as a transport cap"
        )
        reduce_prompt = llm.call.await_args_list[-1].kwargs["messages"][1]["content"]
        assert "approximately 64 tokens" in reduce_prompt, "length target must reach the reduce prompt"

    @pytest.mark.asyncio
    async def test_fitting_history_stays_single_call(self):
        """No overflow → exactly the pre-existing single forced-synthesis call."""
        small = {"memories": [{"id": "mem-1", "text": "one small fact"}]}
        llm = self._mock_llm("Direct answer.")
        # First turn recalls; second turn stops with no tool calls → forced synthesis.
        llm.call_with_tools.side_effect = [
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="recall", arguments={"query": "q"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(tool_calls=[], finish_reason="stop", content="done"),
        ]

        result = await run_reflect_agent(
            llm_config=llm,
            bank_id="b",
            query="q?",
            bank_profile={"name": "Test", "mission": "Testing"},
            **self._functions(small),
        )

        assert result.text == "Direct answer."
        scopes = [c.scope for c in result.llm_trace]
        assert scopes == ["agent_1", "agent_2", "final"], f"unexpected scopes {scopes}"

    @pytest.mark.asyncio
    async def test_map_prompts_partition_the_evidence(self):
        """Every retrieved memory id reaches exactly one map prompt — the
        no-evidence-dropped guarantee, asserted end to end."""
        n = 60
        big = {"memories": [{"id": f"mem-{i}", "text": "fact " + "z" * 600} for i in range(n)]}
        llm = self._mock_llm()
        llm.call_with_tools.return_value = LLMToolCallResult(
            tool_calls=[LLMToolCall(id="1", name="recall", arguments={"query": "q"})],
            finish_reason="tool_calls",
        )

        await run_reflect_agent(
            llm_config=llm,
            bank_id="b",
            query="q?",
            bank_profile={"name": "Test", "mission": "Testing"},
            max_context_tokens=int(_MIN_SPLIT_CHUNK_TOKENS / 0.8) + 10,
            **self._functions(big),
        )

        map_prompts = [
            c.kwargs["messages"][1]["content"]
            for c in llm.call.await_args_list
            if "extract evidence" in c.kwargs["messages"][0]["content"]
        ]
        for i in range(n):
            holders = [p for p in map_prompts if f'"mem-{i}"' in p]
            assert len(holders) == 1, f"mem-{i} appears in {len(holders)} map prompts"


@pytest.mark.hs_llm_core
class TestSplitSynthesisRealLLM:
    """Real-LLM, judge-verified coverage of the map/reduce path.

    The mock tests above pin the mechanics (partitioning, call pattern, budget
    caps). What only a real model can verify is that evidence actually SURVIVES
    the pipeline: a distinctive fact placed in the first chunk and another
    placed in the last must both be extractable from the final answer. Under
    the old drop-whole-blocks behavior the answer was a confident "no
    information" — this is the regression #3122 describes, judged rather than
    string-matched because the model paraphrases.
    """

    @pytest.mark.asyncio
    async def test_facts_from_distinct_chunks_reach_the_answer(self, llm_config):
        from tests.llm_judge import assert_meets_criteria

        # ~60 filler memories force several chunks under the floored budget.
        # The two distinctive facts sit at the extremes so they are guaranteed
        # to land in different map calls.
        memories = [{"id": "mem-zara", "text": "Zara keeps bees on her rooftop.", "mentioned_at": "2026-01-05"}]
        memories += [
            {
                "id": f"mem-filler-{i}",
                "text": f"Team member {i} attended the weekly sync and reported routine progress on task {i}. "
                + "Nothing notable happened. " * 8,
                "mentioned_at": "2026-01-10",
            }
            for i in range(60)
        ]
        memories.append(
            {"id": "mem-marco", "text": "Marco collects vintage synthesizers.", "mentioned_at": "2026-02-01"}
        )

        functions = {
            "search_mental_models_fn": AsyncMock(return_value={"mental_models": []}),
            "search_observations_fn": AsyncMock(return_value={"observations": []}),
            "recall_fn": AsyncMock(return_value={"memories": memories}),
            "expand_fn": AsyncMock(return_value={"memories": []}),
        }

        result = await run_reflect_agent(
            llm_config=llm_config,
            bank_id="split-synth-real",
            query="What hobbies do the people in memory have?",
            bank_profile={"name": "Test", "mission": "Remember the team."},
            max_context_tokens=int(_MIN_SPLIT_CHUNK_TOKENS / 0.8) + 10,
            **functions,
        )

        scopes = [c.scope for c in result.llm_trace]
        assert any(s.startswith("final_map_") for s in scopes), f"split synthesis did not engage: {scopes}"

        await assert_meets_criteria(
            response=result.text,
            criteria=(
                "The answer mentions BOTH hobbies: Zara's beekeeping (bees/beehives) AND Marco's "
                "vintage synthesizer collecting. Mentioning only one, or answering that there is "
                "no information, fails."
            ),
            context=(
                "The memory data contained two hobby facts separated by dozens of filler entries: "
                "'Zara keeps bees on her rooftop' and 'Marco collects vintage synthesizers'. The "
                "synthesis pipeline splits the data into chunks, so each fact traversed a different "
                "intermediate extraction call."
            ),
        )

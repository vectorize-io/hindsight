"""Narrator injection and bank-name isolation (issues #1680 and #3962).

The "Narrator: {name}" line in fact extraction is stamped into the who-dimension
of every first-person fact and the observations derived from them. A bank's
``name`` is display/management metadata rather than a speaker identity, so neither
normal retain nor dry-run extraction may inject it automatically.

The prompt assertions use MockLLM only to capture deterministic prompt assembly;
model attribution behaviour is covered by the real-LLM test alongside this file.
"""

import uuid
from datetime import datetime

import pytest

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.retain.fact_extraction import _build_user_message


class TestNarratorInjection:
    def _msg(self, agent_name, context="agent log"):
        return _build_user_message(
            chunk="I shipped the fix.",
            chunk_index=0,
            total_chunks=1,
            event_date=datetime(2024, 6, 1),
            context=context,
            metadata=None,
            agent_name=agent_name,
        )

    def test_no_narrator_line_without_explicit_override(self):
        msg = self._msg(None)
        assert "Narrator:" not in msg

    def test_narrator_line_present_for_named_agent(self):
        msg = self._msg("Aria")
        assert "Narrator: Aria" in msg

    def test_context_precedence_clause_only_when_context_set(self):
        """The 'Context above takes precedence' clause appears only with a context."""
        with_context = self._msg("Aria", context="chat with a customer")
        assert "Context above takes precedence" in with_context

        without_context = self._msg("Aria", context="")
        assert "Narrator: Aria" in without_context  # base narrator still present
        assert "Context above takes precedence" not in without_context


def _fact_extraction_prompts(memory: MemoryEngine) -> list[str]:
    return [
        message["content"]
        for call in memory._retain_llm_config.get_mock_calls()
        if call["scope"] == "retain_extract_facts"
        for message in call["messages"]
        if message["role"] == "user"
    ]


@pytest.mark.asyncio
async def test_bank_display_name_not_injected_by_retain(memory: MemoryEngine):
    bank_id = f"narrator-retain-{uuid.uuid4().hex[:8]}"
    display_name = "ReviewModelEval0825"
    request_context = RequestContext()

    try:
        await memory.update_bank(bank_id, name=display_name, request_context=request_context)
        memory._retain_llm_config.clear_mock_calls()

        await memory.retain_async(
            bank_id,
            "assistant: I scheduled a truck for next Wednesday morning.",
            context="Conversation between a user and a dispatch assistant.",
            request_context=request_context,
        )

        prompts = _fact_extraction_prompts(memory)
        assert prompts
        assert all(display_name not in prompt for prompt in prompts)
        assert all("Narrator:" not in prompt for prompt in prompts)

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_bank_display_name_not_injected_by_dry_run(memory: MemoryEngine):
    bank_id = f"narrator-dry-run-{uuid.uuid4().hex[:8]}"
    display_name = "crm_test_env"
    request_context = RequestContext()

    try:
        await memory.update_bank(bank_id, name=display_name, request_context=request_context)
        memory._retain_llm_config.clear_mock_calls()

        await memory.extract_dry_run(
            bank_id,
            "assistant: I scheduled a truck for next Wednesday morning.",
            context="Conversation between a user and a dispatch assistant.",
            request_context=request_context,
        )

        prompts = _fact_extraction_prompts(memory)
        assert prompts
        assert all(display_name not in prompt for prompt in prompts)
        assert all("Narrator:" not in prompt for prompt in prompts)

        memory._retain_llm_config.clear_mock_calls()
        await memory.extract_dry_run(
            bank_id,
            "assistant: I scheduled a truck for next Wednesday morning.",
            context="Conversation between a user and a dispatch assistant.",
            agent_name="Dispatch",
            request_context=request_context,
        )

        explicit_prompts = _fact_extraction_prompts(memory)
        assert explicit_prompts
        assert all("Narrator: Dispatch" in prompt for prompt in explicit_prompts)
        assert all(display_name not in prompt for prompt in explicit_prompts)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

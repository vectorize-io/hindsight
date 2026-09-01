"""A bank display name must not become the fact-extraction narrator (#3962).

Prompt isolation is pinned deterministically in ``test_narrator_resolution.py``.
This test drives the real retain pipeline and asks the judge to verify the user-
visible result: project/management metadata is not attributed as a participant.
"""

import uuid

import pytest

from hindsight_api import MemoryEngine, RequestContext
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=2, reruns_delay=2)
async def test_bank_display_name_not_attributed_in_extracted_facts(memory_real_llm: MemoryEngine):
    bank_id = f"bank-name-narrator-{uuid.uuid4().hex[:8]}"
    document_id = f"dispatch-dialog-{uuid.uuid4().hex[:8]}"
    display_name = "ReviewModelEval0825"
    request_context = RequestContext()

    try:
        await memory_real_llm.update_bank(
            bank_id,
            name=display_name,
            config_updates={"enable_observations": False},
            request_context=request_context,
        )
        unit_ids = await memory_real_llm.retain_async(
            bank_id,
            (
                "user: start\n"
                "assistant: Hello, this is dispatch. I've scheduled a truck for you for next Wednesday morning.\n"
                "user: Great, thank you."
            ),
            context="Conversation between a customer and an assistant named Dispatch.",
            document_id=document_id,
            request_context=request_context,
        )

        assert unit_ids, "Should extract at least one fact from the conversation"
        listing = await memory_real_llm.list_memory_units(
            bank_id,
            fact_type=["world", "experience"],
            document_id=document_id,
            limit=100,
            request_context=request_context,
        )
        units = listing["items"]
        assert units, "Should persist the extracted conversation facts"
        facts_summary = "\n".join(
            f"- [{unit['fact_type']}] {unit['text']} | entities: {unit['entities']}" for unit in units
        )
        assert display_name not in facts_summary

        await assert_meets_criteria(
            response=facts_summary,
            criteria=(
                "The facts may attribute the scheduling action to Dispatch, the assistant, or a generic agent, "
                "and may mention the customer/user. They must not mention or attribute any action to the bank's "
                "internal display name 'ReviewModelEval0825', because that name was not part of the conversation."
            ),
            context=(
                "A bank has the internal display name 'ReviewModelEval0825'. The retained content is a conversation "
                "where an assistant named Dispatch schedules a truck for a customer. Bank display names are project "
                "metadata, not conversation participants."
            ),
            msg=f"Bank display name leaked into extracted facts: {facts_summary}",
        )
    finally:
        await memory_real_llm.delete_bank(bank_id, request_context=request_context)

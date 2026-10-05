"""A model inventing a complete source ID cannot turn it into verified evidence.

All facts and model output are synthetic. The invented ID shares a prefix with
one real fact: repairing it by that prefix would hide the error this guards.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio


async def test_inline_sources_are_checked_in_answers_and_persisted_pages(client, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)
    memories = await client.memory.list_memories(bank_id, limit=100)
    (memory,) = [item for item in memories.items if item.state == "valid"]
    real_id = memory.id
    forged_id = real_id[:-1] + ("0" if real_id[-1] != "0" else "1")
    source = f"Source: `{forged_id}`"
    answer = f"Source: `{real_id}`\n\n{source}\n\nThe external request UUID is `{forged_id}`."
    expected = answer.replace(source, source + " (unverified)")
    reflect_loop(llm, answer=answer, query="Alice")

    response = await client.areflect(bank_id=bank_id, query="Where does Alice live?", include_tool_calls=True)
    assert response.text == expected
    recalls = [call for call in response.trace.tool_calls if call.tool == "recall"]
    assert any(item["id"] == real_id for call in recalls for item in call.output.get("memories", []))

    created = await client.mental_models.create_mental_model(
        bank_id,
        {"name": "Alice housing", "source_query": "Where does Alice live?", "trigger": {"exclude_mental_models": True}},
    )
    await settled(bank_id)
    page = await client.mental_models.get_mental_model(bank_id, created.mental_model_id, detail="full")
    assert page.content.strip() == expected


class _ReplaceSectionBlocks(BaseModel):
    op: Literal["replace_section_blocks"] = "replace_section_blocks"
    section_id: str
    blocks: list[str]


class _RetractionReply(BaseModel):
    operations: list[_ReplaceSectionBlocks]


async def test_unsay_edits_are_checked_before_the_page_is_saved(client, llm, bank_id, settled):
    """Later edits can invent citations even when the initial reflect was valid."""
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays cello", who="Alice", entities=["Alice", "cello"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin and plays cello.")
    await settled(bank_id)
    memories = await client.memory.list_memories(bank_id, limit=100)
    housing = next(item for item in memories.items if "Berlin" in item.text)
    music = next(item for item in memories.items if "cello" in item.text)
    forged_id = housing.id[:-1] + ("0" if housing.id[-1] != "0" else "1")
    assert forged_id not in {housing.id, music.id}
    live_block = f"Alice plays cello.\nSource: `{music.id}`"
    removed_block = f"Alice moved to Berlin.\nSource: `{housing.id}`"
    llm.on_step("reflect", tool="done").returns_tool_call(
        "done",
        document={"sections": [{"heading": "Facts", "level": 2, "blocks": [removed_block, live_block]}]},
        memory_ids=[housing.id, music.id],
    )
    llm.on_step("reflect").calls_the_offered_tool(query="Alice")
    created = await client.mental_models.create_mental_model(
        bank_id,
        {
            "name": "Alice facts",
            "source_query": "What is known about Alice?",
            "trigger": {"mode": "delta", "exclude_mental_models": True, "fact_types": ["world"]},
        },
    )
    await settled(bank_id)
    initial = await client.mental_models.get_mental_model(bank_id, created.mental_model_id, detail="full")
    assert initial.content.strip() == f"## Facts\n\n{removed_block}\n\n{live_block}"
    assert {item["id"] for item in initial.reflect_response["based_on"]["world"]} == {housing.id, music.id}

    source = f"Source: `{forged_id}`"
    ordinary_uuid = f"The external request UUID is `{forged_id}`."
    llm.on_step("mental_model_retraction").returns(
        _RetractionReply(
            operations=[_ReplaceSectionBlocks(section_id="facts", blocks=[live_block, f"{source}\n{ordinary_uuid}"])]
        )
    )
    await client.memory.update_memory(bank_id, housing.id, {"state": "invalidated", "reason": "Synthetic retraction"})
    await client.mental_models.refresh_mental_model(bank_id, created.mental_model_id)
    await settled(bank_id)
    page = await client.mental_models.get_mental_model(bank_id, created.mental_model_id, detail="full")
    expected = f"## Facts\n\n{live_block}\n\n{source} (unverified)\n{ordinary_uuid}"
    assert page.content.strip() == expected
    assert len(llm.prompts_for("mental_model_retraction")) == 1, "the real unsay call must run"
    assert {item["id"] for item in page.reflect_response["based_on"]["world"]} == {music.id}

    # A quiet refresh must preserve both the live citation and the single marker.
    await client.mental_models.refresh_mental_model(bank_id, created.mental_model_id)
    await settled(bank_id)
    again = await client.mental_models.get_mental_model(bank_id, created.mental_model_id, detail="full")
    assert again.content == page.content
    assert len(llm.prompts_for("mental_model_retraction")) == 1

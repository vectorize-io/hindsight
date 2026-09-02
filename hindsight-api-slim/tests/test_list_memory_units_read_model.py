"""What ``list_memory_units`` / ``list_entities`` put in each item.

Tests that need a unit's write watermark, its lineage, or an entity's kind used to
read the columns straight out of ``memory_units`` / ``entities``. Those are part of
the read model, so they are on the item — and asserted here through the engine, on
units written by retain rather than seeded with SQL, so the coverage holds for any
store behind the memories seam.
"""

import uuid
from datetime import datetime

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memory_engine import MemoryEngine


async def _retain(memory: MemoryEngine, bank_id: str, content: str, request_context: RequestContext) -> list[str]:
    return await memory.retain_async(bank_id=bank_id, content=content, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_fact_type_accepts_a_list(memory: MemoryEngine, request_context: RequestContext):
    """A list of fact types matches any of them — the source-fact selection callers want."""
    bank_id = f"test-lmu-facttypes-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        await _retain(memory, bank_id, "Alice deployed the release on Friday.", request_context)

        async def listed(fact_type):
            page = await memory.list_memory_units(
                bank_id, fact_type=fact_type, limit=500, request_context=request_context
            )
            return {item["id"] for item in page["items"]}, page["total"]

        world_ids, world_total = await listed("world")
        exp_ids, exp_total = await listed("experience")
        both_ids, both_total = await listed(["world", "experience"])

        # The list arm is exactly the union of the single-value arms, so the
        # assertion holds however the LLM happened to classify the facts.
        assert both_ids == world_ids | exp_ids
        assert both_total == world_total + exp_total

        # An empty list filters nothing, matching the "omitted" case.
        _, empty_total = await listed([])
        _, unfiltered_total = await listed(None)
        assert empty_total == unfiltered_total
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_items_carry_updated_at_and_lineage(memory: MemoryEngine, request_context: RequestContext):
    """Each item carries its write watermark and (for observations) its sources."""
    bank_id = f"test-lmu-readmodel-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        await _retain(memory, bank_id, "Bob moved to Berlin in March.", request_context)

        page = await memory.list_memory_units(bank_id, limit=500, request_context=request_context)
        assert page["items"], "retain must have produced at least one fact"

        source_ids = {item["id"] for item in page["items"] if item["fact_type"] != "observation"}
        assert source_ids, "retain must have produced at least one source fact"

        for item in page["items"]:
            # Parseable rather than merely present: callers do date arithmetic on it.
            assert isinstance(datetime.fromisoformat(item["updated_at"]), datetime)
            if item["fact_type"] == "observation":
                # An observation's lineage points at the facts it was drawn from,
                # and those facts are in this same bank.
                assert item["source_memory_ids"], "an observation must carry its sources"
                assert set(item["source_memory_ids"]) <= source_ids
            else:
                # A source fact has no lineage; the field is always there, never absent.
                assert item["source_memory_ids"] == []

        # The list item and the detail view agree on the unit.
        first = page["items"][0]
        detail = await memory.get_memory_unit(bank_id, first["id"], request_context)
        assert detail["text"] == first["text"]
        if first["fact_type"] == "observation":
            assert detail["source_memory_ids"] == first["source_memory_ids"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_entities_carry_their_kind(memory: MemoryEngine, request_context: RequestContext):
    """list_entities reports how each entity was classified."""
    bank_id = f"test-entities-kind-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        await _retain(memory, bank_id, "Carol works with Dave at Acme.", request_context)

        page = await memory.list_entities(bank_id, limit=500, request_context=request_context)
        assert page["items"], "retain must have produced at least one entity"
        for item in page["items"]:
            assert "entity_kind" in item, "entity_kind is part of the entity read model"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

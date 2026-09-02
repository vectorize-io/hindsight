"""Integration test for in-batch entity dedup (issue #3107).

On a fresh bank, surface-form variants of the same entity that appear in a single retain batch
must collapse to one entity — previously the fuzzy match only ran against already-persisted rows,
so the first sighting of each variant created a distinct entity. Genuinely distinct-but-similar
names must stay separate. Deterministic (in-memory trigram similarity), so asserted directly — no LLM.
"""

import uuid
from datetime import datetime, timezone

import pytest

from hindsight_api.engine.retain.entity_processing import resolve_entities
from hindsight_api.engine.retain.types import EntityRef, ProcessedFact


def _fact(entity_names: list[str]) -> ProcessedFact:
    now = datetime.now(timezone.utc)
    return ProcessedFact(
        fact_text="They collaborated on the review.",
        fact_type="world",
        embedding=[0.0] * 384,
        occurred_start=now,
        occurred_end=None,
        mentioned_at=now,
        context="",
        metadata={},
        entities=[EntityRef(name=n) for n in entity_names],
        content_index=0,
        tags=[],
    )


@pytest.mark.asyncio
async def test_intrabatch_variants_collapse_but_distinct_names_stay_separate(memory, request_context):
    bank_id = f"test-3107-intrabatch-{uuid.uuid4().hex[:8]}"
    # Order matters: resolved_entity_ids is one id per input entity, in this order.
    names = ["Wren 🕯️", "Wren 🗯️", "Merrivale", "Merryvale", "Aster", "aster 0", "Astrid"]
    try:
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        async with memory._pool.acquire() as conn:
            resolution = await resolve_entities(
                entity_resolver=memory.entity_resolver,
                conn=conn,
                bank_id=bank_id,
                unit_ids=[str(uuid.uuid4())],
                facts=[_fact(names)],
                entity_labels=None,
            )
        ids = resolution.resolved_entity_ids
        assert len(ids) == len(names)
        by_name = dict(zip(names, ids))

        # Emoji-only variants and the case/suffix variant collapse; the typo pair collapses.
        assert by_name["Wren 🕯️"] == by_name["Wren 🗯️"], "emoji-only variants must be one entity"
        assert by_name["Merrivale"] == by_name["Merryvale"], "typo variants must be one entity"
        assert by_name["Aster"] == by_name["aster 0"], "case/suffix variants must be one entity"

        # 'Astrid' is a genuinely different person (trigram sim ~0.30) — must NOT merge into 'Aster'.
        assert by_name["Astrid"] != by_name["Aster"], "distinct-but-similar names must stay separate"

        # 7 mentions → 4 distinct entities (Wren, Merrivale, Aster, Astrid).
        assert len(set(ids)) == 4, f"expected 4 distinct entities, got {len(set(ids))}"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_intrabatch_dedup_leaves_label_values_separate(memory, request_context):
    """The fuzzy in-batch pass must exclude label entities — distinct label values stay separate
    even when textually near-identical (GH-1558), same as the persisted-candidate path."""
    bank_id = f"test-3107-label-{uuid.uuid4().hex[:8]}"
    entity_labels = [
        {
            "key": "use",
            "type": "multi-values",
            "tag": True,
            "values": [{"value": "use-001"}, {"value": "use-002"}],
        }
    ]
    try:
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
        async with memory._pool.acquire() as conn:
            resolution = await resolve_entities(
                entity_resolver=memory.entity_resolver,
                conn=conn,
                bank_id=bank_id,
                unit_ids=[str(uuid.uuid4())],
                facts=[_fact(["use:use-001", "use:use-002"])],
                entity_labels=entity_labels,
            )
        assert len(set(resolution.resolved_entity_ids)) == 2, "label values must not be fuzzy-merged"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

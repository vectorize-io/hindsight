"""Tests that list_memory_units surfaces entities for observation rows.

Observations don't carry direct unit_entities rows -- their entity associations
live transitively through source_memory_ids. get_memory_unit and the recall
path already use _entity_rows_for_units_sql to UNION direct + inherited rows
in a single query; list_memory_units used to issue a plain JOIN against
unit_entities, which returned entities="" for every observation row.

This file pins down the desired behavior: list responses for observation
rows must surface entities inherited from their source memories.
"""

import uuid

import pytest


@pytest.fixture(autouse=True)
def enable_observations():
    """Observations only get created when the engine config opts in."""
    from hindsight_api.config import _get_raw_config

    config = _get_raw_config()
    original_value = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original_value


@pytest.mark.asyncio
async def test_list_memory_units_observation_inherits_entities_from_source(memory, request_context):
    """Observations returned by list_memory_units carry their source memories' entities.

    Before the fix this assertion failed because list_memory_units issued a
    direct JOIN against unit_entities -- observations have no rows there, so
    `entities` came back as an empty string even though the same memory
    fetched via get_memory_unit had populated entities (because get_memory_unit
    already used the UNION helper).
    """
    bank_id = f"test-list-obs-entities-{uuid.uuid4().hex[:8]}"

    try:
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        # Retain content with a clearly-named entity. SyncTaskBackend runs
        # consolidation inline, so an observation row is produced before the
        # retain call returns in tests.
        await memory.retain_async(
            bank_id=bank_id,
            content="John Smith is the CEO of Acme Corporation.",
            request_context=request_context,
        )

        result = await memory.list_memory_units(
            bank_id=bank_id,
            fact_type="observation",
            limit=50,
            offset=0,
            request_context=request_context,
        )

        observation_items = [item for item in result["items"] if item["fact_type"] == "observation"]
        assert observation_items, "expected at least one observation row after retain"

        # Inherited entities must surface in the list response. At minimum one
        # of the named entities should land in the comma-joined string -- the
        # LLM may pick either canonical form (e.g. "John Smith", "Acme
        # Corporation", or both), so we accept either rather than pinning a
        # specific name list and making the test brittle.
        observation_with_entities = next(
            (item for item in observation_items if item["entities"]),
            None,
        )
        assert observation_with_entities is not None, (
            "expected list_memory_units to surface inherited entities for at least one "
            "observation row, got entities='' on every observation. This means the "
            "observation→source_memory_ids inheritance is missing from the list path."
        )

        # Cross-check: the same observation fetched via get_memory_unit (which
        # already used the UNION helper before this change) must produce the
        # same entity set. If they diverge we've introduced a regression in
        # one path or the other.
        single = await memory.get_memory_unit(
            bank_id=bank_id,
            memory_id=observation_with_entities["id"],
            request_context=request_context,
        )
        single_entities = set(single["entities"])
        list_entities = {e.strip() for e in observation_with_entities["entities"].split(",") if e.strip()}
        assert single_entities == list_entities, (
            f"list_memory_units entities {list_entities} disagree with "
            f"get_memory_unit entities {single_entities} for the same observation; "
            "both paths should use _entity_rows_for_units_sql."
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_list_memory_units_world_facts_still_return_direct_entities(memory, request_context):
    """Regression guard: world/experience rows continue to receive their own
    direct unit_entities. The UNION helper only adds inherited rows for
    observations whose unit has no direct entities; it must not break the
    common direct-link path that world/experience rely on.
    """
    bank_id = f"test-list-world-entities-{uuid.uuid4().hex[:8]}"

    try:
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        await memory.retain_async(
            bank_id=bank_id,
            content="Maria works as a software engineer at Microsoft.",
            request_context=request_context,
        )

        # Pull world/experience rows specifically -- avoids ordering coupling
        # with the observation produced by the same retain.
        for fact_type in ("world", "experience"):
            result = await memory.list_memory_units(
                bank_id=bank_id,
                fact_type=fact_type,
                limit=50,
                offset=0,
                request_context=request_context,
            )
            if not result["items"]:
                # Not every retain produces both fact_types; skip the type
                # the LLM didn't emit rather than asserting both exist.
                continue

            item_with_entities = next((item for item in result["items"] if item["entities"]), None)
            assert item_with_entities is not None, (
                f"expected at least one {fact_type} row with direct entities; got "
                f"entities='' on every {fact_type} row. The UNION helper may have "
                "broken the direct-link path."
            )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

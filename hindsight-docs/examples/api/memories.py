#!/usr/bin/env python3
"""
Memories API examples for Hindsight — read, list, and curate memory units.
Run: python examples/api/memories.py
"""
import asyncio
import os

from hindsight_client import Hindsight
from hindsight_client_api.models.update_memory_request import UpdateMemoryRequest

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
BANK_ID = "memories-demo-bank"


async def main():
    # =========================================================================
    # Setup (not shown in docs)
    # =========================================================================
    client = Hindsight(base_url=HINDSIGHT_URL)
    await client.acreate_bank(bank_id=BANK_ID, name="Memories Demo")
    await client.aretain(bank_id=BANK_ID, content="The assistant visited Paris in 2023.")
    await client.aretain(bank_id=BANK_ID, content="The deploy server srv-04 runs PostgreSQL 14.")
    await asyncio.sleep(3)  # let extraction finish

    # =========================================================================
    # Doc Examples
    # =========================================================================

    # [docs:list-memories]
    # List memory units in a bank. Invalidated rows are included by default.
    memories = await client.memory.list_memories(bank_id=BANK_ID)
    for unit in memories.items:
        print(f"- [{unit['fact_type']}] {unit['text']}")

    # Filter to only the invalidated facts (e.g. to review duplicates).
    invalidated = await client.memory.list_memories(bank_id=BANK_ID, state="invalidated")
    print(f"{len(invalidated.items)} invalidated fact(s)")
    # [/docs:list-memories]

    # Grab a raw fact (world/experience) to curate in the examples below.
    fact = next((u for u in memories.items if u["fact_type"] in ("world", "experience")), None)
    if fact is None:
        await client.adelete_bank(bank_id=BANK_ID)
        print("memories.py: All examples passed (no facts extracted yet)")
        return
    memory_id = fact["id"]

    # [docs:get-memory]
    # Fetch a single memory unit (includes entities, dates, and state).
    memory = await client.memory.get_memory(bank_id=BANK_ID, memory_id=memory_id)

    print(f"Text: {memory['text']}")
    print(f"Type: {memory['type']}  Entities: {memory['entities']}")
    # [/docs:get-memory]

    # [docs:edit-memory]
    # Correct the fact's text. Re-embeds, drops derived observations/links,
    # re-consolidates, and recomputes the graph automatically.
    await client.memory.update_memory(
        bank_id=BANK_ID,
        memory_id=memory_id,
        update_memory_request=UpdateMemoryRequest(
            text="The user visited Paris in 2023.",
            reason="wrong subject",
        ),
    )
    # [/docs:edit-memory]

    # [docs:edit-memory-fields]
    # Correct dates, fact type, and entities in one call. "" clears a field;
    # entities replaces the set ([] detaches all); omit to leave unchanged.
    await client.memory.update_memory(
        bank_id=BANK_ID,
        memory_id=memory_id,
        update_memory_request=UpdateMemoryRequest(
            occurred_start="2023-06-01",
            fact_type="experience",
            entities=["Alice", "Paris"],
        ),
    )
    # [/docs:edit-memory-fields]

    # [docs:invalidate-memory]
    # Soft-retire a fact: removed from recall/consolidation/graph, links pruned,
    # derived observations recomputed without it — but kept for audit.
    await client.memory.update_memory(
        bank_id=BANK_ID,
        memory_id=memory_id,
        update_memory_request=UpdateMemoryRequest(
            state="invalidated",
            reason="server decommissioned 2026-06-01",
        ),
    )
    # [/docs:invalidate-memory]

    # [docs:restore-memory]
    # Restore a previously invalidated fact.
    await client.memory.update_memory(
        bank_id=BANK_ID,
        memory_id=memory_id,
        update_memory_request=UpdateMemoryRequest(state="valid"),
    )
    # [/docs:restore-memory]

    # An observation (derived) exposes how it evolved as sources arrived.
    observation = next((u for u in memories.items if u["fact_type"] == "observation"), None)
    if observation is not None:
        # [docs:observation-history]
        # Get the refresh history of a derived observation.
        history = await client.memory.get_observation_history(
            bank_id=BANK_ID, memory_id=observation["id"]
        )
        print(f"Observation history entries: {len(history)}")
        # [/docs:observation-history]

    # =========================================================================
    # Cleanup (not shown in docs)
    # =========================================================================
    await client.adelete_bank(bank_id=BANK_ID)
    print("memories.py: All examples passed")


asyncio.run(main())

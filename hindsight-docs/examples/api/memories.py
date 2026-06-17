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


async def wait_for_idle(client, bank_id, *, attempts=120, interval=0.5):
    """Block until the bank has no pending/processing async operations.

    Retain and every update_memory edit re-embed and re-consolidate in the
    background. Draining that queue between curation steps keeps the example
    deterministic — otherwise a later step can race the prior step's
    re-consolidation and 404 on a unit that is mid-rewrite.
    """
    await asyncio.sleep(interval)  # let the just-submitted operation register
    for _ in range(attempts):
        busy = False
        for state in ("pending", "processing"):
            res = await client.operations.list_operations(
                bank_id=bank_id, status=state, limit=1
            )
            if res.operations:
                busy = True
                break
        if not busy:
            return
        await asyncio.sleep(interval)


async def main():
    # =========================================================================
    # Setup (not shown in docs)
    # =========================================================================
    client = Hindsight(base_url=HINDSIGHT_URL)
    await client.acreate_bank(bank_id=BANK_ID, name="Memories Demo")
    await client.aretain(bank_id=BANK_ID, content="The assistant visited Paris in 2023.")
    await client.aretain(bank_id=BANK_ID, content="The deploy server srv-04 runs PostgreSQL 14.")
    await wait_for_idle(client, BANK_ID)  # let extraction + consolidation finish

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

    # An observation (derived) exposes how it evolved as sources arrived. Read
    # it here, before the edit/invalidate/restore below: any update_memory call
    # re-consolidates and recreates observations with new ids, so an id captured
    # before those steps is stale afterward (get_observation_history would 404).
    observation = next((u for u in memories.items if u["fact_type"] == "observation"), None)
    if observation is not None:
        # [docs:observation-history]
        # Get the refresh history of a derived observation.
        history = await client.memory.get_observation_history(
            bank_id=BANK_ID, memory_id=observation["id"]
        )
        print(f"Observation history entries: {len(history)}")
        # [/docs:observation-history]

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

    # The text edit re-consolidates in the background; let it settle before the
    # next edit so the unit isn't mid-rewrite when we touch it again.
    await wait_for_idle(client, BANK_ID)

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

    await wait_for_idle(client, BANK_ID)

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

    # Invalidation also re-derives observations in the background; drain it so the
    # restore below doesn't race a unit that is being moved between tables.
    await wait_for_idle(client, BANK_ID)

    # [docs:restore-memory]
    # Restore a previously invalidated fact.
    await client.memory.update_memory(
        bank_id=BANK_ID,
        memory_id=memory_id,
        update_memory_request=UpdateMemoryRequest(state="valid"),
    )
    # [/docs:restore-memory]

    # =========================================================================
    # Cleanup (not shown in docs)
    # =========================================================================
    await client.adelete_bank(bank_id=BANK_ID)
    print("memories.py: All examples passed")


asyncio.run(main())

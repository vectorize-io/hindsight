"""A receiver hears when a mental model has been refreshed.

Before `mental_model.refreshed`, a receiver that wanted to act on a fresh mental
model had nothing to listen for: `consolidation.completed` fires before the
follow-up refresh has run, and a manual or scheduled refresh sent nothing at all.
So the property worth pinning is the seam between the two features — a refresh
the worker finishes reaches the receiver once, saying which model, which
operation, and what the refresh did.
"""

from __future__ import annotations

import asyncio

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

ANSWER = "Alice lives in Berlin."


@pytest.fixture
async def bank_with_facts(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    reflect_loop(llm, answer=ANSWER)
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)
    return bank_id


async def test_a_finished_refresh_reaches_the_receiver_once(client, stubs, bank_with_facts, stub_server, settled):
    """Subscribed to this event only, so the consolidation the retain ran earlier
    sends nothing here: the one delivery is the refresh that creating the model
    queued, and it carries that operation's id so the receiver can match it."""
    await client.webhooks.create_webhook(
        bank_with_facts, {"url": f"{stub_server.url}/webhook", "event_types": ["mental_model.refreshed"]}
    )
    created = await client.mental_models.create_mental_model(
        bank_with_facts, {"name": "Alice housing", "source_query": "Where does Alice live?"}
    )
    await settled(bank_with_facts)

    for _ in range(30):
        if stubs.webhooks:
            break
        await asyncio.sleep(1)
    assert len(stubs.webhooks) == 1, "exactly one refresh, exactly one event"

    received = stubs.webhooks[0]
    body = received.body
    assert received.headers["x-hindsight-event"] == "mental_model.refreshed"
    assert body["event"] == "mental_model.refreshed"
    assert body["bank_id"] == bank_with_facts
    assert body["operation_id"] == created.operation_id
    assert body["status"] == "completed"
    assert body["data"] == {
        "mental_model_id": created.mental_model_id,
        "name": "Alice housing",
        "outcome": "content_written",
    }

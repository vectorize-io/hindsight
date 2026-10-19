"""A bank that groups observations by subject, set once on the server (#5666).

Clients tag what they know — who wrote a memory, which session it came from.
The subject is something only the server learns, during extraction, from an
entity label with `tag: true`. Under the default `combined` scoping every one of
those tag sets is unique, so every memory consolidated alone: one LLM call per
memory, and two agents' notes on the same subject never met.

Two pieces compose here, and neither is visible to the client: the label that
adds `topic:billing`, and the bank's `default_observation_scopes`
`[["topic:*"]]`, whose glob picks that tag out of each memory's own tags. The
story checks the seam between them — memories from different sources land in
one observation per topic, built in one consolidation call — and that a client
that does declare its scopes is left alone.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import Consolidation, Observation, extracted, fact, fact_ids_in

pytestmark = pytest.mark.asyncio

NOTES = {
    "billing-a": ("Agent A: the customer was double charged on the March invoice.", "topic:billing"),
    "billing-b": ("Agent B: the March invoice refund for the double charge went out.", "topic:billing"),
    "auth": ("Agent A: the customer reset their password after a lockout.", "topic:auth"),
}


@pytest.fixture
def consolidate_calls(llm) -> list[str]:
    """Record the prompt of each consolidation call, and observe every fact it was shown."""
    calls: list[str] = []

    def build(request) -> Consolidation:
        ids = fact_ids_in(request.all_text)
        calls.append(request.all_text)
        return Consolidation(creates=[Observation(text="customer support notes", source_fact_ids=ids, reason="test")])

    for name, (text, topic) in NOTES.items():
        llm.on_step("extract_facts", contains=text).returns(
            extracted(fact(text.split(": ", 1)[1].rstrip("."), entities=[topic]))
        )
    llm.on_step("consolidate").answers_with(build)
    return calls


async def _topic_bank(client, bank_id: str) -> None:
    await client.banks.update_bank_config(
        bank_id,
        {
            "updates": {
                "entity_labels": [
                    {
                        "key": "topic",
                        "type": "value",
                        "tag": True,
                        "values": [{"value": "billing"}, {"value": "auth"}],
                    }
                ],
                "default_observation_scopes": [["topic:*"]],
                # Retain all three before consolidating, so batching decides who shares a call.
                "enable_auto_consolidation": False,
            }
        },
    )


async def _observation_tags(client, bank_id: str) -> list[list[str]]:
    memories = await client.memory.list_memories(bank_id, limit=100)
    return sorted(sorted(m.tags) for m in memories.items if m.fact_type == "observation")


async def test_memories_from_different_sources_consolidate_by_topic(client, bank_id, settled, consolidate_calls):
    await _topic_bank(client, bank_id)
    sources = {"billing-a": "source:agent-a", "billing-b": "source:agent-b", "auth": "source:agent-a"}
    await client.aretain_batch(
        bank_id=bank_id,
        items=[
            {"content": text, "document_id": name, "tags": [sources[name], f"session:{name}"]}
            for name, (text, _) in NOTES.items()
        ],
    )
    await settled(bank_id)
    await client.banks.trigger_consolidation(bank_id)
    await settled(bank_id)

    assert await _observation_tags(client, bank_id) == [["topic:auth"], ["topic:billing"]]
    # One call per topic, and both billing notes were shown to the LLM together — that is
    # the whole saving: under `combined` each of the three memories needs its own call.
    assert len(consolidate_calls) == 2
    billing = [p for p in consolidate_calls if "double charged" in p]
    assert len(billing) == 1 and "refund" in billing[0]


async def test_a_client_that_declares_its_scopes_keeps_them(client, bank_id, settled, consolidate_calls):
    await _topic_bank(client, bank_id)
    text, _ = NOTES["billing-a"]
    await client.aretain_batch(
        bank_id=bank_id,
        items=[{"content": text, "tags": ["source:agent-a"], "observation_scopes": "combined"}],
    )
    await settled(bank_id)
    await client.banks.trigger_consolidation(bank_id)
    await settled(bank_id)

    assert await _observation_tags(client, bank_id) == [["source:agent-a", "topic:billing"]]

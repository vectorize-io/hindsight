"""A label's tags outlive the label group they came from (#6336).

An entity label with `tag: true` copies each fact's label into its tags, so a
`tags` filter can find it. Two things meet here that no single test crosses:
the bank's label config changing, and a later edit of the document's tags.

Retiring a group from the config must only stop NEW extraction from assigning
it. Facts extracted while it existed keep their tag through a document retag —
that retag re-extracts nothing, so it has no business deciding which labels a
fact has. Before the fix it decided by the current config, so the group's tags
looked like caller tags and were wiped on every unit. Reprocessing the document
is the supported way to drop them: it extracts again under the current config.
"""

from __future__ import annotations

import pytest
from hindsight_client_api.models.update_document_request import UpdateDocumentRequest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

DOCUMENT_ID = "decisions"
CONTENT = "We keep Postgres as the only database."
CATEGORY = {"key": "category", "type": "value", "tag": True, "values": [{"value": "durable"}]}


async def _fact_tags(client, bank_id: str) -> list[list[str]]:
    memories = await client.memory.list_memories(bank_id, document_id=DOCUMENT_ID, limit=100)
    return sorted(sorted(m.tags) for m in memories.items if m.fact_type != "observation")


async def _found_by_label(client, bank_id: str) -> int:
    memories = await client.memory.list_memories(bank_id, tags=["category:durable"], tags_match="any_strict", limit=100)
    return sum(1 for m in memories.items if m.fact_type != "observation")


async def test_retiring_a_label_group_keeps_its_tags_until_reprocess(client, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(
        extracted(fact("The team keeps Postgres as the only database", entities=["category:durable"]))
    )
    # A retag and a reprocess both requeue the document's memories for consolidation.
    llm.on_step("consolidate").returns(consolidation())

    await client.banks.update_bank_config(bank_id, {"updates": {"entity_labels": [CATEGORY]}})
    await client.aretain_batch(
        bank_id=bank_id, items=[{"content": CONTENT, "document_id": DOCUMENT_ID, "tags": ["scope:first"]}]
    )
    await settled(bank_id)
    assert await _fact_tags(client, bank_id) == [["category:durable", "scope:first"]]

    # The group leaves the taxonomy; then an ordinary document tag changes.
    await client.banks.update_bank_config(bank_id, {"updates": {"entity_labels": []}})
    await client.documents.update_document(bank_id, DOCUMENT_ID, UpdateDocumentRequest(tags=["scope:second"]))
    await settled(bank_id)

    assert await _fact_tags(client, bank_id) == [["category:durable", "scope:second"]]
    assert await _found_by_label(client, bank_id) == 1

    # Reprocess extracts again under the current config, which no longer has the group.
    await client.documents.reprocess_document(bank_id, DOCUMENT_ID)
    await settled(bank_id)

    assert await _fact_tags(client, bank_id) == [["scope:second"]]
    assert await _found_by_label(client, bank_id) == 0

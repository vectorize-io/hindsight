"""A memory bank that links a knowledge bank reflects with its documents, and only reflect does.

Two capabilities meet here for the first time: a memory bank's reflect loop and a knowledge bank's
search. The link is one bank-config field, ``knowledge_banks``. Once set, reflect's recall tool hands
the model the linked documents' best passages next to the memories, flagged as the authority over a
memory that disagrees. The public recall API does not change: it never returns documents.

The stub writes the answer, so what is asserted is what reached the model: the passage text in a
reflect turn, and its absence once the link is removed.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest

from hindsight_system_tests import KnowledgeBanks, reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

POLICY = "Refunds are accepted within 60 days of purchase, with the original receipt."
QUERY = "How many days do customers have to ask for a refund?"


@pytest.fixture
async def knowledge_bank(client) -> AsyncIterator[str]:
    kb = KnowledgeBanks(client)
    bank_id = f"systest-kb-{uuid.uuid4().hex[:12]}"
    assert (await kb.create(bank_id)).status == 200
    r = await kb.ingest(bank_id, [{"document_id": "refund-policy", "text": POLICY}])
    assert r.status == 200, r.body
    yield bank_id
    await kb.delete(bank_id)


@pytest.fixture
async def memory_bank(client, llm, bank_id, settled) -> str:
    # The memory disagrees with the document: the document is the authority.
    llm.on_step("extract_facts").returns(
        extracted(fact("Customers have 30 days to ask for a refund", who="support", entities=["refund"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Support said customers have 30 days to ask for a refund.")
    await settled(bank_id)
    return bank_id


def _reflect_turns_with(llm, text: str) -> list[str]:
    return [call.all_text for call in llm.calls if text in call.all_text]


async def test_reflect_hands_the_model_the_linked_documents_as_the_authority(
    client, llm, memory_bank, knowledge_bank
):
    await client.aupdate_bank_config(memory_bank, knowledge_banks=[knowledge_bank])
    reflect_loop(llm, answer="60 days, with the original receipt.", query=QUERY)

    response = await client.areflect(bank_id=memory_bank, query=QUERY)

    assert response.text == "60 days, with the original receipt."
    turns = _reflect_turns_with(llm, POLICY)
    assert turns, "no reflect turn saw the linked document"
    # It arrives next to the memory, labelled as the authority, with where it came from.
    assert any("30 days" in t for t in turns)
    assert any("authoritative" in t and "refund-policy" in t for t in turns)


async def test_recall_never_returns_the_linked_documents(client, memory_bank, knowledge_bank):
    await client.aupdate_bank_config(memory_bank, knowledge_banks=[knowledge_bank])

    result = await client.arecall(bank_id=memory_bank, query=QUERY)

    assert [r.text for r in result.results] == ["Customers have 30 days to ask for a refund | Involving: support"]
    assert POLICY not in result.model_dump_json()


async def test_unlinking_takes_the_documents_out_of_reflect(client, llm, memory_bank, knowledge_bank):
    await client.aupdate_bank_config(memory_bank, knowledge_banks=[knowledge_bank])
    await client.aupdate_bank_config(memory_bank, knowledge_banks=[])
    reflect_loop(llm, answer="30 days.", query=QUERY)

    await client.areflect(bank_id=memory_bank, query=QUERY)

    assert not _reflect_turns_with(llm, POLICY)
    assert _reflect_turns_with(llm, "30 days to ask for a refund")


async def test_a_deleted_linked_bank_is_skipped_not_an_error(client, llm, memory_bank):
    kb = KnowledgeBanks(client)
    gone = f"systest-kb-{uuid.uuid4().hex[:12]}"
    assert (await kb.create(gone)).status == 200
    await client.aupdate_bank_config(memory_bank, knowledge_banks=[gone])
    assert (await kb.delete(gone)).status == 200
    reflect_loop(llm, answer="30 days.", query=QUERY)

    response = await client.areflect(bank_id=memory_bank, query=QUERY)

    assert response.text == "30 days."

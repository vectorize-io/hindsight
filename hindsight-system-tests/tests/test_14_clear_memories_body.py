"""A mistaken selection body must leave the bank intact, not clear it (#4337)."""

from __future__ import annotations

import json

import pytest
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio


async def test_selection_body_cannot_clear_the_bank(client, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice lives in Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays cello", who="Alice", entities=["Alice", "cello"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice lives in Berlin and plays cello.", document_id="profile")
    await settled(bank_id)

    before = await client.memory.list_memories(bank_id, limit=100)
    assert sorted(m.text for m in before.items) == [
        "Alice lives in Berlin | Involving: Alice",
        "Alice plays cello | Involving: Alice",
    ]
    ids = sorted(m.id for m in before.items)
    # The generated clear method intentionally has no body argument. Use the
    # published SDK transport to reproduce the malformed caller request.
    api = client.memory.api_client
    response = await api.call_api(
        *api.param_serialize(
            method="DELETE",
            resource_path="/v1/default/banks/{bank_id}/memories",
            path_params={"bank_id": bank_id},
            header_params={"Content-Type": "application/json"},
            body={"ids": [ids[0]]},
        )
    )
    await response.read()
    assert response.status == 400
    assert "does not accept a request body" in json.loads(response.data)["detail"]

    after = await client.memory.list_memories(bank_id, limit=100)
    assert sorted(m.id for m in after.items) == ids
    assert sorted(m.text for m in after.items) == sorted(m.text for m in before.items)
    documents = await client.documents.list_documents(bank_id)
    assert [d.id for d in documents.items] == ["profile"]

    # Supported, bodyless client calls still honor the query filter and can clear.
    result = await client.memory.clear_bank_memories(bank_id, type="observation")
    assert result.success is True
    assert sorted(m.id for m in (await client.memory.list_memories(bank_id)).items) == ids
    result = await client.memory.clear_bank_memories(bank_id)
    assert result.success is True
    assert (await client.memory.list_memories(bank_id)).items == []
    assert (await client.documents.list_documents(bank_id)).items == []
    assert bank_id in [bank.bank_id for bank in (await client.banks.list_banks()).banks]

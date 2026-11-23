"""A replace that is retried after a newer one landed does not roll the document back (#6763).

Retains of one document run in the order they were sent — but a retain waiting out its retry
delay does not hold the line. So when the first version of a document hits a provider blip, the
second version runs and lands while the first one waits, and then the first one's retry used to
run on top of it: the document went back to the older body and its facts with it, and the older
text was paid for again. A replace carries the whole document, so the retry has nothing left to
do once a version sent after it has landed.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.waiting import wait_until_settled

pytestmark = pytest.mark.asyncio

DOCUMENT_ID = "alice-home"
OLDER = "Alice lives in Paris and walks to work along the river every morning."
NEWER = "Alice lives in Berlin now and cycles to work through the park every morning."


async def test_a_retried_older_replace_leaves_the_newer_version_in_place(worker_retry_client, llm, worker_retry_bank):
    client, bank = worker_retry_client, worker_retry_bank
    await client.acreate_bank(bank_id=bank, name="Alice")
    llm.on_step("consolidate").returns(consolidation())
    # The older version's first extraction hits a blip; its retry would extract it fine.
    llm.on_step("extract_facts", contains="Paris").fails_with_status(503, times=1)
    llm.on_step("extract_facts", contains="Paris").returns(
        extracted(fact("Alice lives in Paris", who="Alice", entities=["Alice", "Paris"]))
    )
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(fact("Alice lives in Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )

    async with llm.hold("extract_facts", contains="Berlin") as newer_running:
        older = await client.aretain(bank_id=bank, content=OLDER, document_id=DOCUMENT_ID, retain_async=True)
        await client.aretain(bank_id=bank, content=NEWER, document_id=DOCUMENT_ID, retain_async=True)
        await newer_running.reached()
        # The order this story is about: the newer version is running while the older one waits
        # out its retry. Asserted, not assumed — had the retry run first, the story would pass
        # without ever reaching the case.
        status = await client.operations.get_operation_status(bank, older.operation_id)
        assert status.status == "pending", f"the older retain was {status.status} while the newer one ran"
        assert len(llm.prompts_for("extract_facts")) == 2, "expected one failed Paris call, then Berlin"
    # allow_failed: the older retain carries its blip's error until its retry finishes; its end
    # state is asserted below instead.
    await wait_until_settled(client, bank, allow_failed=True)

    document = await client.documents.get_document(bank, DOCUMENT_ID)
    assert document.original_text == NEWER
    memories = await client.memory.list_memories(bank, limit=100)
    assert sorted(m.text for m in memories.items if m.fact_type != "observation") == [
        "Alice lives in Berlin | Involving: Alice"
    ]
    # The older text was not extracted again: its one call is the blip.
    assert len([p for p in llm.prompts_for("extract_facts") if "Paris" in p]) == 1
    status = await client.operations.get_operation_status(bank, older.operation_id)
    assert status.status == "completed"

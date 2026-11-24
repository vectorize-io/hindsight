"""A page drops what rested on a deleted memory, and never loses new facts doing it.

A delta refresh makes two edits. First it removes what the page says on the strength of
memories that no longer exist; then it folds in the memories new since the last refresh.
Each is unit-tested alone. Two bugs lived where they meet the rest of the bank:

**A bank without observations never got the removal (#6855).** The removal waits until no
fact is pending consolidation, so a re-ingest's replacements exist before old prose goes.
A bank with observations off never consolidates, so every fact read as pending forever:
the wait never ended, nothing scheduled the refresh, and the deleted claim stayed on the
page for good.

**A failed second edit was saved as a success (#6447).** When the removal landed and the
new-facts edit then failed, the refresh still wrote the page with the removal alone, and
moved past the new memories. No later delta refresh would ever show them to the page.
The refresh must fail instead, change nothing, and let the next one redo both edits.
"""

from __future__ import annotations

import asyncio

import pytest

from hindsight_system_tests import reflect_loop, wait_until_settled
from hindsight_system_tests.payloads import (
    DeltaOperations,
    RemoveBlock,
    add_section,
    extracted,
    fact,
    remove_block_saying,
)

pytestmark = pytest.mark.asyncio

QUERY = "What are the rules for deploying the checkout service?"
REVIEWER = "Every checkout deploy needs a second reviewer."
FRIDAY = "Checkout deploys are frozen on Fridays."
CANARY = "A checkout deploy must pass 30 minutes in canary."
PAGE = f"## Checkout deploys\n\n{REVIEWER}\n\n{FRIDAY}"


async def _retain(client, llm, bank_id: str, document_id: str, text: str) -> None:
    llm.on_step("extract_facts", contains=text).returns(extracted(fact(text, entities=["checkout"])))
    await client.aretain(bank_id=bank_id, content=text, document_id=document_id)


async def _refreshes(client, bank_id: str, status: str | None = None) -> int:
    operations = await client.operations.list_operations(bank_id, type="refresh_mental_model", status=status, limit=100)
    return operations.total


async def _content(client, bank_id: str, model_id: str) -> str:
    return (await client.mental_models.get_mental_model(bank_id, model_id, detail="full")).content.strip()


async def _page(client, llm, bank_id: str, settled, *, refresh_on_its_own: bool) -> str:
    """A delta page over two raw facts, on a bank that makes no observations."""
    await client.banks.update_bank_config(
        bank_id, {"updates": {"enable_observations": False, "enable_auto_consolidation": False}}
    )
    await _retain(client, llm, bank_id, "rules", REVIEWER)
    await _retain(client, llm, bank_id, "freeze", FRIDAY)
    await settled(bank_id)
    reflect_loop(llm, answer=PAGE, query=QUERY)
    created = await client.mental_models.create_mental_model(
        bank_id,
        {
            "name": "Checkout deploys",
            "source_query": QUERY,
            "trigger": {
                "mode": "delta",
                "fact_types": ["world"],
                "exclude_mental_models": True,
                "refresh_after_consolidation": refresh_on_its_own,
            },
        },
    )
    await settled(bank_id)
    assert await _content(client, bank_id, created.mental_model_id) == PAGE
    llm.reset()
    reflect_loop(llm, answer=CANARY, query=QUERY)
    return created.mental_model_id


async def test_deleting_the_memory_removes_its_claim_on_a_bank_without_observations(client, llm, bank_id, settled):
    model = await _page(client, llm, bank_id, settled, refresh_on_its_own=True)
    before = await _refreshes(client, bank_id)
    llm.on_step("mental_model_retraction").answers_with(remove_block_saying(FRIDAY))

    await client.documents.delete_document(bank_id, "freeze")
    # The delete schedules the refresh itself: nobody else on this bank ever will.
    for _ in range(30):
        if await _refreshes(client, bank_id) > before:
            break
        await asyncio.sleep(1)
    await settled(bank_id)

    assert await _refreshes(client, bank_id) == before + 1, "deleting the memory scheduled no refresh"
    retractions = llm.prompts_for("mental_model_retraction")
    assert len(retractions) == 1 and FRIDAY in retractions[0], "the removal was never asked for"
    assert await _content(client, bank_id, model) == f"## Checkout deploys\n\n{REVIEWER}"


async def test_a_failed_new_facts_edit_keeps_the_page_and_the_next_refresh_does_both(client, llm, bank_id, settled):
    model = await _page(client, llm, bank_id, settled, refresh_on_its_own=False)
    await client.documents.delete_document(bank_id, "freeze")
    await _retain(client, llm, bank_id, "canary", CANARY)
    await settled(bank_id)

    # The removal lands; the new-facts edit names a section the page does not have, so
    # every one of its operations is rejected.
    llm.on_step("mental_model_retraction").answers_with(remove_block_saying(FRIDAY))
    llm.on_step("mental_model_delta").returns(
        DeltaOperations(operations=[RemoveBlock(section_id="no-such-section", block_id="no-such-block")])
    )
    await client.mental_models.refresh_mental_model(bank_id, model)
    # The failed refresh is this story's subject.
    await wait_until_settled(client, bank_id, allow_failed=True)

    assert await _refreshes(client, bank_id, status="failed") == 1
    assert await _content(client, bank_id, model) == PAGE, "the removal alone was written"

    llm.reset()
    reflect_loop(llm, answer=CANARY, query=QUERY)
    llm.on_step("mental_model_retraction").answers_with(remove_block_saying(FRIDAY))
    llm.on_step("mental_model_delta").returns(add_section("Canary", CANARY))
    await client.mental_models.refresh_mental_model(bank_id, model)
    await wait_until_settled(client, bank_id, allow_failed=True)

    assert await _refreshes(client, bank_id, status="failed") == 1, "the retry failed too"

    retractions = llm.prompts_for("mental_model_retraction")
    edits = llm.prompts_for("mental_model_delta")
    assert len(retractions) == 1 and FRIDAY in retractions[0], "the retry did not redo the removal"
    assert len(edits) == 1 and CANARY in edits[0], "the retry did not see the new memory again"
    assert await _content(client, bank_id, model) == f"## Checkout deploys\n\n{REVIEWER}\n\n## Canary\n\n{CANARY}"

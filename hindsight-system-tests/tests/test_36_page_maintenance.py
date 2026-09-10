"""A page and its model stay in step, and a retraction reaches both.

A knowledge page is a name and a place in a tree over a mental model that does
the work. Two things have to hold as the bank changes underneath them.

**A rename reaches the model.** They are separate rows, so renaming the page and
leaving the model on its old name gives one thing two names — and the model's
name is what shows up wherever a page is referenced by its backing id, so the two
disagree in exactly the places nobody is looking.

**A retraction propagates.** This is the sharper one. Facts are retracted because
they were wrong, and a page written from them keeps repeating the wrong thing
until something refreshes it. Nothing about that page looks stale: it has content,
a timestamp, and an answer that reads perfectly well. The system has to notice on
its own, because a caller who knew to ask would not have needed the page.
"""

from __future__ import annotations

import asyncio

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import extracted, fact, observes

pytestmark = pytest.mark.asyncio

ORIGINAL_ANSWER = "Alice lives in Berlin."
CORRECTED_ANSWER = "There is nothing on record about where Alice lives."


@pytest.fixture
async def page(client, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    # A page refreshes by delta over the *observation* layer (story 35 pins those
    # defaults), so a bank with no observations gives its refresh nothing to read
    # and the page keeps its placeholder — which looks like a broken refresh and
    # is not.
    llm.on_step("consolidate").answers_with(observes("Alice is settled in Berlin"))
    reflect_loop(llm, answer=ORIGINAL_ANSWER)

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)

    created = await client.knowledge_base.create_knowledge_page(
        bank_id, {"name": "Housing", "source_query": "Where does Alice live?"}
    )
    await settled(bank_id)
    return created


async def test_the_page_starts_with_the_answer_it_was_written_from(client, bank_id, page):
    read = await client.knowledge_base.get_knowledge_page(bank_id, page.page_id)

    assert read.name == "Housing"
    assert read.body.strip() == ORIGINAL_ANSWER


async def test_renaming_a_page_renames_its_model(client, bank_id, page):
    """Otherwise one thing has two names, and which you see depends on whether
    you arrived via the tree or via the model id."""
    await client.knowledge_base.update_knowledge_node(bank_id, page.page_id, {"name": "Where Alice lives"})

    model = await client.mental_models.get_mental_model(bank_id, page.mental_model_id, detail="metadata")
    assert model.name == "Where Alice lives"

    tree = await client.knowledge_base.get_knowledge_base_tree(bank_id)
    assert [node.name for node in tree.roots] == ["Where Alice lives"]


async def test_retracting_the_evidence_schedules_the_page_to_be_rewritten(client, llm, bank_id, page, settled):
    """Invalidating the fact the page was written from must enqueue a refresh.

    Without it the page keeps asserting something the bank has explicitly
    retracted — and reads as authoritative while doing so, since a page carries
    no hint of how old its evidence is.
    """
    memories = await client.memory.list_memories(bank_id, limit=100)
    target = next(m for m in memories.items if m.fact_type != "observation")

    await client.memory.update_memory(bank_id, target.id, {"state": "invalidated", "reason": "Alice never moved"})

    refreshed = False
    for _ in range(60):
        operations = await client.operations.list_operations(bank_id, type="refresh_mental_model", limit=100)
        if operations.total:
            refreshed = True
            break
        await asyncio.sleep(1)

    assert refreshed, "retracting the evidence scheduled no refresh — the page will keep repeating it"


async def test_the_scheduled_refresh_completes_cleanly(client, llm, bank_id, page, settled):
    """The refresh runs to completion and leaves the page coherent.

    What is *not* asserted here: that the retracted claim is gone from the text.
    A page refreshes in delta mode — it computes a diff against its current
    content rather than rewriting from scratch — so what the page ends up saying
    is a judgement the model makes about that diff. Driving it from a stub would
    mean writing the delta myself and then asserting I had written it, which
    proves nothing about the product.

    The mechanism is testable and is: the retraction is noticed, a refresh is
    scheduled, it completes without error, and the page is still readable
    afterwards. Whether the wording actually unsays the claim needs a real model
    and belongs with the `hs_llm_core` judge tests.
    """
    memories = await client.memory.list_memories(bank_id, limit=100)
    target = next(m for m in memories.items if m.fact_type != "observation")

    await client.memory.update_memory(bank_id, target.id, {"state": "invalidated", "reason": "Alice never moved"})
    await settled(bank_id)

    failed = await client.operations.list_operations(bank_id, status="failed", limit=100)
    assert failed.operations == [], "the refresh triggered by the retraction failed"

    read = await client.knowledge_base.get_knowledge_page(bank_id, page.page_id)
    assert read.body.strip(), "the page was left empty by the refresh"

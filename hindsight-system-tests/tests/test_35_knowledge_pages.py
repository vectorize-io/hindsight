"""A knowledge page is a mental model with a place in a tree and a name people use.

Pages are the human-facing surface over mental models: same self-maintaining
answer underneath, arranged like a wiki so someone can browse and search it
rather than knowing a model id.

The relationship is the part worth pinning. A page is *backed by* a model — it
has its own id and also names the model's — and the two must stay in step,
because a page that loses its backing model is a title with nothing behind it
and a model orphaned from its page is invisible.

Note the defaults a page picks up that a bare model does not: it refreshes on a
delta after consolidation, over observations rather than raw facts. That is a
deliberate difference in what a page is *for* — a curated summary, not a live
query — and it is the kind of default a refactor silently flattens.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

PAGE_NAME = "Where Alice lives"
SOURCE_QUERY = "Where does Alice live?"
ANSWER = "Alice lives in Berlin."


@pytest.fixture
async def bank_with_page(client, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    reflect_loop(llm, answer=ANSWER)
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)

    created = await client.knowledge_base.create_knowledge_page(
        bank_id, {"name": PAGE_NAME, "source_query": SOURCE_QUERY}
    )
    await settled(bank_id)
    return created


async def test_a_page_is_backed_by_a_mental_model(client, bank_id, bank_with_page):
    """Both ids are returned, and both resolve. A page whose model id dangles is
    a title with nothing behind it."""
    assert bank_with_page.page_id.startswith("kp-")
    assert bank_with_page.mental_model_id.startswith("mm-")

    model = await client.mental_models.get_mental_model(bank_id, bank_with_page.mental_model_id, detail="full")
    assert model.id == bank_with_page.mental_model_id


async def test_the_page_appears_in_the_tree_under_the_name_it_was_given(client, bank_id, bank_with_page):
    """The tree is how a person finds a page, so the name and the wiring have to
    be in it — not just retrievable by an id the person does not have."""
    tree = await client.knowledge_base.get_knowledge_base_tree(bank_id)

    roots = {node.id: node for node in tree.roots}
    page = roots[bank_with_page.page_id]
    assert page.name == PAGE_NAME
    assert page.kind == "page"
    assert page.parent_id is None
    assert page.mental_model_id == bank_with_page.mental_model_id
    assert page.description == SOURCE_QUERY


async def test_a_page_refreshes_over_observations_by_delta(client, bank_id, bank_with_page):
    """The defaults that distinguish a page from a bare mental model.

    A page is a curated summary that follows consolidation, so it updates
    incrementally and reads the observation layer rather than every raw fact.
    Flattening these to a model's defaults would quietly turn every page into a
    full re-read of the whole bank.
    """
    tree = await client.knowledge_base.get_knowledge_base_tree(bank_id)
    page = next(node for node in tree.roots if node.id == bank_with_page.page_id)

    assert page.trigger.mode == "delta"
    assert page.trigger.refresh_after_consolidation is True
    assert page.trigger.fact_types == ["observation"]
    assert page.trigger.exclude_mental_models is True


async def test_a_page_is_findable_by_searching_its_content(client, bank_id, bank_with_page):
    """Searchable is the whole reason for the wiki framing: someone who does not
    know a page exists has to be able to land on it."""
    found = await client.knowledge_base.search_knowledge_base(bank_id, q="Alice")

    assert [result.id for result in found.results] == [bank_with_page.page_id]
    assert found.results[0].name == PAGE_NAME


async def test_deleting_a_page_leaves_the_facts_alone(client, bank_id, bank_with_page, settled):
    """Like every other derived layer: the page can go, the memories cannot."""
    await client.knowledge_base.delete_knowledge_node(bank_id, bank_with_page.page_id)
    await settled(bank_id)

    tree = await client.knowledge_base.get_knowledge_base_tree(bank_id)
    assert [node.id for node in tree.roots] == []

    memories = await client.memory.list_memories(bank_id, limit=100)
    assert [m.text for m in memories.items] == ["Alice moved to Berlin | Involving: Alice"]

"""Entities are what stitch separate documents into one memory.

Every fact names some entities. When two documents retained months apart both
mention Alice, they are joined through that one entity — and the graph arm of
recall can walk it. That walk is what makes a memory system different from a
pile of independently-searchable documents: a question that matches one fact can
surface a related fact that shares nobody's words, only its subject.

Story 02 asserted the graph arm returns nothing for a single document, because
there is nowhere to walk. This is the other half: two documents, one shared
entity, and a query whose wording appears in only one of them.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

BERLIN = "Alice moved to Berlin | Involving: Alice"
CELLO = "Alice plays cello | Involving: Alice"


@pytest.fixture
async def linked_bank(client, llm, bank_id, settled) -> str:
    """Two documents that share only the entity "Alice"."""
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("extract_facts", contains="cello").returns(
        extracted(fact("Alice plays cello", who="Alice", entities=["Alice", "cello"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.", document_id="d-move")
    await client.aretain(bank_id=bank_id, content="Alice plays cello.", document_id="d-music")
    await settled(bank_id)
    return bank_id


async def test_every_named_entity_is_recorded(client, linked_bank):
    listing = await client.entities.list_entities(linked_bank)

    assert listing.total == 3
    assert sorted(e.canonical_name for e in listing.items) == ["Alice", "Berlin", "cello"]


async def test_an_entity_named_by_two_documents_is_one_entity(client, linked_bank):
    """Not two rows called "Alice". The mention count is the evidence that the
    second document was recognised as being about the same person — without that
    merge there is no path between the documents to walk."""
    listing = await client.entities.list_entities(linked_bank)
    alice = next(e for e in listing.items if e.canonical_name == "Alice")

    assert alice.mention_count == 2
    assert alice.first_seen < alice.last_seen, "the second mention did not land on the same entity"

    for name in ("Berlin", "cello"):
        entity = next(e for e in listing.items if e.canonical_name == name)
        assert entity.mention_count == 1


async def test_the_graph_reaches_a_fact_that_shares_no_words_with_the_query(client, linked_bank):
    """The payoff, and the case story 02 could not exercise.

    "cello" appears in one document and nowhere in the other. Semantic and
    keyword search can only find that one. The Berlin fact comes back solely
    because it hangs off the same entity — remove the traversal and it vanishes
    while every other test still passes.
    """
    response = await client.arecall(bank_id=linked_bank, query="cello", trace=True)

    by_arm = {
        (arm["method_name"], arm["fact_type"]): [r["text"] for r in arm["results"]]
        for arm in response.trace["retrieval_results"]
    }
    assert by_arm[("graph", "world")] == [BERLIN], "the graph arm found nothing to traverse to"

    assert sorted(r.text for r in response.results) == sorted([CELLO, BERLIN])


async def test_the_entity_graph_exposes_the_link_a_traversal_follows(client, linked_bank):
    """The same structure, readable directly — so someone debugging a recall can
    see whether the path exists rather than inferring it from what came back."""
    graph = await client.entities.get_entity_graph(linked_bank, limit=50)

    labels = {node.data.label for node in graph.nodes}
    assert labels == {"Alice", "Berlin", "cello"}
    assert graph.edges, "co-occurring entities produced no edges"

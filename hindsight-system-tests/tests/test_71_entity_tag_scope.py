"""A tag filter on the entity reads hides what other scopes wrote about (#5031).

Tags are how one bank holds several scopes — one per end user, say — and recall
already honours them. The entity reads did not: a reader asking only for
`user:dan` still got `Kate` and `billing service` back, which exist only in
Kate's memories, with mention counts that included Kate's. No fact text leaked,
but the names and the counts told Dan what Kate's scope talks about.

The interesting entity is Alice, whom both scopes mention. Hiding the entities
only Kate names is not enough on its own: if Alice came back with her bank-wide
count, the number would still give away how much Kate says about her. So the
counts, the graph edges and the detail view are all checked under the filter,
next to the unfiltered read that must stay as it was.
"""

from __future__ import annotations

import json

import pytest
from hindsight_client_api.exceptions import NotFoundException

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

DAN = ["user:dan"]
KATE = ["user:kate"]


@pytest.fixture
async def scoped_bank(client, llm, bank_id, settled) -> str:
    """Dan's two memories name Alice and Bob; Kate's one names Alice, Kate and the billing service."""
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice"]))
    )
    llm.on_step("extract_facts", contains="chess").returns(
        extracted(fact("Alice plays chess with Bob", who="Alice", entities=["Alice", "Bob"]))
    )
    llm.on_step("extract_facts", contains="billing").returns(
        extracted(
            fact("Kate asked Alice about the billing service", who="Kate", entities=["Alice", "Kate", "billing service"])
        )
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.", tags=DAN)
    await client.aretain(bank_id=bank_id, content="Alice plays chess with Bob.", tags=DAN)
    await client.aretain(bank_id=bank_id, content="Kate asked Alice about the billing service.", tags=KATE)
    await settled(bank_id)
    return bank_id


def _counts(listing) -> dict[str, int]:
    return {e.canonical_name: e.mention_count for e in listing.items}


async def test_the_list_holds_only_what_the_scope_mentions_with_its_own_counts(client, scoped_bank):
    listing = await client.entities.list_entities(scoped_bank, tags=DAN, tags_match="any_strict")

    # Alice is 2, not the bank-wide 3: Kate's mention is not counted either.
    assert _counts(listing) == {"Alice": 2, "Bob": 1}
    assert listing.total == 2


async def test_the_unfiltered_list_still_sees_the_whole_bank(client, scoped_bank):
    listing = await client.entities.list_entities(scoped_bank)

    assert _counts(listing) == {"Alice": 3, "Bob": 1, "Kate": 1, "billing service": 1}
    assert listing.total == 4


async def test_an_entity_only_another_scope_names_is_a_404(client, scoped_bank):
    """Not a 403: "forbidden" would confirm Kate exists somewhere in the bank."""
    everyone = await client.entities.list_entities(scoped_bank)
    ids = {e.canonical_name: e.id for e in everyone.items}

    with pytest.raises(NotFoundException):
        await client.entities.get_entity(scoped_bank, ids["Kate"], tags=DAN, tags_match="any_strict")

    alice = await client.entities.get_entity(scoped_bank, ids["Alice"], tags=DAN, tags_match="any_strict")
    assert alice.mention_count == 2

    # The same entity is readable from the scope that wrote it.
    kate = await client.entities.get_entity(scoped_bank, ids["Kate"], tags=KATE, tags_match="any_strict")
    assert kate.mention_count == 1


async def test_the_graph_draws_only_the_scope_s_own_co_occurrences(client, scoped_bank):
    """The stored co-occurrence table has Alice–Kate and Alice–billing edges too;
    the filtered graph is rebuilt from Dan's memories, so only Alice–Bob is left."""
    graph = await client.entities.get_entity_graph(scoped_bank, tags=DAN, tags_match="any_strict")

    assert {n.data.label: n.data.mention_count for n in graph.nodes} == {"Alice": 2, "Bob": 1}
    assert len(graph.edges) == 1
    assert graph.edges[0].data.weight == 1

    unfiltered = await client.entities.get_entity_graph(scoped_bank)
    assert {n.data.label for n in unfiltered.nodes} == {"Alice", "Bob", "Kate", "billing service"}


async def test_a_compound_tag_group_scopes_the_reads_the_same_way(client, scoped_bank):
    """`tag_groups` travels as JSON in the query string; "everything but Kate's" is Dan's view here."""
    not_kate = json.dumps([{"not": {"tags": KATE, "match": "any_strict"}}])

    listing = await client.entities.list_entities(scoped_bank, tag_groups=not_kate)
    assert _counts(listing) == {"Alice": 2, "Bob": 1}

    graph = await client.entities.get_entity_graph(scoped_bank, tag_groups=not_kate)
    assert {n.data.label for n in graph.nodes} == {"Alice", "Bob"}

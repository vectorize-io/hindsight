"""Entity links follow the facts, including when the facts go away.

Entities are derived from facts, so replacing or deleting a document has to move
them: an entity only the old text named should disappear, one only the new text
names should appear, and the traversal paths that hang off them should follow.

The count has to follow too. `mention_count` is what a caller sees on the entity
list, what sizes a node in the graph, and what the curation listing sorts by —
and until #4291 it was only ever incremented, so a bank that updated documents
accumulated entities whose stated prominence had no relationship to how many
facts actually mentioned them. The drift scaled with how often documents
change, and the coding-agents integration re-retains under one `document_id`
every turn.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

BERLIN = "Alice moved to Berlin | Involving: Alice"
LISBON = "Alice moved to Lisbon | Involving: Alice"
CELLO = "Alice plays cello | Involving: Alice"


@pytest.fixture
async def two_documents(client, llm, bank_id, settled) -> str:
    """`d1` names Alice and Berlin, `d2` names Alice and cello."""
    llm.on_step("extract_facts", contains="Berlin").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("extract_facts", contains="cello").returns(
        extracted(fact("Alice plays cello", who="Alice", entities=["Alice", "cello"]))
    )
    llm.on_step("extract_facts", contains="Lisbon").returns(
        extracted(fact("Alice moved to Lisbon", who="Alice", entities=["Alice", "Lisbon"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.", document_id="d1")
    await client.aretain(bank_id=bank_id, content="Alice plays cello.", document_id="d2")
    await settled(bank_id)
    return bank_id


async def _names(client, bank: str) -> list[str]:
    listing = await client.entities.list_entities(bank)
    return sorted(e.canonical_name for e in listing.items)


async def _replace_d1(client, bank: str, settled) -> None:
    await client.aretain(bank_id=bank, content="Alice moved to Lisbon.", document_id="d1", update_mode="replace")
    await settled(bank)


async def test_an_entity_only_the_old_text_named_is_gone(client, two_documents, settled):
    assert "Berlin" in await _names(client, two_documents)

    await _replace_d1(client, two_documents, settled)

    names = await _names(client, two_documents)
    assert "Berlin" not in names
    assert "Lisbon" in names
    # `cello` came from the untouched document and must be unaffected.
    assert "cello" in names


async def test_the_links_themselves_follow_the_replacement(client, two_documents, settled):
    """Not just the entity list — what each entity is actually attached to. An
    entity whose links point at replaced facts is a dangling path the graph arm
    can still walk."""
    await _replace_d1(client, two_documents, settled)

    listing = await client.entities.list_entities(two_documents)
    alice = next(e for e in listing.items if e.canonical_name == "Alice")

    linked = await client.memory.list_memories(two_documents, entity_id=alice.id, limit=100)
    assert sorted(m.text for m in linked.items) == sorted([LISBON, CELLO])


async def test_traversal_follows_the_replacement(client, two_documents, settled):
    """End to end: the graph arm reaches the *new* fact through the shared
    entity, and cannot reach the replaced one."""
    await _replace_d1(client, two_documents, settled)

    response = await client.arecall(bank_id=two_documents, query="cello", trace=True)
    reached = [
        r["text"]
        for arm in response.trace["retrieval_results"]
        if arm["method_name"] == "graph"
        for r in arm["results"]
    ]

    assert reached == [LISBON]


async def test_deleting_a_document_removes_the_entities_only_it_named(client, two_documents, settled):
    await client.documents.delete_document(two_documents, "d1")
    await settled(two_documents)

    names = await _names(client, two_documents)
    assert "Berlin" not in names
    assert names == sorted(["Alice", "cello"])


async def test_the_mention_count_matches_the_facts_that_mention_it(client, two_documents, settled):
    """The count comes back down when the mentions go (#4291).

    Alice is named by exactly two facts after the replacement, and the link table
    agrees. The denormalised counter used to disagree — incremented for the
    replaced fact and never given back — which was invisible unless you compared
    the two, as this does.

    Asserted against the links rather than a hardcoded number, so this keeps
    holding whatever the fixture grows into.
    """
    await _replace_d1(client, two_documents, settled)

    listing = await client.entities.list_entities(two_documents)
    alice = next(e for e in listing.items if e.canonical_name == "Alice")
    linked = await client.memory.list_memories(two_documents, entity_id=alice.id, limit=100)

    assert alice.mention_count == linked.total, (
        f"mention_count says {alice.mention_count}, but {linked.total} facts mention Alice"
    )

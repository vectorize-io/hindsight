"""A bank can be moved to another instance and still make sense on arrival.

Export/import is how a bank changes home — a migration, a backup, a copy into
staging. It is also the operation with the most ways to arrive subtly wrong,
because everything in a bank points at everything else by id: facts belong to
documents, observations cite facts, mental models cite observations. Every one
of those references is minted by the *source* instance and has to be rewritten
to the ids the *destination* minted, in one pass, without missing a layer.

Miss a layer and the import still succeeds. The bank arrives, the counts look
right, and only the provenance is broken — an observation citing fact ids that
do not exist here, a mental model whose evidence resolves to nothing. That is
the shape of the bug that has been fixed twice on this path, which is why the
assertions below chase the references rather than the counts.

Also pinned: the derived layers are *opt-in*. Observations and knowledge pages
are expensive to carry and reconstructible from the facts, so a plain export
leaves them out — and a caller who did not ask must not silently pay for them.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
from hindsight_client_api.api.document_transfer_api import DocumentTransferApi

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact, observes

pytestmark = pytest.mark.asyncio

BERLIN = "Alice moved to Berlin | Involving: Alice"
LEASE = "Alice renewed her Berlin lease | Involving: Alice"
OBSERVATION = "Alice is settled in Berlin"
PAGE_NAME = "Where Alice lives"


@pytest.fixture
async def source_bank(client, llm, bank_id, settled) -> str:
    """A bank with all three layers: facts, an observation over them, and a page."""
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice renewed her Berlin lease", who="Alice", entities=["Alice", "Berlin"]),
        )
    )
    llm.on_step("consolidate").answers_with(observes(OBSERVATION))
    reflect_loop(llm, answer="Alice lives in Berlin.")

    await client.aretain(
        bank_id=bank_id, content="Alice moved to Berlin. Alice renewed her Berlin lease.", document_id="d1"
    )
    await settled(bank_id)
    await client.knowledge_base.create_knowledge_page(
        bank_id, {"name": PAGE_NAME, "source_query": "Where does Alice live?"}
    )
    await settled(bank_id)
    return bank_id


@pytest.fixture
async def destination(client) -> AsyncIterator[str]:
    bank = f"systest-{uuid.uuid4().hex[:12]}"
    yield bank
    await client.banks.delete_bank(bank)


async def _import(client, llm, bank: str, archive: bytes) -> None:
    """Import the archive into ``bank``, with the destination's own consolidation
    silenced first.

    Importing facts is not an extraction — the docs are explicit that no LLM
    re-extraction happens — but it does leave the destination holding new facts,
    and its auto-consolidation then runs over them like any other write. Left
    alone, an observation appearing in the destination would prove nothing: it
    might have been carried by the archive or invented locally a second later.
    Silencing it makes every observation below necessarily an imported one.
    """
    llm.reset()
    llm.on_step("consolidate").returns(consolidation())

    api = DocumentTransferApi(client.documents.api_client)
    await api.import_documents(bank, archive)


async def _memories(client, bank: str) -> list[dict]:
    return (await client.memory.list_memories(bank, limit=100)).items


async def test_the_facts_arrive(client, llm, source_bank, destination, settled):
    archive = await client.aexport_documents(bank_id=source_bank)
    await _import(client, llm, destination, archive)
    await settled(destination)

    texts = sorted(m.text for m in await _memories(client, destination))
    assert texts == sorted([BERLIN, LEASE])


async def test_the_derived_layers_are_left_behind_unless_asked_for(client, llm, source_bank, destination, settled):
    """Opt-in, both ways. An export that silently carried everything would make
    a "just the documents" migration quietly expensive, and would import stale
    syntheses into a bank that is about to rebuild them anyway."""
    archive = await client.aexport_documents(bank_id=source_bank)
    await _import(client, llm, destination, archive)
    await settled(destination)

    assert [m for m in await _memories(client, destination) if m.fact_type == "observation"] == []
    assert (await client.knowledge_base.get_knowledge_base_tree(destination)).roots == []


async def test_observations_come_across_when_requested(client, llm, source_bank, destination, settled):
    archive = await client.aexport_documents(bank_id=source_bank, include_observations=True)
    await _import(client, llm, destination, archive)
    await settled(destination)

    observations = [m for m in await _memories(client, destination) if m.fact_type == "observation"]
    assert [o.text for o in observations] == [OBSERVATION]


async def test_an_imported_observation_still_points_at_its_evidence(client, llm, source_bank, destination, settled):
    """The reference-rewriting assertion, and the reason this story exists.

    The observation cites fact ids minted by the *source*. Those ids do not exist
    here — the import minted new ones — so every citation has to be rewritten. An
    import that copies them verbatim produces exactly what this checks for: an
    observation whose evidence resolves to nothing, in a bank that otherwise looks
    perfectly healthy.
    """
    archive = await client.aexport_documents(bank_id=source_bank, include_observations=True)
    await _import(client, llm, destination, archive)
    await settled(destination)

    memories = await _memories(client, destination)
    by_id = {m.id: m for m in memories}
    observation = next(m for m in memories if m.fact_type == "observation")

    assert observation.source_memory_ids, "the imported observation lost its evidence entirely"
    for source_id in observation.source_memory_ids:
        assert source_id in by_id, "an imported observation cites a fact id that does not exist in this bank"
        assert by_id[source_id].fact_type != "observation"


async def test_knowledge_pages_come_across_when_requested(client, llm, source_bank, destination, settled):
    archive = await client.aexport_documents(
        bank_id=source_bank, include_observations=True, include_knowledge_base=True
    )
    await _import(client, llm, destination, archive)
    await settled(destination)

    tree = await client.knowledge_base.get_knowledge_base_tree(destination)
    assert [node.name for node in tree.roots] == [PAGE_NAME]

    # And the page's backing model exists *here*, rather than naming one that was
    # left behind on the source instance.
    page = tree.roots[0]
    model = await client.mental_models.get_mental_model(destination, page.mental_model_id, detail="full")
    assert model.id == page.mental_model_id


async def test_the_imported_bank_answers_recalls(client, llm, source_bank, destination, settled):
    """The end-to-end proof. Rows arriving is not the same as a working bank —
    embeddings and search indexes have to be rebuilt for the destination too."""
    archive = await client.aexport_documents(bank_id=source_bank)
    await _import(client, llm, destination, archive)
    await settled(destination)

    response = await client.arecall(bank_id=destination, query="Where does Alice live?")
    assert BERLIN in [r.text for r in response.results]


async def test_importing_does_not_touch_the_source(client, llm, source_bank, destination, settled):
    """A migration is a copy until someone says otherwise."""
    archive = await client.aexport_documents(bank_id=source_bank, include_observations=True)
    await _import(client, llm, destination, archive)
    await settled(destination)

    texts = sorted(m.text for m in await _memories(client, source_bank))
    assert texts == sorted([BERLIN, LEASE, OBSERVATION])

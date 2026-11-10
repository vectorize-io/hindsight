"""An append that fails part-way keeps what it already extracted, and the retry finishes it.

A conversation document grows one append at a time. An append re-chunks the whole stored text and
extracts every chunk that is not stored yet — its own new turn, plus any chunk an earlier failed
retain left unwritten. That work used to be extracted in one step and written only at the end, so
an append that failed (or hit the retain timeout) threw all of it away, and the next append started
the same work from zero. On a large conversation the document stopped growing for good (#6394).

Now the work is written in batches as it is extracted. These stories use a batch of one chunk, so
each extracted paragraph is saved on its own.
"""

from __future__ import annotations

import pytest
from hindsight_client_api.exceptions import ServiceException

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

DOCUMENT_ID = "trip-alice"
CHUNK_SIZE = 200

# One paragraph per chunk: each is well over half the chunk size, so two never share one.
FIRST = "Alice flew into Lisbon on a grey Monday morning, took the slow tram up the hill past the old cathedral, and found her rented flat on the third floor."
TURNS = [
    "Alice took the early train north to Porto on the Tuesday, watched the river boats from the bridge at dusk, and ate grilled sardines at a corner table.",
    "Alice drove a small rented car east to Braga on the Wednesday, climbed the long staircase to the sanctuary on the hill, and stayed until the bells rang.",
    "Alice caught the overnight bus south to Faro on the Thursday, slept through most of the ride, and spent the morning walking along the quiet lagoon shore.",
]
CITIES = ["Lisbon", "Porto", "Braga", "Faro"]


def _memory(city: str) -> str:
    return f"Alice visited {city} | Involving: Alice"


def _extracts(llm, city: str) -> None:
    llm.on_step("extract_facts", contains=city).returns(
        extracted(fact(f"Alice visited {city}", who="Alice", entities=["Alice", city]))
    )


async def _fact_texts(client, bank: str) -> list[str]:
    memories = await client.memory.list_memories(bank, limit=100)
    return sorted(item.text for item in memories.items if item.state == "valid")


@pytest.fixture
async def started(client, llm, bank_id, settled):
    await client.acreate_bank(bank_id=bank_id, name="Alice")
    await client.banks.update_bank_config(
        bank_id, {"updates": {"retain_chunk_size": CHUNK_SIZE, "retain_chunk_batch_size": 1}}
    )
    llm.on_step("consolidate").returns(consolidation())
    _extracts(llm, "Lisbon")
    await client.aretain(bank_id=bank_id, content=FIRST, document_id=DOCUMENT_ID)
    await settled(bank_id)
    return bank_id


async def test_a_failed_append_keeps_its_saved_paragraphs_and_the_retry_finishes_it(client, llm, started, settled):
    _extracts(llm, "Porto")
    _extracts(llm, "Braga")
    llm.on_step("extract_facts", contains="Faro").fails_with_status(503)
    tail = "\n\n".join(TURNS)

    with pytest.raises(ServiceException):
        await client.aretain(bank_id=started, content=tail, document_id=DOCUMENT_ID, update_mode="append")
    await settled(started)

    # What it extracted before the failure is saved; the document's text waits for the whole append.
    assert await _fact_texts(client, started) == sorted(_memory(c) for c in ["Lisbon", "Porto", "Braga"])
    document = await client.documents.get_document(started, DOCUMENT_ID)
    assert document.original_text == FIRST, "a failed append must not change the document's text"

    # The model is back. The retry extracts only the paragraph the failure left, and lands once.
    llm.reset()
    llm.on_step("consolidate").returns(consolidation())
    _extracts(llm, "Faro")
    await client.aretain(bank_id=started, content=tail, document_id=DOCUMENT_ID, update_mode="append")
    await settled(started)

    prompts = llm.prompts_for("extract_facts")
    assert len(prompts) == 1 and "Faro" in prompts[0], "the retry re-extracted paragraphs already saved"
    document = await client.documents.get_document(started, DOCUMENT_ID)
    assert document.original_text.count(TURNS[-1]) == 1 and document.original_text.startswith(FIRST)
    chunks = await client.documents.list_document_chunks(started, DOCUMENT_ID)
    assert chunks.total == 4, "the story needs one chunk per paragraph to cover anything"
    assert await _fact_texts(client, started) == sorted(_memory(c) for c in CITIES)

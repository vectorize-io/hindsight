"""Long content is split, and every piece is extracted from exactly once.

A document larger than the chunk size is cut up before extraction. Three things
have to hold, and each fails differently.

Every chunk must be *extracted from* — a splitter that drops the tail loses the
end of every long document, and the loss is invisible because the document
still stores the full text. Every chunk must be extracted from **once** — a
double pass duplicates facts and doubles the model spend.

And the stored document must remain byte-identical to what was sent, because it
is the input a reprocess re-reads: a document that persisted only its last slice
would silently shrink on the next re-extraction. That is not hypothetical — it
is the shape of a bug this project has already hit.
"""

from __future__ import annotations

import re

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

CHUNK_SIZE = 200

# Each sentence carries its own index, so a fact can name the piece it came from
# and a dropped or repeated chunk is visible by its absence or duplication.
SENTENCES = [f"Marker{i:03d} is a sentence about Alice and Berlin and the cello." for i in range(60)]
CONTENT = " ".join(SENTENCES)

_MARKER = re.compile(r"Marker(\d{3})")


@pytest.fixture
async def chunked_document(client, llm, bank_id, settled) -> str:
    await client.acreate_bank(bank_id=bank_id, name="Alice")
    await client.banks.update_bank_config(bank_id, {"updates": {"retain_chunk_size": CHUNK_SIZE}})

    def one_fact_naming_the_first_marker(request):
        """Answer each extraction call with a fact naming the chunk it was shown.

        The stub cannot know how the server split the text, so it reads the first
        marker out of whatever it was handed — which makes the set of facts a
        direct readout of which pieces reached the model.
        """
        markers = _MARKER.findall(request.all_text)
        return extracted(fact(f"Saw Marker{markers[0]}", who="Alice", entities=["Alice"]))

    llm.on_step("extract_facts").answers_with(one_fact_naming_the_first_marker)
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content=CONTENT, document_id="d1")
    await settled(bank_id)
    return bank_id


async def test_the_document_is_split_into_several_chunks(client, chunked_document):
    chunks = await client.documents.list_document_chunks(chunked_document, "d1")

    assert chunks.total > 1, "content well over the chunk size was not split at all"


async def test_the_chunks_are_numbered_without_gaps(client, chunked_document):
    """Contiguous from zero. A gap means a chunk was dropped between splitting
    and storing, and nothing else in the response would show it."""
    chunks = await client.documents.list_document_chunks(chunked_document, "d1")

    indexes = sorted(chunk.chunk_index for chunk in chunks.items)
    assert indexes == list(range(len(indexes)))


async def test_the_chunks_reassemble_into_the_original(client, chunked_document):
    """No text is lost at the seams.

    Asserted over the concatenation rather than sentence by sentence: the
    splitter cuts on size, not on sentence boundaries, so an individual sentence
    can legitimately straddle two chunks and appear whole in neither. What must
    hold is that putting the pieces back together yields every marker, in order,
    exactly once.
    """
    chunks = await client.documents.list_document_chunks(chunked_document, "d1")
    ordered = sorted(chunks.items, key=lambda chunk: chunk.chunk_index)

    reassembled = "".join(chunk.chunk_text for chunk in ordered)
    assert _MARKER.findall(reassembled) == _MARKER.findall(CONTENT)


async def test_every_chunk_reached_the_model_exactly_once(client, llm, chunked_document):
    """The assertion the markers exist for.

    One extraction call per chunk: fewer means a piece was never read, more means
    the model was paid for the same text twice. Both look like a working retain.
    """
    chunks = await client.documents.list_document_chunks(chunked_document, "d1")
    prompts = llm.prompts_for("extract_facts")

    assert len(prompts) == chunks.total

    first_markers = [_MARKER.findall(prompt)[0] for prompt in prompts]
    assert len(set(first_markers)) == len(first_markers), "a chunk was extracted from twice"


async def test_the_stored_document_is_byte_identical_to_what_was_sent(client, chunked_document):
    """Chunking is for extraction, not storage. The document is what a reprocess
    re-reads, so a version that kept only one slice would quietly shrink the
    bank the next time it was re-extracted."""
    document = await client.documents.get_document(chunked_document, "d1")

    assert document.original_text == CONTENT


async def test_facts_from_every_part_of_the_document_are_recallable(client, chunked_document):
    """End to end: a marker from the last chunk is as findable as one from the
    first. A dropped tail is only visible from the far end of the document."""
    last_marker = _MARKER.findall(SENTENCES[-1])[0]

    memories = await client.memory.list_memories(chunked_document, limit=200)
    seen = {marker for item in memories.items for marker in _MARKER.findall(item.text)}

    assert last_marker in seen or any(int(m) > 40 for m in seen), (
        "no fact came from the end of the document — the tail was never extracted"
    )

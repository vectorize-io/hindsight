"""The same document retained twice at once keeps each fact once (issue #6337).

A client that retries a slow retain, or a task run again while its first run is
still writing, sends the same document with the same text twice. A long
document is written in batches, and both runs carry the same text, so a check
that compares the text cannot tell them apart: both used to write every batch
after the first, and every fact from those batches was stored twice.

The document must end complete either way: whichever run reaches a chunk first
stores it, and the other skips it.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

DOCUMENT_ID = "retried-doc"
SENTENCES = [f"Marker{i:03d} is a sentence about Alice and Berlin and the cello." for i in range(60)]
CONTENT = " ".join(SENTENCES)

_MARKER = re.compile(r"Marker(\d{3})")


async def test_two_identical_retains_at_once_store_each_fact_once(client, llm, bank_id, settled):
    await client.acreate_bank(bank_id=bank_id, name="Alice")
    # Small chunks, two per batch: a short document is then written in many batches.
    await client.banks.update_bank_config(
        bank_id, {"updates": {"retain_chunk_size": 200, "retain_chunk_batch_size": 2}}
    )

    def one_fact_per_chunk(request):
        markers = _MARKER.findall(request.all_text)
        return extracted(fact(f"Saw Marker{markers[0]}", who="Alice", entities=["Alice"]))

    llm.on_step("extract_facts").answers_with(one_fact_per_chunk)
    llm.on_step("consolidate").returns(consolidation())

    await asyncio.gather(
        *(client.aretain(bank_id=bank_id, content=CONTENT, document_id=DOCUMENT_ID) for _ in range(2))
    )
    await settled(bank_id)

    chunks = await client.documents.list_document_chunks(bank_id, DOCUMENT_ID)
    memories = await client.memory.list_memories(bank_id, limit=500)

    assert chunks.total > 2, "the document fit in one batch, so the race was never set up"
    # Exactly one fact per chunk, naming that chunk's first marker: a fact stored twice and a
    # chunk with no fact both break the equality.
    expected = sorted(f"Marker{_MARKER.findall(chunk.chunk_text)[0]}" for chunk in chunks.items)
    stored = sorted(f"Marker{marker}" for item in memories.items for marker in _MARKER.findall(item.text)[:1])
    assert stored == expected
    assert len(memories.items) == chunks.total

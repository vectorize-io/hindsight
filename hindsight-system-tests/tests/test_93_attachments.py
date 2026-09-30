"""An image retained inline reaches the model, and stays fetchable afterwards.

Content can be a list of blocks rather than a string, so a screenshot sits where
it actually appeared in the conversation. Two halves have to work, and they fail
differently.

The image has to reach the *model* — an attachment stored but never sent means
extraction describes text that references a picture it never saw, and the facts
are confidently wrong rather than missing. And it has to survive as a
**retrievable** attachment, because a fact drawn from an image is unverifiable
without it: "Alice stood in front of the Brandenburg Gate" is only auditable if
someone can still look at the photo.

The stub records what it was sent, so the first half is directly observable
rather than inferred from the facts that came back.
"""

from __future__ import annotations

import base64

import pytest
from hindsight_client_api.exceptions import NotFoundException
from hindsight_client_api.models.dry_run_extract_request import DryRunExtractRequest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

# A 1x1 transparent PNG — the smallest thing that is unambiguously an image.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_BASE64 = base64.b64encode(PNG).decode()

CAPTION = "Alice sent a photo from her trip:"
FACT = "Alice stood in front of the Brandenburg Gate | Involving: Alice"

CONTENT = [
    {"type": "text", "text": CAPTION},
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_BASE64}},
]


@pytest.fixture
async def bank_with_photo(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice stood in front of the Brandenburg Gate", who="Alice", entities=["Alice"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain_batch(bank_id=bank_id, items=[{"content": CONTENT}], document_id="d1")
    await settled(bank_id)
    return bank_id


def _parts(prompt_messages: list[dict]) -> list[str]:
    return [
        part.get("type")
        for message in prompt_messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict)
    ]


async def test_the_image_is_sent_to_the_extraction_model(client, llm, bank_with_photo):
    """Observed at the stub, not inferred.

    An attachment that is stored but never sent produces facts about a picture
    the model never saw — plausible, specific and wrong, which is worse than no
    facts at all.
    """
    calls = [call for call in llm.calls if any(a in call.all_text for a in [CAPTION])]
    assert calls, "the extraction call never carried the caption"

    kinds = _parts(calls[0].messages)
    assert "text" in kinds
    assert "image_url" in kinds, "the image was dropped between the API and the model"


async def test_the_caption_and_the_image_arrive_together(client, llm, bank_with_photo):
    """Order is the point of inline blocks. A photo hoisted to the end of the
    prompt loses which sentence it belonged to, which is most of its meaning in a
    conversation."""
    call = next(call for call in llm.calls if CAPTION in call.all_text)
    kinds = _parts(call.messages)

    assert kinds.index("text") < kinds.index("image_url")


async def test_the_attachment_is_stored_against_the_document(client, bank_with_photo):
    document = await client.documents.get_document(bank_with_photo, "d1")

    assert document.attachments, "the image was not kept"
    attachment = document.attachments[0]
    assert attachment.kind == "image"
    assert attachment.media_type == "image/png"
    assert attachment.byte_size == len(PNG)


async def test_the_attachment_hangs_off_the_chunk_it_appeared_in(client, bank_with_photo):
    """Provenance at the granularity extraction actually works at: a fact points
    at a chunk, so the chunk has to carry the image that fact was drawn from."""
    chunks = await client.documents.list_document_chunks(bank_with_photo, "d1")

    attachments = [a for chunk in chunks.items for a in (chunk.attachments or [])]
    assert [a.kind for a in attachments] == ["image"]


async def test_the_stored_image_is_byte_identical(client, bank_with_photo):
    """The audit path. A fact drawn from a picture is unverifiable unless the
    picture comes back exactly as sent.

    Fetched through the published client on purpose: until #4292 the endpoint
    declared `application/json` alongside `application/octet-stream`, and every
    generated SDK decoded the bytes as text and died on `0x89`, the first byte of
    the PNG magic number. The server was fine; only a client-driven test could
    see it.
    """
    document = await client.documents.get_document(bank_with_photo, "d1")
    attachment = document.attachments[0]

    fetched = await client.memory.get_bank_attachment(bank_with_photo, attachment.id)

    assert fetched == PNG


async def _attachment_id(client, bank: str, document_id: str) -> str:
    document = await client.documents.get_document(bank, document_id)
    assert document.attachments, f"{document_id} carries no attachment"
    return document.attachments[0].id


async def test_a_shared_image_stays_fetchable_until_its_last_document_goes(client, llm, bank_id, settled):
    """Two documents carrying the same photo: one identity, a stored copy each.

    The id is the content hash, so both documents name the same attachment, but
    each owns its own copy of the bytes. Deleting one must leave the photo
    downloadable for the other — reclaiming it there is data loss — and deleting
    the last must retire it, or it is kept forever behind an id nothing can reach.
    """
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice stood in front of the Brandenburg Gate", who="Alice", entities=["Alice"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content=CONTENT, document_id="d1")
    await client.aretain(bank_id=bank_id, content=CONTENT, document_id="d2")
    await settled(bank_id)
    attachment_id = await _attachment_id(client, bank_id, "d1")
    assert await _attachment_id(client, bank_id, "d2") == attachment_id

    await client.documents.delete_document(bank_id, "d1")
    assert await client.memory.get_bank_attachment(bank_id, attachment_id) == PNG

    await client.documents.delete_document(bank_id, "d2")
    with pytest.raises(NotFoundException):
        await client.memory.get_bank_attachment(bank_id, attachment_id)


async def test_re_retaining_without_the_image_retires_it(client, llm, bank_with_photo, settled):
    """#4364. The rewrite drops the document's reference; the image must go with it."""
    attachment_id = await _attachment_id(client, bank_with_photo, "d1")
    llm.on_step("extract_facts").returns(extracted(fact("Alice went on a trip", who="Alice", entities=["Alice"])))

    await client.aretain(bank_id=bank_with_photo, content="Alice went on a trip.", document_id="d1")
    await settled(bank_with_photo)

    with pytest.raises(NotFoundException):
        await client.memory.get_bank_attachment(bank_with_photo, attachment_id)


async def test_clearing_the_bank_retires_its_images(client, bank_with_photo):
    """A cleared bank goes on existing, so nothing cascades from the bank row: an
    image that outlived the clear stayed downloadable from a bank with no documents."""
    attachment_id = await _attachment_id(client, bank_with_photo, "d1")

    await client.memory.clear_bank_memories(bank_with_photo)

    with pytest.raises(NotFoundException):
        await client.memory.get_bank_attachment(bank_with_photo, attachment_id)


async def test_the_fact_drawn_from_the_image_is_recallable(client, bank_with_photo):
    """End to end: the image produced a memory like any other."""
    response = await client.arecall(bank_id=bank_with_photo, query="Where was Alice photographed?")

    assert [r.text for r in response.results] == [FACT]


async def test_dry_run_multimodal_extraction_via_client(client, llm, bank_id):
    """Multimodal content can be dry-run extracted through the client without persisting memories."""
    llm.on_step("extract_facts").returns(
        extracted(
            fact(
                "Alice stood in front of the Brandenburg Gate",
                who="Alice",
                entities=["Alice"],
                from_attachments=[1],
            )
        )
    )

    await client.acreate_bank(bank_id=bank_id, name="Test Bank")

    request = DryRunExtractRequest.from_dict({"content": CONTENT})
    result = await client.memory.dry_run_extract_memories(
        bank_id=bank_id,
        dry_run_extract_request=request,
    )

    # Deterministic assertion on candidate facts and attachment attribution
    assert len(result.facts) == 1
    extracted_fact = result.facts[0]
    assert extracted_fact.text == "Alice stood in front of the Brandenburg Gate | Involving: Alice"
    assert extracted_fact.fact_type == "world"
    assert extracted_fact.entities == ["Alice"]
    assert extracted_fact.chunk_index == 0
    assert len(extracted_fact.attachments) == 1
    attachment = extracted_fact.attachments[0]
    assert attachment.block_index == 1
    assert attachment.type == "image"
    assert attachment.media_type == "image/png"

    # Deterministic assertion on chunks
    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert chunk.text == f"{CAPTION}\n\n[Block #1: image (image/png)]"
    assert chunk.fact_count == 1

    # Token usage is returned
    assert result.usage is not None
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert result.usage.total_tokens == result.usage.input_tokens + result.usage.output_tokens

    memories = await client.memory.list_memories(bank_id, limit=100)
    assert memories.total == 0

    documents = await client.documents.list_documents(bank_id)
    assert len(documents.items) == 0

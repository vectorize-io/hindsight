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


async def test_the_fact_drawn_from_the_image_is_recallable(client, bank_with_photo):
    """End to end: the image produced a memory like any other."""
    response = await client.arecall(bank_id=bank_with_photo, query="Where was Alice photographed?")

    assert [r.text for r in response.results] == [FACT]

"""Delta re-retain of a document that carries attachments.

Delta retain re-extracts only the chunks whose text changed. The document row,
though, is rewritten whole on every pass, and that rewrite is what decides which
attachments the document still carries: `sync_document_attachments` derives the
set from the canonical text and deletes every row the text no longer names.

So the load-bearing assumption is that a delta write hands that function the
**whole** document body, not just the changed part. If it ever handed it a
fragment, the attachments in the untouched chunks would have their rows deleted
and -- since an attachment now owns its bytes -- their blobs reclaimed along with
them. That failure is silent: the facts extracted from the image survive, the
document still renders its placeholder, and only the picture is gone.

Delta updates are frequent, so every combination is pinned here: untouched,
edited elsewhere, replaced, removed, added, and one-of-two changed. Each test
first asserts the document really did split into several chunks -- without that
premise the interesting cases collapse into a plain full re-ingest and the suite
would be green for the wrong reason.
"""

import base64
import logging
import uuid

import pytest

from hindsight_api.engine.retain.attachment_content import compute_attachment_hash

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
OTHER_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
THIRD_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

PNG = compute_attachment_hash(PNG_BYTES)
OTHER = compute_attachment_hash(OTHER_BYTES)
THIRD = compute_attachment_hash(THIRD_BYTES)

DOCUMENT_ID = "runbook"
# Comfortably longer than the chunk size set below, so each part is its own chunk.
PART_A = "First section. " * 30
PART_B = "Second section. " * 30
PART_C = "Third section. " * 30


def _image(data: bytes) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(data).decode()},
    }


def _text(text: str) -> dict:
    return {"type": "text", "text": text}


async def _bank(api_client) -> str:
    """A bank whose chunk budget is small enough to split the documents below."""
    bank_id = f"delta-att-{uuid.uuid4().hex[:8]}"
    created = await api_client.put(f"/v1/default/banks/{bank_id}", json={})
    assert created.status_code == 200, created.text
    config = await api_client.patch(f"/v1/default/banks/{bank_id}/config", json={"updates": {"retain_chunk_size": 200}})
    assert config.status_code == 200, config.text
    return bank_id


async def _retain(api_client, bank_id: str, blocks: list) -> None:
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": blocks, "document_id": DOCUMENT_ID}], "async": False},
    )
    assert response.status_code == 200, response.text


async def _assert_split(api_client, bank_id: str) -> None:
    """The premise every test here rests on: more than one chunk to be 'untouched'."""
    chunks = await api_client.get(f"/v1/default/banks/{bank_id}/documents/{DOCUMENT_ID}/chunks")
    assert chunks.status_code == 200, chunks.text
    assert len(chunks.json()["items"]) > 1, "the document did not split; the delta cases would be vacuous"


def _assert_delta_used(caplog) -> None:
    """The re-retain must have taken the delta path, not a full re-ingest.

    The chunk-count premise alone is not enough: if chunking or the delta gate
    ever changed so that a re-retain fell back to a full replace, every test here
    would still pass while quietly covering a different code path. The delta
    orchestrator logs its chunk diff, so that is what we pin.
    """
    assert any("[delta]" in record.getMessage() for record in caplog.records), (
        "the re-retain did not take the delta path; these tests no longer cover what they claim"
    )


async def _rows(memory, bank_id: str) -> set[str]:
    """The attachment hashes this bank holds for the document under test."""
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        rows = await conn.fetch(
            "SELECT attachment_hash FROM attachments WHERE bank_id = $1 AND document_id = $2",
            bank_id,
            DOCUMENT_ID,
        )
    return {row["attachment_hash"] for row in rows}


async def _blob_exists(memory, bank_id: str, attachment_hash: str) -> bool:
    from hindsight_api.engine.retain.attachment_store import attachment_storage_key

    try:
        await memory._file_storage.retrieve(attachment_storage_key(bank_id, DOCUMENT_ID, attachment_hash))
        return True
    except FileNotFoundError:
        return False


async def _assert_held(memory, bank_id: str, *hashes: str) -> None:
    """Exactly these attachments, rows and bytes both."""
    assert await _rows(memory, bank_id) == set(hashes)
    for attachment_hash in hashes:
        assert await _blob_exists(memory, bank_id, attachment_hash), f"{attachment_hash[:12]} lost its bytes"


@pytest.mark.asyncio
async def test_editing_another_chunk_keeps_the_untouched_chunks_attachment(api_client, memory, caplog):
    """The case this file exists for: the image is in a chunk the edit never touched.

    A delta write that passed only the changed text to the document rewrite would
    delete this attachment's row and reclaim its bytes, while leaving every fact
    drawn from it in place.
    """
    bank_id = await _bank(api_client)
    await _retain(api_client, bank_id, [_text(PART_A), _image(PNG_BYTES), _text(PART_B)])
    await _assert_split(api_client, bank_id)
    await _assert_held(memory, bank_id, PNG)

    # Only the second section changes; the image and the prose around it are untouched.
    with caplog.at_level(logging.INFO):
        await _retain(api_client, bank_id, [_text(PART_A), _image(PNG_BYTES), _text(PART_B + "Now with an addendum. ")])

    _assert_delta_used(caplog)
    await _assert_held(memory, bank_id, PNG)


@pytest.mark.asyncio
async def test_re_retaining_identical_content_keeps_the_attachment(api_client, memory):
    """The pure no-op delta: nothing changed, so nothing may be reclaimed."""
    bank_id = await _bank(api_client)
    blocks = [_text(PART_A), _image(PNG_BYTES), _text(PART_B)]
    await _retain(api_client, bank_id, blocks)
    await _assert_split(api_client, bank_id)

    await _retain(api_client, bank_id, blocks)

    await _assert_held(memory, bank_id, PNG)


@pytest.mark.asyncio
async def test_replacing_the_image_reclaims_the_old_one_and_stores_the_new(api_client, memory):
    bank_id = await _bank(api_client)
    await _retain(api_client, bank_id, [_text(PART_A), _image(PNG_BYTES), _text(PART_B)])
    await _assert_split(api_client, bank_id)

    await _retain(api_client, bank_id, [_text(PART_A), _image(OTHER_BYTES), _text(PART_B)])

    await _assert_held(memory, bank_id, OTHER)
    assert not await _blob_exists(memory, bank_id, PNG), "the replaced image kept its bytes"


@pytest.mark.asyncio
async def test_removing_the_image_reclaims_it(api_client, memory):
    bank_id = await _bank(api_client)
    await _retain(api_client, bank_id, [_text(PART_A), _image(PNG_BYTES), _text(PART_B)])
    await _assert_split(api_client, bank_id)

    await _retain(api_client, bank_id, [_text(PART_A), _text(PART_B)])

    assert await _rows(memory, bank_id) == set()
    assert not await _blob_exists(memory, bank_id, PNG)


@pytest.mark.asyncio
async def test_adding_an_image_to_an_existing_document_stores_it(api_client, memory):
    """The other direction: a delta write that gains an attachment must record it."""
    bank_id = await _bank(api_client)
    await _retain(api_client, bank_id, [_text(PART_A), _text(PART_B)])
    await _assert_split(api_client, bank_id)
    assert await _rows(memory, bank_id) == set()

    await _retain(api_client, bank_id, [_text(PART_A), _image(PNG_BYTES), _text(PART_B)])

    await _assert_held(memory, bank_id, PNG)


@pytest.mark.asyncio
async def test_changing_one_of_two_images_leaves_the_other_alone(api_client, memory, caplog):
    """Two attachments in two different chunks; only the second one is swapped.

    The sharpest version of the trap: the rewrite must reclaim exactly one blob
    and keep exactly one, in a single pass.
    """
    bank_id = await _bank(api_client)
    await _retain(
        api_client,
        bank_id,
        [_text(PART_A), _image(PNG_BYTES), _text(PART_B), _image(OTHER_BYTES), _text(PART_C)],
    )
    await _assert_split(api_client, bank_id)
    await _assert_held(memory, bank_id, PNG, OTHER)

    with caplog.at_level(logging.INFO):
        await _retain(
            api_client,
            bank_id,
            [_text(PART_A), _image(PNG_BYTES), _text(PART_B), _image(THIRD_BYTES), _text(PART_C)],
        )

    _assert_delta_used(caplog)
    await _assert_held(memory, bank_id, PNG, THIRD)
    assert not await _blob_exists(memory, bank_id, OTHER), "the swapped-out image kept its bytes"

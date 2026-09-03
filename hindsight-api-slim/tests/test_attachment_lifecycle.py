"""Which documents reference an attachment, and when its bytes are reclaimed.

Content-addressing makes deletion non-obvious: one blob can back ten documents,
so deleting a document must reclaim *only* what nothing else still needs. The
`document_attachments` edge exists for exactly this, and is derived from the
canonical text on every write rather than supplied — so a re-ingest that drops an
attachment cannot leave a stale row behind.

These are the tests that would catch either failure mode: reclaiming a blob two
documents share (data loss), or keeping one nothing references (a leak).
"""

import base64
import uuid

import pytest

from hindsight_api.engine.retain.attachment_content import compute_attachment_hash

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
OTHER_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _image_block(data: bytes = PNG_BYTES) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(data).decode()},
    }


async def _retain(client, bank_id: str, content, document_id: str):
    response = await client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": content, "document_id": document_id}], "async": False},
    )
    assert response.status_code == 200, response.text


async def _edges(memory, bank_id: str) -> set[tuple[str, str]]:
    """The (document_id, attachment_hash) pairs recorded for the bank.

    Read directly: the point under test is the derived table itself, which no
    public read surface exposes as such.
    """
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        rows = await conn.fetch(
            "SELECT document_id, attachment_hash FROM document_attachments WHERE bank_id = $1", bank_id
        )
    return {(row["document_id"], row["attachment_hash"]) for row in rows}


async def _attachment_hashes(memory, bank_id: str) -> set[str]:
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        rows = await conn.fetch("SELECT attachment_hash FROM attachments WHERE bank_id = $1", bank_id)
    return {row["attachment_hash"] for row in rows}


async def _blob_exists(memory, bank_id: str, attachment_hash: str) -> bool:
    from hindsight_api.engine.retain.attachment_store import attachment_storage_key

    try:
        await memory._file_storage.retrieve(attachment_storage_key(bank_id, attachment_hash))
        return True
    except FileNotFoundError:
        return False


@pytest.mark.asyncio
async def test_the_edge_is_recorded_for_every_document_that_references_it(api_client, memory):
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    png = compute_attachment_hash(PNG_BYTES)

    await _retain(api_client, bank_id, [{"type": "text", "text": "one"}, _image_block()], "doc-a")
    await _retain(api_client, bank_id, [{"type": "text", "text": "two"}, _image_block()], "doc-b")

    assert await _edges(memory, bank_id) == {("doc-a", png), ("doc-b", png)}
    # Content-addressed: two documents, one blob.
    assert await _attachment_hashes(memory, bank_id) == {png}


@pytest.mark.asyncio
async def test_re_ingesting_without_an_attachment_drops_its_edge(api_client, memory):
    """The edge is derived, so it cannot outlive the text that justified it."""
    bank_id = f"life-{uuid.uuid4().hex[:8]}"

    await _retain(api_client, bank_id, [{"type": "text", "text": "before"}, _image_block()], "doc")
    assert await _edges(memory, bank_id)

    await _retain(api_client, bank_id, "plain text now, no attachment", "doc")

    assert await _edges(memory, bank_id) == set()


@pytest.mark.asyncio
async def test_deleting_the_last_referencing_document_reclaims_the_blob(api_client, memory):
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    png = compute_attachment_hash(PNG_BYTES)
    await _retain(api_client, bank_id, [{"type": "text", "text": "only"}, _image_block()], "doc")
    assert await _blob_exists(memory, bank_id, png)

    response = await api_client.delete(f"/v1/default/banks/{bank_id}/documents/doc")
    assert response.status_code == 200, response.text

    assert await _attachment_hashes(memory, bank_id) == set()
    assert not await _blob_exists(memory, bank_id, png)


@pytest.mark.asyncio
async def test_a_shared_blob_survives_deleting_one_of_its_documents(api_client, memory):
    """The failure this guards against is data loss, not a leak."""
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    png = compute_attachment_hash(PNG_BYTES)
    await _retain(api_client, bank_id, [{"type": "text", "text": "one"}, _image_block()], "doc-a")
    await _retain(api_client, bank_id, [{"type": "text", "text": "two"}, _image_block()], "doc-b")

    response = await api_client.delete(f"/v1/default/banks/{bank_id}/documents/doc-a")
    assert response.status_code == 200, response.text

    assert await _attachment_hashes(memory, bank_id) == {png}
    assert await _blob_exists(memory, bank_id, png)
    assert await _edges(memory, bank_id) == {("doc-b", png)}


@pytest.mark.asyncio
async def test_deleting_a_document_leaves_another_documents_own_attachment_alone(api_client, memory):
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    kept = compute_attachment_hash(OTHER_BYTES)
    await _retain(api_client, bank_id, [{"type": "text", "text": "a"}, _image_block(PNG_BYTES)], "doc-a")
    await _retain(api_client, bank_id, [{"type": "text", "text": "b"}, _image_block(OTHER_BYTES)], "doc-b")

    response = await api_client.delete(f"/v1/default/banks/{bank_id}/documents/doc-a")
    assert response.status_code == 200, response.text

    assert await _attachment_hashes(memory, bank_id) == {kept}
    assert await _blob_exists(memory, bank_id, kept)

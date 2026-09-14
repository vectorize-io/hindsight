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

from hindsight_api.engine.retain.attachment_content import compute_attachment_hash, short_attachment_id

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


async def _edges(client, bank_id: str) -> set[tuple[str, str]]:
    """The (document_id, attachment short id) pairs the bank's documents reference.

    Read through the document API rather than the ``document_attachments`` table.
    The property is "which documents still reference which blob", and that is the
    same question on every backend -- but only a SQL bank answers it from that
    table; a store-owned bank derives it from the stored text. Reading the table
    would test one backend's bookkeeping instead of the behaviour.
    """
    listed = await client.get(f"/v1/default/banks/{bank_id}/documents", params={"limit": 1000})
    assert listed.status_code == 200, listed.text
    pairs: set[tuple[str, str]] = set()
    for item in listed.json()["items"]:
        document = await client.get(f"/v1/default/banks/{bank_id}/documents/{item['id']}")
        assert document.status_code == 200, document.text
        pairs.update((item["id"], attachment["id"]) for attachment in document.json().get("attachments") or [])
    return pairs


async def _attachment_hashes(memory, bank_id: str) -> set[str]:
    """The bank's ``attachments`` rows: the blob registry, written at ingress on every backend."""
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

    assert await _edges(api_client, bank_id) == {
        ("doc-a", short_attachment_id(png)),
        ("doc-b", short_attachment_id(png)),
    }
    # Content-addressed: two documents, one blob.
    assert await _attachment_hashes(memory, bank_id) == {png}


@pytest.mark.asyncio
async def test_re_ingesting_without_an_attachment_drops_its_edge(api_client, memory):
    """The edge is derived, so it cannot outlive the text that justified it."""
    bank_id = f"life-{uuid.uuid4().hex[:8]}"

    await _retain(api_client, bank_id, [{"type": "text", "text": "before"}, _image_block()], "doc")
    assert await _edges(api_client, bank_id)

    await _retain(api_client, bank_id, "plain text now, no attachment", "doc")

    assert await _edges(api_client, bank_id) == set()


@pytest.mark.asyncio
async def test_re_ingesting_without_an_attachment_reclaims_it(api_client, memory):
    """#4364: the edge goes on the rewrite, so the row and bytes must go with it."""
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    png = compute_attachment_hash(PNG_BYTES)
    await _retain(api_client, bank_id, [{"type": "text", "text": "before"}, _image_block()], "doc")

    await _retain(api_client, bank_id, "plain text now, no attachment", "doc")

    assert await _attachment_hashes(memory, bank_id) == set()
    assert not await _blob_exists(memory, bank_id, png)


@pytest.mark.asyncio
async def test_re_ingesting_keeps_an_attachment_another_document_still_references(api_client, memory):
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    png = compute_attachment_hash(PNG_BYTES)
    await _retain(api_client, bank_id, [{"type": "text", "text": "one"}, _image_block()], "doc-a")
    await _retain(api_client, bank_id, [{"type": "text", "text": "two"}, _image_block()], "doc-b")

    await _retain(api_client, bank_id, "doc-a dropped its image", "doc-a")

    assert await _attachment_hashes(memory, bank_id) == {png}
    assert await _blob_exists(memory, bank_id, png)


@pytest.mark.asyncio
async def test_deleting_the_bank_deletes_its_attachment_blobs(api_client, memory):
    """#4365: the rows cascade with the bank, and the bytes must not outlive them."""
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    png = compute_attachment_hash(PNG_BYTES)
    await _retain(api_client, bank_id, [{"type": "text", "text": "only"}, _image_block()], "doc")
    assert await _blob_exists(memory, bank_id, png)

    response = await api_client.delete(f"/v1/default/banks/{bank_id}")
    assert response.status_code == 200, response.text

    assert not await _blob_exists(memory, bank_id, png)


@pytest.mark.asyncio
async def test_clearing_the_bank_reclaims_its_attachments(api_client, memory):
    """A cleared bank stays, and `attachments` hangs off the bank, so nothing cascades."""
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    png = compute_attachment_hash(PNG_BYTES)
    await _retain(api_client, bank_id, [{"type": "text", "text": "only"}, _image_block()], "doc")

    response = await api_client.delete(f"/v1/default/banks/{bank_id}/memories")
    assert response.status_code == 200, response.text

    assert await _attachment_hashes(memory, bank_id) == set()
    assert not await _blob_exists(memory, bank_id, png)


@pytest.mark.asyncio
async def test_deleting_the_bank_deletes_blobs_stored_under_the_tenantless_layout(api_client, memory):
    """Rows written before keys carried the tenant sit outside the swept prefix."""
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    png = compute_attachment_hash(PNG_BYTES)
    await _retain(api_client, bank_id, [{"type": "text", "text": "only"}, _image_block()], "doc")
    legacy_key = f"banks/{bank_id}/attachments/sha256-{png}"
    await memory._file_storage.store(file_data=PNG_BYTES, key=legacy_key)
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        await conn.execute("UPDATE attachments SET storage_key = $2 WHERE bank_id = $1", bank_id, legacy_key)

    response = await api_client.delete(f"/v1/default/banks/{bank_id}")
    assert response.status_code == 200, response.text

    with pytest.raises(FileNotFoundError):
        await memory._file_storage.retrieve(legacy_key)


@pytest.mark.asyncio
async def test_attachment_keys_are_scoped_to_the_tenant(memory):
    """Object stores share one bucket across tenant schemas; the key must say whose it is."""
    from hindsight_api.engine.memory_engine import get_current_schema
    from hindsight_api.engine.retain.attachment_store import attachment_storage_key

    assert attachment_storage_key("b", "abc") == f"tenants/{get_current_schema()}/banks/b/attachments/sha256-abc"
    # A bank id holding "/" must not nest under another bank's prefix.
    assert attachment_storage_key("a/b", "abc").startswith(f"tenants/{get_current_schema()}/banks/a%2Fb/")


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
    assert await _edges(api_client, bank_id) == {("doc-b", short_attachment_id(png))}


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

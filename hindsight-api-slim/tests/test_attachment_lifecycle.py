"""Which document an attachment belongs to, and when its bytes are reclaimed.

An attachment belongs to one document. Two documents carrying the same image
hold two copies, so deleting either one deletes exactly its own bytes and never
has to ask whether somebody else still needs them — the question a bank whose
documents live in a memories store cannot answer, and the reason reclaim used to
bail out for those banks entirely.

These are the tests that would catch either failure mode: reclaiming a blob a
document still displays (data loss), or keeping one nothing references (a leak).
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
    """The (document_id, attachment short id) pairs the bank's documents carry.

    Read through the document API rather than the ``attachments`` table, so the
    assertion is about what a caller can see rather than one backend's
    bookkeeping.
    """
    listed = await client.get(f"/v1/default/banks/{bank_id}/documents", params={"limit": 1000})
    assert listed.status_code == 200, listed.text
    pairs: set[tuple[str, str]] = set()
    for item in listed.json()["items"]:
        document = await client.get(f"/v1/default/banks/{bank_id}/documents/{item['id']}")
        assert document.status_code == 200, document.text
        pairs.update((item["id"], attachment["id"]) for attachment in document.json().get("attachments") or [])
    return pairs


async def _attachment_rows(memory, bank_id: str) -> set[tuple[str, str]]:
    """The bank's ``attachments`` rows as (document_id, hash) — one per document that carries it."""
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        rows = await conn.fetch("SELECT document_id, attachment_hash FROM attachments WHERE bank_id = $1", bank_id)
    return {(row["document_id"], row["attachment_hash"]) for row in rows}


async def _blob_exists(memory, bank_id: str, document_id: str, attachment_hash: str) -> bool:
    from hindsight_api.engine.retain.attachment_store import attachment_storage_key

    try:
        await memory._file_storage.retrieve(attachment_storage_key(bank_id, document_id, attachment_hash))
        return True
    except FileNotFoundError:
        return False


@pytest.mark.asyncio
async def test_every_document_that_carries_an_attachment_gets_its_own_copy(api_client, memory):
    """Cross-document dedup is given up on purpose — this is what replaces it."""
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    png = compute_attachment_hash(PNG_BYTES)

    await _retain(api_client, bank_id, [{"type": "text", "text": "one"}, _image_block()], "doc-a")
    await _retain(api_client, bank_id, [{"type": "text", "text": "two"}, _image_block()], "doc-b")

    assert await _edges(api_client, bank_id) == {
        ("doc-a", short_attachment_id(png)),
        ("doc-b", short_attachment_id(png)),
    }
    # One row and one blob per document, not one shared by both.
    assert await _attachment_rows(memory, bank_id) == {("doc-a", png), ("doc-b", png)}
    assert await _blob_exists(memory, bank_id, "doc-a", png)
    assert await _blob_exists(memory, bank_id, "doc-b", png)


@pytest.mark.asyncio
async def test_re_ingesting_without_an_attachment_drops_it(api_client, memory):
    """What a document carries is derived from its text, so it cannot outlive it."""
    bank_id = f"life-{uuid.uuid4().hex[:8]}"

    await _retain(api_client, bank_id, [{"type": "text", "text": "before"}, _image_block()], "doc")
    assert await _edges(api_client, bank_id)

    await _retain(api_client, bank_id, "plain text now, no attachment", "doc")

    assert await _edges(api_client, bank_id) == set()


@pytest.mark.asyncio
async def test_re_ingesting_without_an_attachment_reclaims_it(api_client, memory):
    """#4364: the row goes on the rewrite, so the bytes must go with it."""
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    png = compute_attachment_hash(PNG_BYTES)
    await _retain(api_client, bank_id, [{"type": "text", "text": "before"}, _image_block()], "doc")

    await _retain(api_client, bank_id, "plain text now, no attachment", "doc")

    assert await _attachment_rows(memory, bank_id) == set()
    assert not await _blob_exists(memory, bank_id, "doc", png)


@pytest.mark.asyncio
async def test_re_ingesting_leaves_another_documents_copy_alone(api_client, memory):
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    png = compute_attachment_hash(PNG_BYTES)
    await _retain(api_client, bank_id, [{"type": "text", "text": "one"}, _image_block()], "doc-a")
    await _retain(api_client, bank_id, [{"type": "text", "text": "two"}, _image_block()], "doc-b")

    await _retain(api_client, bank_id, "doc-a dropped its image", "doc-a")

    assert await _attachment_rows(memory, bank_id) == {("doc-b", png)}
    assert not await _blob_exists(memory, bank_id, "doc-a", png)
    assert await _blob_exists(memory, bank_id, "doc-b", png)


@pytest.mark.asyncio
async def test_deleting_the_bank_deletes_its_attachment_blobs(api_client, memory):
    """#4365: the rows cascade with the bank, and the bytes must not outlive them."""
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    png = compute_attachment_hash(PNG_BYTES)
    await _retain(api_client, bank_id, [{"type": "text", "text": "only"}, _image_block()], "doc")
    assert await _blob_exists(memory, bank_id, "doc", png)

    response = await api_client.delete(f"/v1/default/banks/{bank_id}")
    assert response.status_code == 200, response.text

    assert not await _blob_exists(memory, bank_id, "doc", png)


@pytest.mark.asyncio
async def test_clearing_the_bank_reclaims_its_attachments(api_client, memory):
    """A cleared bank stays, and `attachments` hangs off the bank, so nothing cascades."""
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    png = compute_attachment_hash(PNG_BYTES)
    await _retain(api_client, bank_id, [{"type": "text", "text": "only"}, _image_block()], "doc")

    response = await api_client.delete(f"/v1/default/banks/{bank_id}/memories")
    assert response.status_code == 200, response.text

    assert await _attachment_rows(memory, bank_id) == set()
    assert not await _blob_exists(memory, bank_id, "doc", png)


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
async def test_attachment_keys_are_scoped_to_the_tenant_and_the_document(memory):
    """Object stores share one bucket across tenant schemas; the key must say whose it is."""
    from hindsight_api.engine.memory_engine import get_current_schema
    from hindsight_api.engine.retain.attachment_store import attachment_storage_key

    prefix = f"tenants/{get_current_schema()}/banks"
    assert attachment_storage_key("b", "d", "abc") == f"{prefix}/b/documents/d/attachments/sha256-abc"
    # A bank id holding "/" must not nest under another bank's prefix...
    assert attachment_storage_key("a/b", "d", "abc").startswith(f"{prefix}/a%2Fb/")
    # ...and neither must a document id.
    assert attachment_storage_key("b", "d/e", "abc") == f"{prefix}/b/documents/d%2Fe/attachments/sha256-abc"


@pytest.mark.asyncio
async def test_deleting_the_document_reclaims_its_blob(api_client, memory):
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    png = compute_attachment_hash(PNG_BYTES)
    await _retain(api_client, bank_id, [{"type": "text", "text": "only"}, _image_block()], "doc")
    assert await _blob_exists(memory, bank_id, "doc", png)

    response = await api_client.delete(f"/v1/default/banks/{bank_id}/documents/doc")
    assert response.status_code == 200, response.text

    assert await _attachment_rows(memory, bank_id) == set()
    assert not await _blob_exists(memory, bank_id, "doc", png)


@pytest.mark.asyncio
async def test_deleting_one_document_leaves_the_others_copy_of_the_same_image(api_client, memory):
    """The failure this guards against is data loss, not a leak."""
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    png = compute_attachment_hash(PNG_BYTES)
    await _retain(api_client, bank_id, [{"type": "text", "text": "one"}, _image_block()], "doc-a")
    await _retain(api_client, bank_id, [{"type": "text", "text": "two"}, _image_block()], "doc-b")

    response = await api_client.delete(f"/v1/default/banks/{bank_id}/documents/doc-a")
    assert response.status_code == 200, response.text

    assert await _attachment_rows(memory, bank_id) == {("doc-b", png)}
    assert not await _blob_exists(memory, bank_id, "doc-a", png)
    assert await _blob_exists(memory, bank_id, "doc-b", png)
    assert await _edges(api_client, bank_id) == {("doc-b", short_attachment_id(png))}


@pytest.mark.asyncio
async def test_deleting_a_document_leaves_another_documents_own_attachment_alone(api_client, memory):
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    kept = compute_attachment_hash(OTHER_BYTES)
    await _retain(api_client, bank_id, [{"type": "text", "text": "a"}, _image_block(PNG_BYTES)], "doc-a")
    await _retain(api_client, bank_id, [{"type": "text", "text": "b"}, _image_block(OTHER_BYTES)], "doc-b")

    response = await api_client.delete(f"/v1/default/banks/{bank_id}/documents/doc-a")
    assert response.status_code == 200, response.text

    assert await _attachment_rows(memory, bank_id) == {("doc-b", kept)}
    assert await _blob_exists(memory, bank_id, "doc-b", kept)


@pytest.mark.asyncio
async def test_a_legacy_shared_blob_survives_until_its_last_document_goes(api_client, memory):
    """Banks retained before this rule share one blob across their documents.

    The migration gives each document its own row but does not move the bytes, so
    the reclaim has to hold off until no row names that key any more. Reclaiming
    it with the first document would blank the image in the other.
    """
    bank_id = f"life-{uuid.uuid4().hex[:8]}"
    png = compute_attachment_hash(PNG_BYTES)
    await _retain(api_client, bank_id, [{"type": "text", "text": "one"}, _image_block()], "doc-a")
    await _retain(api_client, bank_id, [{"type": "text", "text": "two"}, _image_block()], "doc-b")

    # Put both documents back on one shared key, as the migration leaves them.
    shared_key = f"banks/{bank_id}/attachments/sha256-{png}"
    await memory._file_storage.store(file_data=PNG_BYTES, key=shared_key)
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        await conn.execute("UPDATE attachments SET storage_key = $2 WHERE bank_id = $1", bank_id, shared_key)

    assert (await api_client.delete(f"/v1/default/banks/{bank_id}/documents/doc-a")).status_code == 200
    assert await memory._file_storage.retrieve(shared_key), "doc-b still shows these bytes"

    assert (await api_client.delete(f"/v1/default/banks/{bank_id}/documents/doc-b")).status_code == 200
    with pytest.raises(FileNotFoundError):
        await memory._file_storage.retrieve(shared_key)

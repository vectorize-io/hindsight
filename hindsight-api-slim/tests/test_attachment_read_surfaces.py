"""Every read surface hands back the attachments behind what it returns.

An agent that recalls a fact derived from a screenshot needs to be able to *show*
that screenshot, and it should not matter which endpoint it happened to arrive
through. So this pins the whole surface at once — recall, get-memory,
list-memories, get-document, list-chunks, get-chunk — because the failure mode
here is one endpoint quietly lacking what the others have.

Note where a memory's attachments come from: not its own text, which carries a
readable `[image: ...]` note rather than a placeholder, but the per-fact edge the
extractor recorded — which attachments it actually looked at to produce that
fact.

The bank runs in `chunks` extraction mode so that edge is deterministic here: a
chunk-mode fact *is* its chunk, so it carries exactly the chunk's attachments,
with no model judgement in the loop. Whether a real extractor attributes
correctly is a question about the model, and is judged in
`test_attachment_attribution.py`; these tests are about whether every endpoint
hands back what was recorded.
"""

import base64
import uuid

import pytest

from hindsight_api.engine.retain.attachment_content import compute_attachment_hash, short_attachment_id

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_ID = short_attachment_id(compute_attachment_hash(PNG_BYTES))
DOCUMENT_ID = "vpn-article"


@pytest.fixture
async def bank_with_attachment(api_client):
    """A bank holding one document whose text carries one inline image."""
    bank_id = f"read-{uuid.uuid4().hex[:8]}"
    assert (await api_client.put(f"/v1/default/banks/{bank_id}", json={})).status_code == 200
    config = await api_client.patch(
        f"/v1/default/banks/{bank_id}/config",
        json={"updates": {"retain_extraction_mode": "chunks"}},
    )
    assert config.status_code == 200, config.text
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                {
                    "content": [
                        {"type": "text", "text": "To reset the VPN, click the button shown below."},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(PNG_BYTES).decode(),
                            },
                        },
                        {"type": "text", "text": "Then reconnect to the corporate network."},
                    ],
                    "document_id": DOCUMENT_ID,
                }
            ],
            "async": False,
        },
    )
    assert response.status_code == 200, response.text
    return bank_id


def _assert_handle(attachments) -> None:
    """Every surface must emit the same handle, so assert it in one place."""
    assert attachments, "no attachments returned"
    entry = attachments[0]
    assert entry["id"] == PNG_ID
    assert entry["kind"] == "image"
    assert entry["media_type"] == "image/png"
    assert entry["byte_size"] == len(PNG_BYTES)
    assert entry["url"].endswith(f"/attachments/{PNG_ID}")


PDF_BYTES = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n"
PDF_ID = short_attachment_id(compute_attachment_hash(PDF_BYTES))
PDF_NAME = "vpn-reset-guide.pdf"
NAMED_DOCUMENT_ID = "vpn-guide"


@pytest.fixture
async def bank_with_named_file(api_client):
    """A bank holding one document whose text carries one file the caller named."""
    bank_id = f"named-{uuid.uuid4().hex[:8]}"
    assert (await api_client.put(f"/v1/default/banks/{bank_id}", json={})).status_code == 200
    config = await api_client.patch(
        f"/v1/default/banks/{bank_id}/config",
        json={"updates": {"retain_extraction_mode": "chunks"}},
    )
    assert config.status_code == 200, config.text
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                {
                    "content": [
                        {"type": "text", "text": "To reset the VPN, follow the attached guide."},
                        {
                            "type": "file",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": base64.b64encode(PDF_BYTES).decode(),
                            },
                            "filename": PDF_NAME,
                        },
                    ],
                    "document_id": NAMED_DOCUMENT_ID,
                }
            ],
            "async": False,
        },
    )
    assert response.status_code == 200, response.text
    return bank_id


def _assert_named(attachments, where: str) -> None:
    assert attachments, f"{where}: no attachments returned"
    assert [a["id"] for a in attachments] == [PDF_ID], where
    assert attachments[0]["filename"] == PDF_NAME, f"{where}: filename was {attachments[0].get('filename')!r}"


@pytest.mark.asyncio
async def test_every_surface_returns_the_filename_the_caller_gave(api_client, bank_with_named_file):
    """The name the caller gave an attachment is part of the handle, on every surface.

    It lives on the document edge, not the blob, so each surface has to reach the document to find
    it -- and "which surface forgot to" is exactly the failure this pins.
    """
    bank_id = bank_with_named_file
    base = f"/v1/default/banks/{bank_id}"

    listed = await api_client.get(f"{base}/memories/list")
    assert listed.status_code == 200, listed.text
    with_file = [m for m in listed.json()["items"] if m.get("attachments")]
    assert with_file, "no memory carries the file"
    _assert_named(with_file[0]["attachments"], "list memories")

    detail = await api_client.get(f"{base}/memories/{with_file[0]['id']}")
    assert detail.status_code == 200, detail.text
    _assert_named(detail.json().get("attachments"), "get memory")

    recall = await api_client.post(f"{base}/memories/recall", json={"query": "How do I reset the VPN?"})
    assert recall.status_code == 200, recall.text
    recalled = [r for r in recall.json()["results"] if r.get("attachments")]
    assert recalled, "no recalled fact carries the file"
    _assert_named(recalled[0]["attachments"], "recall")

    document = await api_client.get(f"{base}/documents/{NAMED_DOCUMENT_ID}")
    assert document.status_code == 200, document.text
    _assert_named(document.json().get("attachments"), "get document")

    chunks = await api_client.get(f"{base}/documents/{NAMED_DOCUMENT_ID}/chunks")
    assert chunks.status_code == 200, chunks.text
    chunk_items = [c for c in chunks.json()["items"] if c.get("attachments")]
    assert chunk_items, "no chunk carries the file"
    _assert_named(chunk_items[0]["attachments"], "list chunks")

    chunk = await api_client.get(f"/v1/default/chunks/{chunk_items[0]['chunk_id']}")
    assert chunk.status_code == 200, chunk.text
    _assert_named(chunk.json().get("attachments"), "get chunk")


@pytest.mark.asyncio
async def test_get_document_returns_its_attachments(api_client, bank_with_attachment):
    response = await api_client.get(f"/v1/default/banks/{bank_with_attachment}/documents/{DOCUMENT_ID}")

    assert response.status_code == 200, response.text
    _assert_handle(response.json().get("attachments"))


@pytest.mark.asyncio
async def test_listing_a_documents_chunks_returns_their_attachments(api_client, bank_with_attachment):
    response = await api_client.get(f"/v1/default/banks/{bank_with_attachment}/documents/{DOCUMENT_ID}/chunks")

    assert response.status_code == 200, response.text
    chunks = response.json()["items"]
    _assert_handle(next(c["attachments"] for c in chunks if c.get("attachments")))


@pytest.mark.asyncio
async def test_get_chunk_returns_its_attachments(api_client, bank_with_attachment):
    listed = await api_client.get(f"/v1/default/banks/{bank_with_attachment}/documents/{DOCUMENT_ID}/chunks")
    chunk_id = next(c["chunk_id"] for c in listed.json()["items"] if c.get("attachments"))

    response = await api_client.get(f"/v1/default/chunks/{chunk_id}")

    assert response.status_code == 200, response.text
    _assert_handle(response.json().get("attachments"))


@pytest.mark.asyncio
async def test_listing_memories_returns_the_attachments_behind_each_fact(api_client, bank_with_attachment):
    response = await api_client.get(f"/v1/default/banks/{bank_with_attachment}/memories/list")

    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert items, "no memories were extracted"
    _assert_handle(next(m["attachments"] for m in items if m.get("attachments")))


@pytest.mark.asyncio
async def test_get_memory_returns_the_attachments_behind_the_fact(api_client, bank_with_attachment):
    listed = await api_client.get(f"/v1/default/banks/{bank_with_attachment}/memories/list")
    memory_id = next(m["id"] for m in listed.json()["items"] if m.get("attachments"))

    response = await api_client.get(f"/v1/default/banks/{bank_with_attachment}/memories/{memory_id}")

    assert response.status_code == 200, response.text
    _assert_handle(response.json().get("attachments"))


@pytest.mark.asyncio
async def test_recall_returns_them_on_chunks_and_on_memories(api_client, bank_with_attachment):
    response = await api_client.post(
        f"/v1/default/banks/{bank_with_attachment}/memories/recall",
        json={"query": "how do I reset the VPN", "include": {"chunks": {}}, "limit": 10},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    chunks = (body.get("chunks") or {}).values()
    _assert_handle(next(c["attachments"] for c in chunks if c.get("attachments")))

    # And on the results themselves, so an agent that did not ask for chunks can
    # still show what a fact was drawn from. The two are not the same set: a
    # chunk lists everything its text references, a result only what the
    # extractor attributed to that fact.
    results = body["results"]
    assert results, "recall returned no results"
    _assert_handle(next(r["attachments"] for r in results if r.get("attachments")))


@pytest.mark.asyncio
async def test_a_text_only_bank_reports_no_attachments_anywhere(api_client):
    """The field is absent, not an empty list, for the overwhelmingly common case."""
    bank_id = f"read-{uuid.uuid4().hex[:8]}"
    await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": "The VPN client is called Sentinel.", "document_id": "plain"}], "async": False},
    )

    document = await api_client.get(f"/v1/default/banks/{bank_id}/documents/plain")
    memories = await api_client.get(f"/v1/default/banks/{bank_id}/memories/list")

    recall = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": "what is the VPN client called"},
    )

    assert document.json().get("attachments") is None
    assert all(m.get("attachments") is None for m in memories.json()["items"])
    assert all(r.get("attachments") is None for r in recall.json()["results"])

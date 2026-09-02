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

    assert document.json().get("attachments") is None
    assert all(m.get("attachments") is None for m in memories.json()["items"])

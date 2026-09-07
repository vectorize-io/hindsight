"""Recall hands back the image behind an image-derived fact.

A fact links to its chunk; the chunk's text keeps the placeholder where the image
sat. These tests cover the last hop: turning that placeholder into something an
agent can actually fetch, and making sure the fetch is authorized against the
bank rather than against knowledge of the hash.

The hash is content-derived, so it is emphatically NOT a capability — anyone
holding the same PNG can compute it. The 404-for-everything behaviour is what
stops it being used to probe which images a bank has retained.
"""

import base64
import uuid

import pytest

from hindsight_api.engine.retain.attachment_content import compute_attachment_hash, short_attachment_id

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_HASH = compute_attachment_hash(PNG_BYTES)
PNG_ID = short_attachment_id(PNG_HASH)


def _image_block() -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(PNG_BYTES).decode()},
    }


async def _retain_article(client, bank_id: str) -> None:
    response = await client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                {
                    "content": [
                        {"type": "text", "text": "To reset the VPN connection, click the button shown below."},
                        _image_block(),
                        {"type": "text", "text": "After clicking it, reconnect to the corporate network."},
                    ],
                    "document_id": "vpn-article",
                }
            ],
            "async": False,
        },
    )
    assert response.status_code == 200, response.text


async def _recall(client, bank_id: str) -> dict:
    response = await client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": "how do I reset the VPN", "include": {"chunks": {}}, "limit": 10},
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.asyncio
async def test_a_recalled_chunk_carries_a_handle_for_the_image_it_shows(api_client, memory):
    bank_id = f"prov-{uuid.uuid4().hex[:8]}"
    await _retain_article(api_client, bank_id)

    chunks = (await _recall(api_client, bank_id)).get("chunks") or {}
    assert chunks, "recall returned no chunks"

    with_images = [chunk for chunk in chunks.values() if chunk.get("attachments")]
    assert with_images, "no chunk reported the attachment it references"

    image = with_images[0]["attachments"][0]
    # The short id is what document text carries; the full digest still identifies
    # the bytes.
    assert image["id"] == PNG_ID
    assert image["hash"] == PNG_HASH
    assert image["kind"] == "image"
    assert image["media_type"] == "image/png"
    assert image["byte_size"] == len(PNG_BYTES)
    assert image["url"] == f"/v1/default/banks/{bank_id}/attachments/{PNG_ID}"

    # The placeholder stays in the text, marking where the image belongs.
    assert PNG_ID in with_images[0]["text"]


@pytest.mark.asyncio
async def test_the_returned_url_serves_the_original_bytes(api_client, memory):
    bank_id = f"prov-{uuid.uuid4().hex[:8]}"
    await _retain_article(api_client, bank_id)

    chunks = (await _recall(api_client, bank_id)).get("chunks") or {}
    url = next(chunk["attachments"][0]["url"] for chunk in chunks.values() if chunk.get("attachments"))

    response = await api_client.get(url)

    assert response.status_code == 200
    assert response.content == PNG_BYTES
    assert response.headers["content-type"] == "image/png"


@pytest.mark.asyncio
async def test_a_text_only_chunk_reports_no_images(api_client, memory):
    """The field is absent, not an empty list, for the overwhelmingly common case."""
    bank_id = f"prov-{uuid.uuid4().hex[:8]}"
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": "The VPN client is called Sentinel.", "document_id": "plain"}], "async": False},
    )
    assert response.status_code == 200, response.text

    chunks = (await _recall(api_client, bank_id)).get("chunks") or {}
    assert chunks
    assert all(chunk.get("attachments") is None for chunk in chunks.values())


@pytest.mark.asyncio
async def test_another_banks_image_is_not_readable_even_with_the_right_hash(api_client, memory):
    """The hash is content-derived, so it must not act as a bearer token."""
    owner = f"prov-{uuid.uuid4().hex[:8]}"
    await _retain_article(api_client, owner)

    other = f"prov-{uuid.uuid4().hex[:8]}"
    await api_client.post(
        f"/v1/default/banks/{other}/memories",
        json={"items": [{"content": "unrelated", "document_id": "x"}], "async": False},
    )

    response = await api_client.get(f"/v1/default/banks/{other}/attachments/{PNG_ID}")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_an_unknown_hash_is_a_404(api_client, memory):
    bank_id = f"prov-{uuid.uuid4().hex[:8]}"
    await _retain_article(api_client, bank_id)

    response = await api_client.get(f"/v1/default/banks/{bank_id}/attachments/{'0' * 12}")

    assert response.status_code == 404

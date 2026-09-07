"""Retain accepting inline images: what happens at the API boundary.

Covers the contract and the storage seam, both deterministic:

- a block-form item flattens to canonical placeholder text, and the raw bytes
  are committed content-addressed *before* anything is submitted, so no base64
  ever reaches the retain pipeline or an async operation's payload;
- identical images dedupe to one blob and one row, across items and re-ingests;
- malformed, oversized and over-numerous images are the caller's error (400),
  named down to the offending item and block.

What the vision model then *makes* of an image is a separate, non-deterministic
question -- see the judge test in test_retain_multimodal_extraction.py.
"""

import base64
import uuid

import pytest

from hindsight_api.engine.retain.attachment_content import (
    compute_attachment_hash,
    attachment_placeholder,
    iter_placeholder_ids,
    short_attachment_id,
)
from hindsight_api.engine.retain.attachment_store import attachment_storage_key

# A one-pixel PNG. Real bytes rather than a fake string so the media type is not
# a lie, and small enough that the size limits stay the thing under test.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
OTHER_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _image_block(data: bytes = PNG_BYTES, media_type: str = "image/png") -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(data).decode(),
        },
    }


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


async def _retain(client, bank_id: str, content, **item_fields):
    return await client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={"items": [{"content": content, **item_fields}], "async": False},
    )


async def _document_text(memory, bank_id: str, document_id: str) -> str:
    """The exact canonical body retain stored.

    Read straight from `documents` on purpose: the property under test is that
    the placeholder text is byte-for-byte what the pipeline persists (that is
    what content_hash idempotency keys on), and no engine read method exposes
    the stored body verbatim.
    """
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        return await conn.fetchval(
            "SELECT original_text FROM documents WHERE id = $1 AND bank_id = $2",
            document_id,
            bank_id,
        )


async def _bank_attachment_rows(memory, bank_id: str) -> list[dict]:
    """The bank's stored image rows.

    Direct SQL because the assertion is about storage-layer state the public API
    cannot express — that an image retained N times occupies exactly one row and
    one content-addressed key.
    """
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        rows = await conn.fetch(
            "SELECT attachment_hash, short_id, media_type, byte_size, storage_key FROM attachments WHERE bank_id = $1",
            bank_id,
        )
    return [dict(row) for row in rows]


@pytest.mark.asyncio
async def test_image_block_becomes_a_placeholder_in_the_stored_document(api_client, memory):
    """The image lands between the sentences that frame it, as a placeholder.

    This is the whole design: the document stays plain text, so content_hash
    idempotency, append and chunk-delta re-extraction keep working untouched.
    """
    bank_id = f"img-{uuid.uuid4().hex[:8]}"
    document_id = "vpn-reset"

    response = await _retain(
        api_client,
        bank_id,
        [
            _text_block("To reset the VPN, click the button shown:"),
            _image_block(),
            _text_block("...then reconnect."),
        ],
        document_id=document_id,
    )
    assert response.status_code == 200, response.text

    stored = await _document_text(memory, bank_id, document_id)
    expected_hash = compute_attachment_hash(PNG_BYTES)
    assert stored == (
        f"To reset the VPN, click the button shown:\n\n{attachment_placeholder(expected_hash)}\n\n...then reconnect."
    )
    # No base64 anywhere in what the pipeline persisted.
    assert base64.b64encode(PNG_BYTES).decode() not in stored


@pytest.mark.asyncio
async def test_image_bytes_are_stored_content_addressed_and_retrievable(api_client, memory):
    bank_id = f"img-{uuid.uuid4().hex[:8]}"

    response = await _retain(api_client, bank_id, [_text_block("see:"), _image_block()], document_id="d1")
    assert response.status_code == 200, response.text

    expected_hash = compute_attachment_hash(PNG_BYTES)
    rows = await _bank_attachment_rows(memory, bank_id)
    assert len(rows) == 1
    assert rows[0]["attachment_hash"] == expected_hash
    assert rows[0]["media_type"] == "image/png"
    assert rows[0]["byte_size"] == len(PNG_BYTES)
    assert rows[0]["storage_key"] == attachment_storage_key(bank_id, expected_hash)

    assert await memory._file_storage.retrieve(rows[0]["storage_key"]) == PNG_BYTES


@pytest.mark.asyncio
async def test_the_same_image_across_documents_is_stored_once(api_client, memory):
    """Content-addressing is what makes re-ingesting a KB article cheap."""
    bank_id = f"img-{uuid.uuid4().hex[:8]}"

    for document_id in ("article-1", "article-2"):
        response = await _retain(
            api_client,
            bank_id,
            [_text_block(f"body of {document_id}"), _image_block()],
            document_id=document_id,
        )
        assert response.status_code == 200, response.text

    rows = await _bank_attachment_rows(memory, bank_id)
    assert len(rows) == 1

    # Both documents still name it.
    for document_id in ("article-1", "article-2"):
        text = await _document_text(memory, bank_id, document_id)
        assert list(iter_placeholder_ids(text)) == [short_attachment_id(compute_attachment_hash(PNG_BYTES))]


@pytest.mark.asyncio
async def test_distinct_images_get_distinct_blobs(api_client, memory):
    bank_id = f"img-{uuid.uuid4().hex[:8]}"

    response = await _retain(
        api_client,
        bank_id,
        [_image_block(PNG_BYTES), _text_block("and"), _image_block(OTHER_PNG_BYTES)],
        document_id="two-images",
    )
    assert response.status_code == 200, response.text

    rows = await _bank_attachment_rows(memory, bank_id)
    assert {row["attachment_hash"] for row in rows} == {
        compute_attachment_hash(PNG_BYTES),
        compute_attachment_hash(OTHER_PNG_BYTES),
    }


@pytest.mark.asyncio
async def test_re_retaining_identical_multimodal_content_is_idempotent(api_client, memory):
    """The placeholder body must hash identically on the second pass.

    If canonicalization were not deterministic, every re-ingest of an unchanged
    article would look like a changed document and re-run extraction over it.
    """
    bank_id = f"img-{uuid.uuid4().hex[:8]}"
    content = [_text_block("intro"), _image_block(), _text_block("outro")]

    first = await _retain(api_client, bank_id, content, document_id="stable")
    assert first.status_code == 200, first.text
    text_after_first = await _document_text(memory, bank_id, "stable")

    second = await _retain(api_client, bank_id, content, document_id="stable")
    assert second.status_code == 200, second.text
    text_after_second = await _document_text(memory, bank_id, "stable")

    assert text_after_first == text_after_second
    assert len(await _bank_attachment_rows(memory, bank_id)) == 1


@pytest.mark.asyncio
async def test_a_lone_text_block_is_identical_to_the_plain_string_form(api_client, memory):
    """Migrating an image-free caller to the block form must be a no-op."""
    bank_id = f"img-{uuid.uuid4().hex[:8]}"

    assert (await _retain(api_client, bank_id, "Alice joined the AI team", document_id="s")).status_code == 200
    assert (
        await _retain(api_client, bank_id, [_text_block("Alice joined the AI team")], document_id="b")
    ).status_code == 200

    assert await _document_text(memory, bank_id, "s") == await _document_text(memory, bank_id, "b")


@pytest.mark.asyncio
async def test_malformed_base64_is_a_client_error_naming_the_block(api_client):
    bank_id = f"img-{uuid.uuid4().hex[:8]}"

    response = await _retain(
        api_client,
        bank_id,
        [
            _text_block("before"),
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "not base64!!"}},
        ],
    )

    assert response.status_code == 400
    assert "items[0].content[1]" in response.json()["detail"]


@pytest.mark.asyncio
async def test_any_well_formed_media_type_is_accepted(api_client, memory):
    """There is no allowlist: what the model can read is the model's answer to give.

    A type we have never heard of is stored and sent; a provider that cannot read
    it fails the retain with its own error, which is far more informative than a
    guess made here.
    """
    bank_id = f"img-{uuid.uuid4().hex[:8]}"

    response = await _retain(api_client, bank_id, [_image_block(media_type="image/avif")], document_id="exotic")

    assert response.status_code == 200, response.text
    assert (await _bank_attachment_rows(memory, bank_id))[0]["media_type"] == "image/avif"


@pytest.mark.asyncio
async def test_a_malformed_media_type_is_still_rejected(api_client):
    """ "png" is not a media type. That is a request error, not a model question."""
    bank_id = f"img-{uuid.uuid4().hex[:8]}"

    response = await _retain(api_client, bank_id, [_image_block(media_type="png")])

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_an_oversized_image_is_rejected_with_its_size(api_client, monkeypatch):
    from hindsight_api.config import get_config

    # The size caps are static server-level fields, so the handler reads them off
    # the global config; patch the object the proxy delegates to.
    monkeypatch.setattr(get_config()._config, "retain_attachment_max_size_mb", 0)
    bank_id = f"img-{uuid.uuid4().hex[:8]}"

    response = await _retain(api_client, bank_id, [_image_block()])

    assert response.status_code == 400
    assert "exceeding" in response.json()["detail"]


@pytest.mark.asyncio
async def test_too_many_images_in_one_item_is_rejected(api_client, monkeypatch):
    from hindsight_api.config import get_config

    monkeypatch.setattr(get_config()._config, "retain_attachment_max_count", 1)
    bank_id = f"img-{uuid.uuid4().hex[:8]}"

    response = await _retain(api_client, bank_id, [_image_block(PNG_BYTES), _image_block(OTHER_PNG_BYTES)])

    assert response.status_code == 400
    assert "more than 1 attachments" in response.json()["detail"]


@pytest.mark.asyncio
async def test_empty_content_is_still_rejected_in_block_form(api_client):
    bank_id = f"img-{uuid.uuid4().hex[:8]}"

    assert (await _retain(api_client, bank_id, [])).status_code == 422
    assert (await _retain(api_client, bank_id, [_text_block("   ")])).status_code == 422


@pytest.mark.asyncio
async def test_an_image_alone_is_content_even_with_no_prose(api_client, memory):
    """A screenshot with no surrounding text is a legitimate document."""
    bank_id = f"img-{uuid.uuid4().hex[:8]}"

    response = await _retain(api_client, bank_id, [_image_block()], document_id="bare")

    assert response.status_code == 200, response.text
    assert await _document_text(memory, bank_id, "bare") == attachment_placeholder(compute_attachment_hash(PNG_BYTES))


@pytest.mark.asyncio
async def test_plain_string_content_cannot_summon_an_image(api_client, memory):
    """Text must never be able to conjure a picture, whichever form it arrives in.

    Block text was scrubbed from the start; plain-string content was not, so a
    caller could hand-write a placeholder and have extraction resolve it to any
    image already retained in that bank. Regression for that gap.
    """
    bank_id = f"img-{uuid.uuid4().hex[:8]}"
    # Retain a real image, so there is something in the bank worth stealing.
    assert (await _retain(api_client, bank_id, [_image_block()], document_id="owner")).status_code == 200
    stolen = attachment_placeholder(compute_attachment_hash(PNG_BYTES))

    response = await _retain(api_client, bank_id, f"see {stolen} here", document_id="thief")

    assert response.status_code == 200, response.text
    assert await _document_text(memory, bank_id, "thief") == "see  here"


@pytest.mark.asyncio
async def test_editing_a_documents_text_keeps_its_own_attachments(api_client, memory):
    """Re-sending a document's own body is an edit, not an attempt to steal.

    The control plane's content editor loads `original_text` — placeholders and
    all — into a textarea and retains it back as a plain string. Scrubbing those
    placeholders silently deleted every screenshot in the article.
    """
    bank_id = f"img-{uuid.uuid4().hex[:8]}"
    await _retain(api_client, bank_id, [_text_block("before:"), _image_block()], document_id="article")
    stored = await _document_text(memory, bank_id, "article")
    assert list(iter_placeholder_ids(stored))

    edited = stored.replace("before:", "after the rewrite:")
    response = await _retain(api_client, bank_id, edited, document_id="article")

    assert response.status_code == 200, response.text
    text = await _document_text(memory, bank_id, "article")
    assert "after the rewrite:" in text
    assert list(iter_placeholder_ids(text)) == list(iter_placeholder_ids(stored))


@pytest.mark.asyncio
async def test_the_exemption_is_scoped_to_the_document_that_owns_it(api_client, memory):
    """Document A's attachment must not be summonable from document B's text."""
    bank_id = f"img-{uuid.uuid4().hex[:8]}"
    await _retain(api_client, bank_id, [_text_block("owner"), _image_block()], document_id="owner")
    owner_text = await _document_text(memory, bank_id, "owner")
    stolen = f"see {owner_text.split(chr(10))[2]} here"

    response = await _retain(api_client, bank_id, stolen, document_id="thief")

    assert response.status_code == 200, response.text
    assert not list(iter_placeholder_ids(await _document_text(memory, bank_id, "thief")))

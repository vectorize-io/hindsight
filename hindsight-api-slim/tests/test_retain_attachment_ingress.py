"""Retain accepting inline images: what happens at the API boundary.

Covers the contract and the storage seam, both deterministic:

- a block-form item flattens to canonical placeholder text, and the raw bytes
  are committed content-addressed *before* anything is submitted, so no base64
  ever reaches the retain pipeline or an async operation's payload;
- identical images within one document dedupe to one blob and one row, across
  items and re-ingests; a second document carrying the same image gets its own;
- malformed, oversized and over-numerous images are the caller's error (400),
  named down to the offending item and block.

What the vision model then *makes* of an image is a separate, non-deterministic
question -- see the judge test in test_retain_multimodal_extraction.py.
"""

import asyncio
import base64
import json
import uuid

import pytest

from hindsight_api.worker.exceptions import RetryTaskAt

from hindsight_api.engine.retain.attachment_content import (
    attachment_placeholder,
    compute_attachment_hash,
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


async def _document_text(client, bank_id: str, document_id: str) -> str:
    """The exact canonical body retain stored, as the document API returns it.

    The property under test is that the placeholder text is byte-for-byte what
    the pipeline persists (that is what content_hash idempotency keys on).
    ``original_text`` on get-document is that stored body verbatim on every
    backend -- a SQL bank reads it from ``documents``, a store-owned bank from
    the store's document record, where its ``documents`` column is NULL.
    """
    response = await client.get(f"/v1/default/banks/{bank_id}/documents/{document_id}")
    assert response.status_code == 200, response.text
    return response.json()["original_text"]


async def _bank_attachment_rows(memory, bank_id: str) -> list[dict]:
    """The bank's stored image rows.

    Direct SQL because the assertion is about storage-layer state the public API
    cannot express — that an image a document carries N times occupies exactly one
    row and one key, and that a second document carrying it gets its own.
    """
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        rows = await conn.fetch(
            "SELECT document_id, attachment_hash, short_id, media_type, byte_size, storage_key "
            "FROM attachments WHERE bank_id = $1",
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

    stored = await _document_text(api_client, bank_id, document_id)
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
    assert rows[0]["storage_key"] == attachment_storage_key(bank_id, "d1", expected_hash)

    assert await memory._file_storage.retrieve(rows[0]["storage_key"]) == PNG_BYTES


@pytest.mark.asyncio
async def test_the_same_image_is_stored_once_per_document(api_client, memory):
    """Dedup within a document, a copy per document.

    Cross-document dedup is given up deliberately: it is the only reason a delete
    would have to ask whether another document still needs the bytes, and that is
    what cannot be answered for a bank whose documents live in a memories store.
    """
    bank_id = f"img-{uuid.uuid4().hex[:8]}"

    for document_id in ("article-1", "article-2"):
        response = await _retain(
            api_client,
            bank_id,
            # Twice in the one document: that dedupes to a single row.
            [_text_block(f"body of {document_id}"), _image_block(), _text_block("again:"), _image_block()],
            document_id=document_id,
        )
        assert response.status_code == 200, response.text

    rows = await _bank_attachment_rows(memory, bank_id)
    assert {row["document_id"] for row in rows} == {"article-1", "article-2"}
    assert len(rows) == 2, "one row per document, and one per document only"
    assert len({row["storage_key"] for row in rows}) == 2, "each document holds its own copy of the bytes"

    # Both documents still name it -- twice each, since that is what they were sent.
    for document_id in ("article-1", "article-2"):
        text = await _document_text(api_client, bank_id, document_id)
        ids = list(iter_placeholder_ids(text))
        assert set(ids) == {short_attachment_id(compute_attachment_hash(PNG_BYTES))}
        assert len(ids) == 2, "the text keeps every placeholder; it is the storage that dedupes"


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
    text_after_first = await _document_text(api_client, bank_id, "stable")

    second = await _retain(api_client, bank_id, content, document_id="stable")
    assert second.status_code == 200, second.text
    text_after_second = await _document_text(api_client, bank_id, "stable")

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

    assert await _document_text(api_client, bank_id, "s") == await _document_text(api_client, bank_id, "b")


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
    assert await _document_text(api_client, bank_id, "bare") == attachment_placeholder(
        compute_attachment_hash(PNG_BYTES)
    )


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
    assert await _document_text(api_client, bank_id, "thief") == "see  here"


@pytest.mark.asyncio
async def test_editing_a_documents_text_keeps_its_own_attachments(api_client, memory):
    """Re-sending a document's own body is an edit, not an attempt to steal.

    The control plane's content editor loads `original_text` — placeholders and
    all — into a textarea and retains it back as a plain string. Scrubbing those
    placeholders silently deleted every screenshot in the article.
    """
    bank_id = f"img-{uuid.uuid4().hex[:8]}"
    await _retain(api_client, bank_id, [_text_block("before:"), _image_block()], document_id="article")
    stored = await _document_text(api_client, bank_id, "article")
    assert list(iter_placeholder_ids(stored))

    edited = stored.replace("before:", "after the rewrite:")
    response = await _retain(api_client, bank_id, edited, document_id="article")

    assert response.status_code == 200, response.text
    text = await _document_text(api_client, bank_id, "article")
    assert "after the rewrite:" in text
    assert list(iter_placeholder_ids(text)) == list(iter_placeholder_ids(stored))


@pytest.mark.asyncio
async def test_the_exemption_is_scoped_to_the_document_that_owns_it(api_client, memory):
    """Document A's attachment must not be summonable from document B's text."""
    bank_id = f"img-{uuid.uuid4().hex[:8]}"
    await _retain(api_client, bank_id, [_text_block("owner"), _image_block()], document_id="owner")
    owner_text = await _document_text(api_client, bank_id, "owner")
    stolen = f"see {owner_text.split(chr(10))[2]} here"

    response = await _retain(api_client, bank_id, stolen, document_id="thief")

    assert response.status_code == 200, response.text
    assert not list(iter_placeholder_ids(await _document_text(api_client, bank_id, "thief")))


@pytest.mark.asyncio
async def test_retain_failure_discards_ingress_attachments(api_client, memory, monkeypatch):
    """When a retain fails (e.g. LLM failure, timeout, unhandled error),
    attachments written at ingress must not leak into attachments table or storage.
    """
    bank_id = f"fail-{uuid.uuid4().hex[:8]}"

    async def _failing_retain(*args, **kwargs):
        raise RuntimeError("LLM extraction crashed")

    monkeypatch.setattr(memory, "_run_retain_execution", _failing_retain)

    response = await _retain(api_client, bank_id, [_image_block()], document_id="failed_doc")
    assert response.status_code == 500

    # Verify attachment was discarded and not leaked
    rows = await _bank_attachment_rows(memory, bank_id)
    assert rows == [], f"Attachments leaked after retain failure: {rows}"

    storage_key = attachment_storage_key(bank_id, "failed_doc", compute_attachment_hash(PNG_BYTES))
    assert not await memory._file_storage.exists(storage_key)
    with pytest.raises(FileNotFoundError):
        await memory._file_storage.retrieve(storage_key)


@pytest.mark.asyncio
async def test_retain_failure_before_execution_discards_attachments(api_client, memory):
    """When a retain request fails before retain_batch_async starts (e.g. 400 bad request),
    attachments written at ingress must be discarded.
    """
    bank_id = f"fail-pre-{uuid.uuid4().hex[:8]}"

    # Sending operation_id with multiple strategies triggers 400 in http.py after attachments are stored
    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                {"content": [_image_block()], "strategy": "s1", "document_id": "doc1"},
                {"content": [_image_block()], "strategy": "s2", "document_id": "doc2"},
            ],
            "operation_id": str(uuid.uuid4()),
            "async": True,
        },
    )
    assert response.status_code == 400
    assert "operation_id requires all retain items to resolve to a single strategy" in response.text

    # Verify attachments were not left behind
    rows = await _bank_attachment_rows(memory, bank_id)
    assert rows == [], f"Attachments leaked after pre-execution failure: {rows}"


@pytest.mark.asyncio
async def test_direct_retain_batch_async_failure_discards_ingress_attachments(memory, monkeypatch):
    """A direct call to retain_batch_async with ingress_attachments must clean up on failure."""
    from hindsight_api.engine.retain.attachment_content import RetainAttachment
    from hindsight_api.models import RequestContext

    bank_id = f"fail-direct-{uuid.uuid4().hex[:8]}"
    ctx = RequestContext(tenant_id="public")

    # Store attachment through memory engine
    att = RetainAttachment(
        attachment_hash=compute_attachment_hash(PNG_BYTES),
        media_type="image/png",
        data=PNG_BYTES,
        block_index=0,
        kind="image",
    )
    ingress = await memory.store_retain_attachments(
        bank_id=bank_id,
        images_by_document={"doc_direct": [att]},
        request_context=ctx,
    )
    assert ingress

    rows_before = await _bank_attachment_rows(memory, bank_id)
    assert len(rows_before) == 1

    # Simulate retain_batch_async failure during execution
    async def _failing_retain(*args, **kwargs):
        raise RuntimeError("Direct execution crashed")

    monkeypatch.setattr(memory, "_run_retain_execution", _failing_retain)

    with pytest.raises(RuntimeError, match="Direct execution crashed"):
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "placeholder text", "document_id": "doc_direct"}],
            request_context=ctx,
            ingress_attachments=ingress,
        )

    # Attachments must be cleaned up
    rows_after = await _bank_attachment_rows(memory, bank_id)
    assert rows_after == [], f"Attachments leaked after direct retain failure: {rows_after}"

    storage_key = attachment_storage_key(bank_id, "doc_direct", compute_attachment_hash(PNG_BYTES))
    assert not await memory._file_storage.exists(storage_key)
    with pytest.raises(FileNotFoundError):
        await memory._file_storage.retrieve(storage_key)


@pytest.mark.asyncio
async def test_multigroup_retain_partial_failure_preserves_successful_attachments(api_client, memory, monkeypatch):
    """When a multi-strategy retain partially succeeds, the successful group's attachments
    are kept, while the failing group's attachments are discarded.
    """
    bank_id = f"fail-multi-{uuid.uuid4().hex[:8]}"
    real_run = memory._run_retain_execution

    async def _conditional_retain(*args, **kwargs):
        contents = kwargs.get("contents") or (args[1] if len(args) > 1 else [])
        for item in contents:
            if item.get("document_id") == "doc2":
                raise RuntimeError("doc2 crashed")
        return await real_run(*args, **kwargs)

    monkeypatch.setattr(memory, "_run_retain_execution", _conditional_retain)

    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                {"content": [_image_block(PNG_BYTES)], "strategy": "s1", "document_id": "doc1"},
                {"content": [_image_block(OTHER_PNG_BYTES)], "strategy": "s2", "document_id": "doc2"},
            ],
            "async": False,
        },
    )
    assert response.status_code == 500

    # doc1 succeeded and must keep its attachment; doc2 failed and its attachment must be discarded
    rows = await _bank_attachment_rows(memory, bank_id)
    doc_ids = {r["document_id"] for r in rows}
    assert doc_ids == {"doc1"}, f"Expected only doc1 attachments to survive, got: {rows}"

    doc1_key = attachment_storage_key(bank_id, "doc1", compute_attachment_hash(PNG_BYTES))
    doc2_key = attachment_storage_key(bank_id, "doc2", compute_attachment_hash(OTHER_PNG_BYTES))
    assert await memory._file_storage.exists(doc1_key)
    assert not await memory._file_storage.exists(doc2_key)


@pytest.mark.asyncio
async def test_single_batch_multi_document_partial_failure_preserves_successful_attachments(
    api_client, memory, monkeypatch
):
    """When a single retain batch containing multiple documents partially fails,
    the successful document's attachments are kept, while the failing document's
    attachments are discarded.
    """
    bank_id = f"fail-single-multi-{uuid.uuid4().hex[:8]}"
    real_internal = memory._retain_batch_async_internal

    async def _conditional_internal(*args, **kwargs):
        doc_id = kwargs.get("document_id")
        if doc_id == "doc2":
            raise RuntimeError("doc2 failed in retain internal")
        return await real_internal(*args, **kwargs)

    monkeypatch.setattr(memory, "_retain_batch_async_internal", _conditional_internal)

    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                {"content": [_image_block(PNG_BYTES)], "document_id": "doc1"},
                {"content": [_image_block(OTHER_PNG_BYTES)], "document_id": "doc2"},
                # Include a second item for doc1 so has_shared_document is True and groups loop runs
                {"content": "second turn for doc1", "document_id": "doc1"},
            ],
            "async": False,
        },
    )
    assert response.status_code == 500

    # doc1 succeeded and must keep its attachment; doc2 failed and its attachment must be discarded
    rows = await _bank_attachment_rows(memory, bank_id)
    doc_ids = {r["document_id"] for r in rows}
    assert doc_ids == {"doc1"}, f"Expected only doc1 attachments to survive, got: {rows}"

    doc1_key = attachment_storage_key(bank_id, "doc1", compute_attachment_hash(PNG_BYTES))
    doc2_key = attachment_storage_key(bank_id, "doc2", compute_attachment_hash(OTHER_PNG_BYTES))
    assert await memory._file_storage.exists(doc1_key)
    assert not await memory._file_storage.exists(doc2_key)


@pytest.mark.asyncio
async def test_async_retain_worker_failure_discards_attachments(api_client, memory, monkeypatch):
    """When an async retain operation permanently fails in the worker,
    the ingress attachments are discarded and not leaked.
    """
    bank_id = f"fail-async-worker-{uuid.uuid4().hex[:8]}"

    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [{"content": [_image_block(PNG_BYTES)], "document_id": "doc_async"}],
            "async": True,
        },
    )
    assert response.status_code == 200

    # Verify attachment was written at ingress
    rows_before = await _bank_attachment_rows(memory, bank_id)
    assert len(rows_before) == 1

    # Simulate worker task execution failure with a permanent error
    async def _failing_retain(*args, **kwargs):
        raise RuntimeError("Worker permanent failure")

    monkeypatch.setattr(memory, "_run_retain_execution", _failing_retain)

    # Fetch child operation from async_operations
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        child_row = await conn.fetchrow(
            "SELECT operation_id, task_payload FROM async_operations WHERE bank_id = $1 AND operation_type = 'retain'",
            bank_id,
        )
    assert child_row is not None
    task_payload = child_row["task_payload"]
    if isinstance(task_payload, str):
        task_payload = json.loads(task_payload)

    # Assert ingress_attachments is present in task_payload
    assert "ingress_attachments" in task_payload
    assert "doc_async" in task_payload["ingress_attachments"]

    # Execute task with retry_count equal to worker_max_retries (permanent failure)
    task_payload["_retry_count"] = 3
    with pytest.raises(RuntimeError, match="Worker permanent failure"):
        await memory.execute_task(task_payload)

    # Attachments must be cleaned up after permanent failure
    rows_after = await _bank_attachment_rows(memory, bank_id)
    assert rows_after == [], f"Attachments leaked after worker failure: {rows_after}"

    doc_key = attachment_storage_key(bank_id, "doc_async", compute_attachment_hash(PNG_BYTES))
    assert not await memory._file_storage.exists(doc_key)


@pytest.mark.asyncio
async def test_async_retain_worker_transient_failure_preserves_attachments_for_retry(api_client, memory, monkeypatch):
    """When an async retain operation hits a transient failure under the retry limit,
    ingress attachments are preserved so subsequent retry attempts can load them.
    """
    bank_id = f"retry-async-worker-{uuid.uuid4().hex[:8]}"

    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [{"content": [_image_block(PNG_BYTES)], "document_id": "doc_retry"}],
            "async": True,
        },
    )
    assert response.status_code == 200

    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        child_row = await conn.fetchrow(
            "SELECT operation_id, task_payload FROM async_operations WHERE bank_id = $1 AND operation_type = 'retain'",
            bank_id,
        )
    assert child_row is not None
    task_payload = child_row["task_payload"]
    if isinstance(task_payload, str):
        task_payload = json.loads(task_payload)

    # Simulate transient failure on retry_count = 0
    async def _failing_retain(*args, **kwargs):
        raise RuntimeError("Transient LLM error")

    monkeypatch.setattr(memory, "_run_retain_execution", _failing_retain)

    task_payload["_retry_count"] = 0
    with pytest.raises(RetryTaskAt):
        await memory.execute_task(task_payload)

    # Attachments must still exist for the upcoming retry
    rows_retry = await _bank_attachment_rows(memory, bank_id)
    assert len(rows_retry) == 1, "Attachments were prematurely discarded on transient failure"
    doc_key = attachment_storage_key(bank_id, "doc_retry", compute_attachment_hash(PNG_BYTES))
    assert await memory._file_storage.exists(doc_key)


@pytest.mark.asyncio
async def test_subbatch_distinct_documents_partial_failure_preserves_successful_attachments(
    api_client, memory, monkeypatch
):
    """When distinct documents (has_shared_document == False) are split across
    sub-batches and an intermediate sub-batch fails, documents committed by
    earlier sub-batches preserve their attachments, while failing documents have
    their attachments discarded.
    """
    from hindsight_api.config import get_config

    bank_id = f"subbatch-distinct-{uuid.uuid4().hex[:8]}"
    # Force each item into its own sub-batch by setting a tiny token limit
    monkeypatch.setattr(get_config()._config, "retain_batch_tokens", 10)

    real_internal = memory._retain_batch_async_internal

    async def _conditional_internal(*args, **kwargs):
        contents = kwargs.get("contents") or (args[1] if len(args) > 1 else [])
        if any(c.get("document_id") == "doc2" for c in contents):
            raise RuntimeError("doc2 sub-batch failure")
        return await real_internal(*args, **kwargs)

    monkeypatch.setattr(memory, "_retain_batch_async_internal", _conditional_internal)

    response = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                {
                    "content": [_image_block(PNG_BYTES), _text_block("doc1 long text with enough tokens")],
                    "document_id": "doc1",
                },
                {
                    "content": [_image_block(OTHER_PNG_BYTES), _text_block("doc2 long text with enough tokens")],
                    "document_id": "doc2",
                },
            ],
            "async": False,
        },
    )
    assert response.status_code == 500

    # doc1 committed in sub-batch 1 and must keep its attachment; doc2 failed in sub-batch 2
    rows = await _bank_attachment_rows(memory, bank_id)
    doc_ids = {r["document_id"] for r in rows}
    assert doc_ids == {"doc1"}, f"Expected only doc1 attachments to survive, got: {rows}"

    doc1_key = attachment_storage_key(bank_id, "doc1", compute_attachment_hash(PNG_BYTES))
    doc2_key = attachment_storage_key(bank_id, "doc2", compute_attachment_hash(OTHER_PNG_BYTES))
    assert await memory._file_storage.exists(doc1_key)
    assert not await memory._file_storage.exists(doc2_key)


@pytest.mark.asyncio
async def test_submit_async_retain_cancellation_discards_attachments(memory, monkeypatch):
    """Cancellation during submit_async_retain must clean up ingress attachments."""
    from hindsight_api.engine.retain.attachment_content import RetainAttachment
    from hindsight_api.models import RequestContext

    bank_id = f"cancel-async-{uuid.uuid4().hex[:8]}"
    ctx = RequestContext(tenant_id="public")

    att = RetainAttachment(
        attachment_hash=compute_attachment_hash(PNG_BYTES),
        media_type="image/png",
        data=PNG_BYTES,
        block_index=0,
        kind="image",
    )
    ingress = await memory.store_retain_attachments(
        bank_id=bank_id,
        images_by_document={"doc_cancel": [att]},
        request_context=ctx,
    )
    assert ingress

    rows_before = await _bank_attachment_rows(memory, bank_id)
    assert len(rows_before) == 1

    # Simulate cancellation during bank creation in submit_async_retain
    async def _cancelled_ensure(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(memory, "_ensure_bank_exists", _cancelled_ensure)

    with pytest.raises(asyncio.CancelledError):
        await memory.submit_async_retain(
            bank_id=bank_id,
            contents=[{"content": "placeholder text", "document_id": "doc_cancel"}],
            request_context=ctx,
            ingress_attachments=ingress,
        )

    # Attachments must be cleaned up despite cancellation
    rows_after = await _bank_attachment_rows(memory, bank_id)
    assert rows_after == [], f"Attachments leaked after cancellation: {rows_after}"
    doc_key = attachment_storage_key(bank_id, "doc_cancel", compute_attachment_hash(PNG_BYTES))
    assert not await memory._file_storage.exists(doc_key)

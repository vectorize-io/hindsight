"""An uploaded file whose extraction fails still says so, tied to the upload.

A file upload is two operations: a conversion that turns the file into text,
then a retain it queues for the same document. The conversion finishes — and is
marked completed — before extraction even starts. A caller watching the upload
that only reads the conversion sees success while extraction is still running,
and keeps seeing it after extraction fails: no document, no error (#5310).

So the retain the conversion queues has to name the upload it belongs to — the
same document_id and filename — or nothing can tell the two apart from the
operations list.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.asyncio


async def test_a_failed_extraction_is_traceable_to_its_upload(client, llm, bank_id, tmp_path):
    # Unlike a text retain, a file upload does not create the bank on first use.
    await client.acreate_bank(bank_id=bank_id, name="Uploads")
    llm.on_step("extract_facts").returns_text("this is not the json you asked for")

    upload = tmp_path / "notes.txt"
    upload.write_text("Alice moved to Berlin.")
    await client.aretain_files(bank_id=bank_id, files=[upload], files_metadata=[{"document_id": "notes"}])

    failed = None
    for _ in range(60):
        listing = await client.operations.list_operations(bank_id, status="failed", limit=100)
        if listing.operations:
            failed = listing.operations[0]
            break
        await asyncio.sleep(1)
    assert failed is not None, "the failed extraction never showed up"

    conversions = await client.operations.list_operations(bank_id, type="file_convert_retain", limit=100)
    [conversion] = conversions.operations
    assert conversion.status == "completed"
    assert (conversion.document_id, conversion.filename) == ("notes", "notes.txt")

    assert failed.task_type == "retain"
    assert (failed.document_id, failed.filename) == ("notes", "notes.txt")
    assert failed.error_message

"""What identifies an attachment, and what merely describes a reference to it.

An attachment is identified by its **content**: sha256 of the decoded bytes,
prefixed to the short id that document text carries. The same bytes in two
documents therefore resolve to the same id and the same hash, and re-retaining an
unchanged document is free.

An attachment nonetheless *belongs* to one document — two documents carrying the
same PDF hold a row and a copy each — and the *filename* is a property of that
reference rather than of the bytes: the same PDF is "policy-v1.pdf" in one
document and "escalation-runbook.pdf" in another. These tests pin both halves:
one identity, one name per document.
"""

import base64
import uuid

import pytest

from hindsight_api.engine.retain.attachment_content import compute_attachment_hash, short_attachment_id

PDF = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n"
PDF_ID = short_attachment_id(compute_attachment_hash(PDF))
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _file_block(filename: str, data: bytes = PDF, media_type: str = "application/pdf") -> dict:
    return {
        "type": "file",
        "source": {"type": "base64", "media_type": media_type, "data": base64.b64encode(data).decode()},
        "filename": filename,
    }


async def _retain(client, bank_id: str, document_id: str, *blocks) -> None:
    response = await client.post(
        f"/v1/default/banks/{bank_id}/memories",
        json={
            "items": [
                {
                    "content": [{"type": "text", "text": "See the attached policy."}, *blocks],
                    "document_id": document_id,
                }
            ],
            "async": False,
        },
    )
    assert response.status_code == 200, response.text


async def _attachments_of(client, bank_id: str, document_id: str) -> list[dict]:
    response = await client.get(f"/v1/default/banks/{bank_id}/documents/{document_id}")
    assert response.status_code == 200, response.text
    return response.json().get("attachments") or []


@pytest.mark.asyncio
async def test_identical_content_in_two_documents_has_one_identity(api_client, memory):
    """Identity is the content hash, so both documents name the same attachment.

    Each holds its own copy of the bytes — see test_attachment_lifecycle.py — but
    what the id and the hash *mean* must not depend on which document you ask.
    """
    bank_id = f"ident-{uuid.uuid4().hex[:8]}"
    await _retain(api_client, bank_id, "doc-a", _file_block("policy-v1.pdf"))
    await _retain(api_client, bank_id, "doc-b", _file_block("escalation-runbook.pdf"))

    a = await _attachments_of(api_client, bank_id, "doc-a")
    b = await _attachments_of(api_client, bank_id, "doc-b")

    assert [x["id"] for x in a] == [PDF_ID]
    assert [x["id"] for x in b] == [PDF_ID], "the same bytes must resolve to the same attachment"
    assert a[0]["hash"] == b[0]["hash"]
    assert a[0]["byte_size"] == b[0]["byte_size"] == len(PDF)


@pytest.mark.asyncio
async def test_each_document_keeps_the_filename_it_supplied(api_client, memory):
    """The name describes the reference, so it must not be shared across documents.

    Before the filename moved onto the document's own attachment row, `doc-b`
    reported "policy-v1.pdf" — the insert is ON CONFLICT DO NOTHING, and the
    conflict was on the content hash alone, so the second document's name was
    silently discarded.
    """
    bank_id = f"ident-{uuid.uuid4().hex[:8]}"
    await _retain(api_client, bank_id, "doc-a", _file_block("policy-v1.pdf"))
    await _retain(api_client, bank_id, "doc-b", _file_block("escalation-runbook.pdf"))

    a = await _attachments_of(api_client, bank_id, "doc-a")
    b = await _attachments_of(api_client, bank_id, "doc-b")

    assert a[0]["filename"] == "policy-v1.pdf"
    assert b[0]["filename"] == "escalation-runbook.pdf"


@pytest.mark.asyncio
async def test_re_retaining_a_document_keeps_its_own_name(api_client, memory):
    """A second write of the same document must not inherit the other's name."""
    bank_id = f"ident-{uuid.uuid4().hex[:8]}"
    await _retain(api_client, bank_id, "doc-a", _file_block("policy-v1.pdf"))
    await _retain(api_client, bank_id, "doc-b", _file_block("escalation-runbook.pdf"))
    await _retain(api_client, bank_id, "doc-b", _file_block("escalation-runbook.pdf"))

    b = await _attachments_of(api_client, bank_id, "doc-b")
    assert b[0]["filename"] == "escalation-runbook.pdf"


@pytest.mark.asyncio
async def test_different_content_is_a_different_attachment(api_client, memory):
    """The converse: same filename, different bytes, two attachments."""
    bank_id = f"ident-{uuid.uuid4().hex[:8]}"
    await _retain(api_client, bank_id, "doc-a", _file_block("report.pdf"))
    await _retain(api_client, bank_id, "doc-b", _file_block("report.pdf", data=PNG, media_type="image/png"))

    a = await _attachments_of(api_client, bank_id, "doc-a")
    b = await _attachments_of(api_client, bank_id, "doc-b")

    assert a[0]["id"] != b[0]["id"], "different bytes must not collapse onto one attachment"
    assert a[0]["media_type"] == "application/pdf"
    assert b[0]["media_type"] == "image/png"

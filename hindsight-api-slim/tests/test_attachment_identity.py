"""What identifies an attachment, and what merely describes a reference to it.

An attachment is identified by its **content**: sha256 of the decoded bytes,
prefixed to the short id that document text carries. Upload the same bytes twice
and there is one blob, one row, one placeholder — which is the whole point of
content-addressing, and what makes re-ingesting an unchanged document free.

But the *filename* is not a property of the bytes. The same PDF can be attached
to one document as "policy-v1.pdf" and to another as "escalation-runbook.pdf",
and the blob row is written once, for whichever document arrived first. Keeping
the name on the blob therefore made the first upload's name win everywhere —
so it lives on the document edge instead, and these tests pin both halves.
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
async def test_identical_content_in_two_documents_is_one_attachment(api_client, memory):
    """Identity is the content hash, so the second upload stores nothing new."""
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

    Before the filename moved to the document edge, `doc-b` reported
    "policy-v1.pdf" — the insert is ON CONFLICT DO NOTHING on the content hash,
    so the second document's metadata was silently discarded.
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

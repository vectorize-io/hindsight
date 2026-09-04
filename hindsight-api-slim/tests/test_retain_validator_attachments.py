"""What an extension's retain hook can see and enforce about inline attachments.

An extension exists to make policy decisions before content lands. For
attachments that means two things, and neither worked when they were first
written:

  * it must be able to *see* them — a size quota or a media-type rule cannot be
    written against placeholder tokens alone; and
  * refusing the retain must actually keep the bytes out, which is not automatic
    because they are written at the API ingress before the hook can run (the
    async path deliberately carries only placeholder text through its operation
    row, so the bytes cannot travel with it).

Both are asserted here against the real HTTP surface, because both were true of
the code as written and invisible from the inside.
"""

import base64
import uuid

import pytest

from hindsight_api.engine.retain.attachment_content import compute_attachment_hash, short_attachment_id
from hindsight_api.extensions.operation_validator import ValidationResult

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_ID = short_attachment_id(compute_attachment_hash(PNG))
PDF = b"%PDF-1.4 tiny"
PDF_ID = short_attachment_id(compute_attachment_hash(PDF))


class _Validator:
    """Records what validate_retain saw; refuses when asked to."""

    def __init__(self, allow: bool = True) -> None:
        self._allow = allow
        self.seen: list = []

    async def validate_retain(self, ctx) -> ValidationResult:
        self.seen = list(ctx.attachments)
        if self._allow:
            return ValidationResult(allowed=True)
        return ValidationResult(allowed=False, reason="policy: no attachments", status_code=403)

    def __getattr__(self, name):
        async def permissive(*a, **k):
            return ValidationResult(allowed=True)

        return permissive


def _item() -> dict:
    return {
        "content": [
            {"type": "text", "text": "To reset the VPN, click the button shown:"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(PNG).decode()},
            },
            {
                "type": "file",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.b64encode(PDF).decode(),
                },
                "filename": "policy.pdf",
            },
        ],
        "document_id": "d1",
    }


async def _retain(client, bank_id: str):
    return await client.post(f"/v1/default/banks/{bank_id}/memories", json={"items": [_item()], "async": False})


@pytest.mark.asyncio
async def test_the_hook_sees_each_attachment_described(api_client, memory):
    """Enough to write a size quota or a media-type rule against."""
    validator = _Validator(allow=True)
    memory._operation_validator = validator
    try:
        response = await _retain(api_client, f"val-{uuid.uuid4().hex[:8]}")
    finally:
        memory._operation_validator = None
    assert response.status_code == 200, response.text

    by_id = {a.short_id: a for a in validator.seen}
    assert set(by_id) == {PNG_ID, PDF_ID}, "the hook did not see every attachment"

    image = by_id[PNG_ID]
    assert image.media_type == "image/png"
    assert image.byte_size == len(PNG), "a size quota needs the real byte count"
    assert image.kind == "image"
    assert image.filename is None, "the caller gave this one no name"

    document = by_id[PDF_ID]
    assert document.media_type == "application/pdf"
    assert document.byte_size == len(PDF)
    assert document.kind == "file"
    assert document.filename == "policy.pdf"


@pytest.mark.asyncio
async def test_a_text_only_retain_reports_no_attachments(api_client, memory):
    validator = _Validator(allow=True)
    memory._operation_validator = validator
    try:
        response = await api_client.post(
            f"/v1/default/banks/val-{uuid.uuid4().hex[:8]}/memories",
            json={"items": [{"content": "just prose"}], "async": False},
        )
    finally:
        memory._operation_validator = None
    assert response.status_code == 200, response.text
    assert validator.seen == []


@pytest.mark.asyncio
async def test_refusing_the_retain_discards_the_bytes(api_client, memory):
    """The point of a content policy: refused bytes must not remain fetchable.

    They are written before the hook runs, and nothing else would ever remove
    them — reclaim is driven by document deletion, and a refused retain creates
    no document.
    """
    bank_id = f"val-{uuid.uuid4().hex[:8]}"
    memory._operation_validator = _Validator(allow=False)
    try:
        response = await _retain(api_client, bank_id)
    finally:
        memory._operation_validator = None
    assert response.status_code == 403, response.text

    for attachment_id in (PNG_ID, PDF_ID):
        fetched = await api_client.get(f"/v1/default/banks/{bank_id}/attachments/{attachment_id}")
        assert fetched.status_code == 404, (
            f"attachment {attachment_id} survived a refused retain — a content policy "
            f"that rejects a retain cannot keep its bytes out of the bank"
        )

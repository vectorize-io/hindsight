"""A retain the write scope refuses leaves none of its attachments behind (#5416).

Attachment bytes are stored at the API ingress, before any check runs, so a refusal
has to take them back out — nothing else ever would: reclaim follows document
deletion, and a refused retain creates no document. The operation validator's own
refusal did this; the write-scope check that runs right after it (here, Dan tagging a
memory `user:kate`) returned its 403 and kept the photo, fetchable by anyone who can
read the bank.

The second half is the control: the same photo, retained under Dan's own tag, is
fetchable — so the 404 before it is the reclaim working, not the lookup failing.
"""

from __future__ import annotations

import base64
import hashlib

import pytest
from hindsight_client import Hindsight
from hindsight_client_api.exceptions import ForbiddenException, NotFoundException

from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.waiting import wait_until_settled

pytestmark = pytest.mark.asyncio

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
# The attachment id is the content hash's first 12 hex digits.
PNG_ID = hashlib.sha256(PNG).hexdigest()[:12]

CONTENT = [
    {"type": "text", "text": "Dan sent a photo of the whiteboard:"},
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(PNG).decode()}},
]


@pytest.fixture
async def clients(scoped_server):
    admin = Hindsight(base_url=scoped_server.url)
    dan = Hindsight(base_url=scoped_server.url, api_key="dan")
    yield admin, dan
    for client in (admin, dan):
        await client.aclose()


@pytest.mark.parametrize("retain_async", [False, True], ids=["sync", "async"])
async def test_a_write_scope_refusal_keeps_no_attachment(clients, llm, bank_id, retain_async):
    admin, dan = clients
    await admin.acreate_bank(bank_id=bank_id, name="Whiteboards")

    with pytest.raises(ForbiddenException):
        await dan.aretain(
            bank_id=bank_id, content=CONTENT, document_id="kates", tags=["user:kate"], retain_async=retain_async
        )
    with pytest.raises(NotFoundException):
        await admin.memory.get_bank_attachment(bank_id, PNG_ID)

    llm.on_step("extract_facts").returns(extracted(fact("Dan photographed the whiteboard", who="Dan", entities=["Dan"])))
    llm.on_step("consolidate").returns(consolidation())
    await dan.aretain(bank_id=bank_id, content=CONTENT, document_id="dans", tags=["user:dan"])
    await wait_until_settled(admin, bank_id)

    assert await admin.memory.get_bank_attachment(bank_id, PNG_ID) == PNG

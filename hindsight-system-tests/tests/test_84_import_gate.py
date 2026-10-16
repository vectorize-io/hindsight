"""A metering deployment can refuse an import before it lands, by what the archive holds.

Importing an archive brings documents in without running extraction, so a deployment that
charges per document had no say: nothing asked it before the archive was queued. Now the
server's validator sees every document of the archive at submission, and a refusal reaches
the caller instead of a background task. The server's validator
(``server_extensions/tag_scope_validator.py``) refuses, with a 402, any archive holding a
document that mentions ``OVER-QUOTA``.

The composition under test is transfer meeting the operation validator: the gate must see
the archive's real document text on both doors in (a document import and a whole-bank
restore), let an ordinary archive through, and leave nothing behind when it refuses.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
from hindsight_client import Hindsight
from hindsight_client_api.api.document_transfer_api import DocumentTransferApi
from hindsight_client_api.exceptions import ApiException

from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.waiting import wait_until_settled

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def banks(scoped_server, llm, bank_id) -> AsyncIterator[tuple[Hindsight, str, str, str]]:
    """An admin client, an ordinary bank (``bank_id``), a bank holding an over-quota document,
    and an empty neighbour to import into."""
    admin = Hindsight(base_url=scoped_server.url)
    heavy = f"systest-{uuid.uuid4().hex[:12]}"
    neighbour = f"systest-{uuid.uuid4().hex[:12]}"
    llm.on_step("extract_facts").returns(extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice"])))
    llm.on_step("consolidate").returns(consolidation())
    await admin.aretain(bank_id=bank_id, content="Alice moved to Berlin.", document_id="d1")
    await admin.aretain(bank_id=heavy, content="OVER-QUOTA: Alice moved to Berlin.", document_id="d2")
    await admin.acreate_bank(neighbour)
    await wait_until_settled(admin, bank_id)
    await wait_until_settled(admin, heavy)
    yield admin, bank_id, heavy, neighbour
    for bank in (heavy, neighbour):
        await admin.banks.delete_bank(bank)
    await admin.aclose()


async def test_the_import_gate_refuses_an_archive_by_its_documents(banks):
    admin, ordinary, heavy, neighbour = banks
    documents = DocumentTransferApi(admin.documents.api_client)

    # An ordinary archive passes the gate and lands.
    await documents.import_documents(neighbour, await admin.aexport_documents(bank_id=ordinary))
    await wait_until_settled(admin, neighbour)

    # The over-quota one is refused at submission, with the validator's status, and the
    # neighbour keeps only what it had.
    with pytest.raises(ApiException) as refused:
        await documents.import_documents(neighbour, await admin.aexport_documents(bank_id=heavy))
    assert refused.value.status == 402
    listing = await admin.documents.list_documents(neighbour)
    assert [d.id for d in listing.items] == ["d1"]

    # A whole-bank restore goes through the same gate, and a refusal creates no bank.
    target = f"systest-{uuid.uuid4().hex[:12]}"
    with pytest.raises(ApiException) as refused:
        await admin.aimport_bank(heavy, await admin.aexport_bank(heavy, poll_interval=0), target_bank_id=target)
    assert refused.value.status == 402
    assert [b.bank_id for b in (await admin.banks.list_banks(q=target)).banks] == []

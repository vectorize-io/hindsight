"""A key confined to one bank can import into it, and into no other bank that already exists.

Deployments isolate API keys per bank with an operation validator: every write asks
``validate_bank_write`` whether this key may touch this bank. Importing a document archive
skipped that question when the target bank already existed — the only check on the import
path was the one that runs when a bank is created — so a key scoped to its own bank could
pour an archive into anyone else's (#5137). The server's validator
(``server_extensions/tag_scope_validator.py``) lets a ``bank:<id>`` key write only bank
``<id>``.

The composition under test is transfer meeting per-key isolation: the same archive, the same
key, accepted at home and refused next door, with the neighbour left exactly as it was.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
from hindsight_client import Hindsight
from hindsight_client_api.api.bank_transfer_api import BankTransferApi
from hindsight_client_api.api.document_transfer_api import DocumentTransferApi
from hindsight_client_api.exceptions import ForbiddenException

from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.waiting import wait_until_settled

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def banks(scoped_server, llm, bank_id) -> AsyncIterator[tuple[Hindsight, str, str]]:
    """An admin client, the key's own bank (``bank_id``, holding one document) and a neighbour
    bank that already exists but is empty."""
    admin = Hindsight(base_url=scoped_server.url)
    neighbour = f"systest-{uuid.uuid4().hex[:12]}"
    llm.on_step("extract_facts").returns(extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice"])))
    llm.on_step("consolidate").returns(consolidation())
    await admin.aretain(bank_id=bank_id, content="Alice moved to Berlin.", document_id="d1")
    await admin.acreate_bank(neighbour)
    await wait_until_settled(admin, bank_id)
    yield admin, bank_id, neighbour
    await admin.banks.delete_bank(neighbour)
    await admin.aclose()


async def test_a_bank_scoped_key_imports_only_into_its_own_bank(scoped_server, banks):
    admin, own, neighbour = banks
    archive = await admin.aexport_documents(bank_id=own)
    scoped = Hindsight(base_url=scoped_server.url, api_key=f"bank:{own}")
    documents = DocumentTransferApi(scoped.documents.api_client)
    transfer = BankTransferApi(scoped.documents.api_client)
    try:
        await documents.import_documents(own, archive)
        await wait_until_settled(admin, own)

        # Both doors into a merge reach the same check: the documents endpoint and the unified
        # transfer endpoint in merge mode.
        with pytest.raises(ForbiddenException):
            await documents.import_documents(neighbour, archive)
        with pytest.raises(ForbiddenException):
            await transfer.import_bank_transfer(neighbour, archive, mode="merge")
    finally:
        await scoped.aclose()

    listing = await admin.documents.list_documents(neighbour)
    assert listing.total == 0
    assert listing.items == []

"""With HINDSIGHT_API_LLM_TRACE_DELETE_WITH_DOCUMENT on, deleting a document deletes its request history.

A client that offers "forget this conversation" deletes the document, its memories and its
observations — but every model call that read the text keeps its raw prompt and answer in the
bank's request history until the retention sweep (#6098). With the flag on, the delete takes
those rows too: the document's retain runs, and the consolidation that consumed its memories.
Another document's history in the same bank stays.

The flag is server-wide, so the story runs its own server, in its own schema so that server's
worker claims only this story's work.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from hindsight_client import Hindsight
from hindsight_client_api.api.llm_traces_api import LLMTracesApi

from hindsight_system_tests import wait_until_settled
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.server import HindsightServer, start_hindsight_server

from .conftest import BANK_PREFIX

pytestmark = pytest.mark.asyncio

SCHEMA = "systest_delete_doc_traces"


@pytest.fixture
def server(stub_server, tmp_path: Path) -> Iterator[HindsightServer]:
    started = start_hindsight_server(
        stub_url=stub_server.url,
        log_path=tmp_path / "server.log",
        extra_env={"HINDSIGHT_API_DATABASE_SCHEMA": SCHEMA, "HINDSIGHT_API_LLM_TRACE_DELETE_WITH_DOCUMENT": "true"},
    )
    yield started
    started.stop()


async def test_deleting_a_document_deletes_its_request_history(server: HindsightServer, llm):
    bank = f"{BANK_PREFIX}{uuid.uuid4().hex[:12]}"
    llm.on_step("extract_facts", contains="Lisbon").returns(extracted(fact("Alice lives in Lisbon")))
    llm.on_step("extract_facts", contains="Oslo").returns(extracted(fact("Bob lives in Oslo")))
    llm.on_step("consolidate").returns(consolidation())

    client = Hindsight(base_url=server.url)
    try:
        await client.aretain(bank_id=bank, content="Alice lives in Lisbon.", document_id="alice")
        await client.aretain(bank_id=bank, content="Bob lives in Oslo.", document_id="bob")
        await wait_until_settled(client, bank)

        traces = LLMTracesApi(client._api_client)
        # Rows are buffered and written in batches, and memory ids are patched on afterwards.
        for _ in range(50):
            alice = (await traces.list_llm_requests(bank, document_id="alice", operation="retain")).items
            bob = (await traces.list_llm_requests(bank, document_id="bob", operation="retain")).items
            if alice and bob and alice[0].metadata.get("memory_ids"):
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError(f"retain left no request history: alice={len(alice)} bob={len(bob)}")
        [alice_memory] = alice[0].metadata["memory_ids"]
        # The retained fact was consolidated, and that run is linked to it.
        for _ in range(50):
            used = (await traces.list_llm_requests(bank, memory_id=alice_memory)).items
            if any(r.operation == "consolidation" for r in used):
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError(f"no consolidation run linked to the memory: {[r.operation for r in used]}")

        await client.documents.delete_document(bank, "alice")

        assert (await traces.list_llm_requests(bank, document_id="alice")).total == 0
        assert (await traces.list_llm_requests(bank, memory_id=alice_memory)).total == 0
        assert [r.trace_id for r in (await traces.list_llm_requests(bank, document_id="bob")).items] == [
            r.trace_id for r in bob
        ]
        await client.banks.delete_bank(bank)
    finally:
        await client.aclose()

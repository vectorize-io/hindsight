"""A synchronous retain stops when its caller hangs up.

`async: false` runs the whole extraction inline, so the request *is* the work. For
a long time nothing could stop it: the disconnect middleware watched only recall
and reflect, and a sync retain creates no operation row, so there was no token to
trip and no `DELETE /operations/{id}` to call either. An abandoned one therefore
ran to completion — measured at 1880s against a single-slot LLM — holding its
provider slot the whole way and committing a document into a bank that had been
deleted 85s earlier (#4526).

This is a composition story, which is why it lives here: the mechanism (the
middleware's path gate) and the pipeline (the retain task) each work on their own.
What was broken is that they never met on this route, and the only place that is
visible is a real server answering a real client that walks away mid-request.

The caller walks away via a short client timeout — the one abandonment a published
client can express. The LLM call is held open so the hang-up lands while the
extraction is genuinely in flight, rather than racing the stub's instant answer.
"""

from __future__ import annotations

import pytest
from hindsight_client import Hindsight

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

CONTENT = "Alice moved to Berlin and plays cello."
DOCUMENT_ID = "abandoned-retain"

#: Shorter than the hold, long enough that the request is in extraction when it fires.
_CLIENT_TIMEOUT_SECONDS = 2.0


@pytest.fixture
async def impatient(hindsight_server) -> Hindsight:
    """A client that gives up mid-retain, the way a timing-out caller does."""
    client = Hindsight(base_url=hindsight_server.url, timeout=_CLIENT_TIMEOUT_SECONDS)
    yield client
    await client.aclose()


async def test_an_abandoned_sync_retain_commits_nothing(client, impatient, llm, bank_id, settled):
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    async with llm.hold("extract_facts") as held:
        with pytest.raises(BaseException) as abandoned:  # noqa: B017, PT011
            await impatient.aretain(bank_id=bank_id, content=CONTENT, document_id=DOCUMENT_ID)
        # Broad on purpose: what a client raises when it gives up is its transport's
        # business, not a contract this story should pin. The assertion that matters
        # is that it gave up at all — recorded here so a *server* error, which would
        # also abort the call, cannot masquerade as the hang-up.
        assert "timeout" in repr(abandoned.value).lower(), f"gave up for the wrong reason: {abandoned.value!r}"
        # The request really did reach extraction, so the hang-up landed mid-flight
        # rather than before the work started.
        await held.reached()

    await settled(bank_id)

    documents = await client.documents.list_documents(bank_id)
    assert documents.total == 0, "the abandoned retain committed its document anyway"
    memories = await client.memory.list_memories(bank_id, limit=100)
    assert memories.total == 0, "the abandoned retain committed memory units anyway"


async def test_a_patient_caller_still_gets_its_retain(client, llm, bank_id, settled):
    """The guard above must not be the whole story: cancelling on disconnect is only
    correct if a request nobody abandons still commits everything it extracted."""
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays cello", who="Alice", entities=["Alice", "cello"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content=CONTENT, document_id=DOCUMENT_ID)
    await settled(bank_id)

    document = await client.documents.get_document(bank_id, DOCUMENT_ID)
    assert document.original_text == CONTENT
    assert document.memory_unit_count == 2

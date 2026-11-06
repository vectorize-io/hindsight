"""An extension reviews what retain extracted and what consolidation decided, before either is
written: it can drop an item or replace its tags.

A deployment uses this to keep secrets out of a bank and to mark what it has vetted. The two
reviews meet: consolidation only ever sees the memories the retain review kept, and the
observation it writes carries the tags of both reviews. The server's validator
(``server_extensions/memory_review_validator.py``) drops anything mentioning a password and tags
the rest.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from hindsight_client import Hindsight

from hindsight_system_tests.payloads import extracted, fact, observes
from hindsight_system_tests.server import HindsightServer, start_hindsight_server
from hindsight_system_tests.waiting import wait_until_settled

pytestmark = pytest.mark.asyncio

#: Its own schema: consolidation is queued work, and another server's worker sharing the schema
#: (without this validator) could claim it.
REVIEW_SCHEMA = "systest_memory_review"
EXTENSIONS_DIR = Path(__file__).resolve().parent.parent / "server_extensions"


@pytest.fixture(scope="module")
def review_server(stub_server, tmp_path_factory: pytest.TempPathFactory) -> Iterator[HindsightServer]:
    server = start_hindsight_server(
        stub_url=stub_server.url,
        log_path=tmp_path_factory.mktemp("review-server") / "server.log",
        extra_env={
            "HINDSIGHT_API_DATABASE_SCHEMA": REVIEW_SCHEMA,
            "HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION": "memory_review_validator:MemoryReviewer",
            "PYTHONPATH": str(EXTENSIONS_DIR),
        },
    )
    yield server
    server.stop()


async def test_the_extension_drops_and_tags_memories_and_observations(review_server, llm, bank_id):
    llm.on_step("extract_facts", contains="Theodora").returns(
        extracted(
            fact("Theodora lives in Reykjavík", who="Theodora", entities=["Theodora"]),
            fact("Theodora's password is hunter2", who="Theodora", entities=["Theodora"]),
        )
    )
    llm.on_step("consolidate", contains="Theodora").answers_with(observes("Theodora is settled in Reykjavík"))

    client = Hindsight(base_url=review_server.url)
    try:
        await client.aretain(
            bank_id=bank_id,
            content="Theodora lives in Reykjavík. Her password is hunter2.",
            tags=["user:theodora"],
        )
        await wait_until_settled(client, bank_id)
        memories = await client.memory.list_memories(bank_id, limit=100)
    finally:
        # This server's own schema: the shared ``bank_id`` cleanup deletes on the default server.
        await client.banks.delete_bank(bank_id)
        await client.aclose()

    stored = sorted((m.fact_type, m.text, sorted(m.tags or [])) for m in memories.items)
    assert len(stored) == 2, stored
    (obs_type, obs_text, obs_tags), (fact_type, fact_text, fact_tags) = stored
    assert (obs_type, obs_text) == ("observation", "Theodora is settled in Reykjavík")
    # The observation inherits its source memory's tags, then the consolidation review adds its own.
    assert obs_tags == ["reviewed", "reviewed:observation", "user:theodora"]
    assert fact_type == "world" and "Reykjavík" in fact_text and "password" not in fact_text
    assert fact_tags == ["reviewed", "user:theodora"]

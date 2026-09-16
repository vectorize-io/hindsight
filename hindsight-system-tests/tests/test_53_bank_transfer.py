"""A whole bank can be copied, and the copy is a bank of its own.

The document transfer next door moves memories. This moves the *bank*: its
config, its mental models, its directives — and then has to keep the copy and
the original apart forever after. Both halves are easy to get wrong in ways that
look fine at the moment they happen.

The copy arriving empty of everything except facts is the first failure: the
rows that describe a bank (directives, webhooks) are keyed by ids that are
unique across the whole schema, so a copy made on the same instance collided
with the original and wrote nothing — while reporting every row as imported. The
counts said 1, the bank had none, and nobody finds that until they wonder why
the copied agent stopped following its own rules.

The second failure is the copy staying attached to the original. A bank carries
queued work and webhook history in the same table the worker reads its queue
from, so a restored bank can arrive already holding jobs — pointed at the
original's endpoints. What the copy owes is nothing: whatever was in flight when
the export ran belongs to the bank that was exported.

So the assertions here are about separateness as much as fidelity.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import pytest
from hindsight_client_api.exceptions import ApiException

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

CELLO = "Nadia plays the cello | Involving: Nadia"
TOUR = "Nadia toured Japan in spring | Involving: Nadia"
DIRECTIVE_NAME = "house-style"


@pytest.fixture
async def source_bank(client, llm, bank_id, settled) -> str:
    """A bank with facts, a directive, and a per-bank config override."""
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Nadia plays the cello", who="Nadia", entities=["Nadia"]),
            fact("Nadia toured Japan in spring", who="Nadia", entities=["Nadia", "Japan"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    reflect_loop(llm, answer="Nadia is a cellist.")

    await client.acreate_bank(bank_id=bank_id, name="Nadia")
    await client.aretain(
        bank_id=bank_id,
        content="Nadia plays the cello. Nadia toured Japan in spring.",
        document_id="d1",
    )
    await client.acreate_directive(bank_id, name=DIRECTIVE_NAME, content="Answer in two sentences.")
    await settled(bank_id)
    return bank_id


@pytest.fixture
async def copy_bank(client) -> AsyncIterator[str]:
    bank = f"systest-{uuid.uuid4().hex[:12]}"
    yield bank
    await client.banks.delete_bank(bank)


async def _restore(client, source: str, archive: bytes, target: str) -> dict:
    """Restore ``archive`` into ``target`` and wait for the operation to finish.

    The restore is a background operation — it re-embeds every fact — so the
    submit returns an id, not an outcome. Polling here rather than in the wrapper
    keeps the wrapper honest about that; the story wants the finished state.
    """
    operation_id = await client.aimport_bank(source, archive, target_bank_id=target)
    deadline = asyncio.get_running_loop().time() + 60
    while True:
        status = await client.operations.get_operation_status(source, operation_id)
        if status.status in ("completed", "failed", "cancelled"):
            break
        assert asyncio.get_running_loop().time() < deadline, f"restore did not finish: {status}"
        await asyncio.sleep(0.1)
    assert status.status == "completed", status
    return status.result_metadata or {}


async def test_a_copied_bank_holds_the_same_memories(client, llm, source_bank, copy_bank, settled):
    archive = await client.aexport_bank(source_bank, poll_interval=0)
    await _restore(client, source_bank, archive, copy_bank)
    await settled(copy_bank)

    texts = sorted(m.text for m in (await client.memory.list_memories(copy_bank, limit=100)).items)
    assert texts == sorted([CELLO, TOUR])


async def test_a_copied_bank_keeps_its_directives(client, llm, source_bank, copy_bank, settled):
    """The regression this story exists for: directives are keyed by a
    schema-unique id, so copying a bank on one instance used to insert them
    against the still-present originals, write nothing, and report success."""
    archive = await client.aexport_bank(source_bank, poll_interval=0)
    await _restore(client, source_bank, archive, copy_bank)

    directives = await client.directives.list_directives(copy_bank)
    assert [d.name for d in directives.items] == [DIRECTIVE_NAME]


async def test_the_memories_can_be_copied_without_the_configuration(client, llm, source_bank, copy_bank, settled):
    """Cloning an agent's memory without cloning what it is configured to do —
    including webhooks, which point at the source's own consumer."""
    archive = await client.aexport_bank(source_bank, include_bank_config=False, poll_interval=0)
    await _restore(client, source_bank, archive, copy_bank)
    await settled(copy_bank)

    assert (await client.memory.list_memories(copy_bank, limit=100)).items
    assert (await client.directives.list_directives(copy_bank)).items == []


async def test_the_copy_and_the_original_evolve_apart(client, llm, source_bank, copy_bank, settled):
    """The whole point of a copy: what happens to one after the split must not
    reach the other."""
    archive = await client.aexport_bank(source_bank, poll_interval=0)
    await _restore(client, source_bank, archive, copy_bank)
    await settled(copy_bank)

    # Reset first: rules are matched in registration order, so without this the
    # fixture's extraction rule still answers and the retain below re-extracts the
    # facts the copy already has — leaving the assertion to pass or fail for a
    # reason that has nothing to do with the two banks being separate.
    llm.reset()
    llm.on_step("extract_facts").returns(extracted(fact("Nadia bought a viola", who="Nadia", entities=["Nadia"])))
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=copy_bank, content="Nadia bought a viola.", document_id="d2")
    await settled(copy_bank)

    copy_texts = [m.text for m in (await client.memory.list_memories(copy_bank, limit=100)).items]
    source_texts = [m.text for m in (await client.memory.list_memories(source_bank, limit=100)).items]
    assert any("viola" in t for t in copy_texts)
    assert not any("viola" in t for t in source_texts)


async def test_restoring_onto_an_existing_bank_is_refused(client, llm, source_bank):
    """A restore writes a whole bank; merging it into a live one would mix two
    configurations silently. The caller is told immediately, not by a failed
    background operation."""
    archive = await client.aexport_bank(source_bank, poll_interval=0)

    with pytest.raises(ApiException) as excinfo:
        await client.aimport_bank(source_bank, archive, target_bank_id=source_bank)
    assert "already exists" in str(excinfo.value)

"""A caller can correct or retract an individual memory.

Extraction is a model reading text, so it is sometimes wrong, and the fix cannot
always be "re-retain the document" — the document may be right and the derived
fact wrong. Curation is the escape hatch: edit the text of one memory, or
invalidate it outright.

Invalidation is the part worth pinning. A retracted memory has to stop answering
recalls while remaining visible to an operator asking what happened — deleting
the row would lose the audit trail, and leaving it valid would keep it shaping
answers. The state machine has to say "this was here, and it is no longer true".
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "What does Alice play?"

CELLO = "Alice plays cello | Involving: Alice"
BERLIN = "Alice moved to Berlin | Involving: Alice"
CORRECTED = "Alice plays the double bass"


async def _memory_named(client, bank: str, needle: str, *, state: str | None = None) -> dict:
    """Find one memory by a substring of its text.

    `state` is needed to see a retracted memory at all: the default listing shows
    only valid ones, which is the right default for a caller and the wrong one for
    an audit.
    """
    memories = await client.memory.list_memories(bank, limit=100, state=state)
    matches = [item for item in memories.items if needle in item.text]
    assert len(matches) == 1, f"expected exactly one memory containing {needle!r}, got {len(matches)}"
    return matches[0]


@pytest.fixture
async def curated_bank(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]),
            fact("Alice plays cello", who="Alice", entities=["Alice", "cello"]),
        )
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin and plays cello.")
    await settled(bank_id)
    return bank_id


async def test_editing_a_memory_changes_what_recall_returns(client, curated_bank):
    """The correction has to reach the read path, not just the row. An edit that
    updates the stored text but leaves the search index stale would look fixed in
    the listing and keep answering with the old wording."""
    target = await _memory_named(client, curated_bank, "cello")

    await client.memory.update_memory(curated_bank, target.id, {"text": CORRECTED})

    response = await client.arecall(bank_id=curated_bank, query=QUERY)
    texts = [r.text for r in response.results]
    assert CORRECTED in texts
    assert CELLO not in texts


async def test_an_edit_is_marked_as_such(client, curated_bank):
    """An edited memory is no longer purely what the model extracted, and the row
    says so — otherwise a later audit cannot tell derived facts from curated ones."""
    target = await _memory_named(client, curated_bank, "cello")
    assert target.edited_at is None

    await client.memory.update_memory(curated_bank, target.id, {"text": CORRECTED})

    edited = await _memory_named(client, curated_bank, "double bass")
    assert edited.edited_at is not None
    assert edited.state == "valid"


async def test_an_invalidated_memory_stops_answering_recalls(client, curated_bank):
    target = await _memory_named(client, curated_bank, "cello")

    await client.memory.update_memory(
        curated_bank, target.id, {"state": "invalidated", "reason": "Alice never played the cello"}
    )

    response = await client.arecall(bank_id=curated_bank, query=QUERY)
    assert CELLO not in [r.text for r in response.results]


async def test_an_invalidated_memory_keeps_its_paper_trail(client, curated_bank):
    """Retracted, not erased. The row survives with the reason and the timestamp,
    so "why did this stop being true" is answerable later."""
    target = await _memory_named(client, curated_bank, "cello")

    await client.memory.update_memory(
        curated_bank, target.id, {"state": "invalidated", "reason": "Alice never played the cello"}
    )

    retracted = await _memory_named(client, curated_bank, "cello", state="invalidated")
    assert retracted.state == "invalidated"
    assert retracted.invalidation_reason == "Alice never played the cello"
    assert retracted.invalidated_at is not None

    # And it is gone from the default listing, so an ordinary caller never sees it.
    remaining = await client.memory.list_memories(curated_bank, limit=100)
    assert all("cello" not in item.text for item in remaining.items)


async def test_curating_one_memory_leaves_its_neighbour_alone(client, curated_bank):
    """Both facts came from the same document and the same retain; invalidating
    one must not take the other with it."""
    target = await _memory_named(client, curated_bank, "cello")

    await client.memory.update_memory(curated_bank, target.id, {"state": "invalidated", "reason": "wrong"})

    neighbour = await _memory_named(client, curated_bank, "Berlin")
    assert neighbour.state == "valid"

    response = await client.arecall(bank_id=curated_bank, query="Where does Alice live?")
    assert BERLIN in [r.text for r in response.results]

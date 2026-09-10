"""A caller can hand Hindsight entities, and decide whether they get merged.

Entity resolution is what turns "Alice Smyth" in one document and "Alice Smith"
in another into one person with a history. It is also, by construction, the
thing that can merge two *different* people — so `resolve_entities` exists to
turn it off for callers whose entity names are already authoritative (an id from
their own system, a name they will not have spelled twice).

The distinction worth knowing, and the reason this story exists: the flag gates
*fuzzy* resolution, not normalisation. Case and whitespace differences collapse
whatever it is set to, because "Alice Smith" and "alice smith" are the same
string wearing a hat. Only a genuine variant — a typo, an alias — is what the
flag decides about. Testing it with a case difference alone shows no difference
at all and quietly proves nothing.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

EXISTING = "Alice Smith"
FUZZY_VARIANT = "Alice Smyth"
CASE_VARIANT = "alice smith"


@pytest.fixture(autouse=True)
def _extraction(llm):
    llm.on_step("extract_facts", contains="joined").returns(
        extracted(fact("Alice Smith joined the team", who=EXISTING, entities=[EXISTING]))
    )
    # The second document names nobody: the entity under test is the one the
    # caller supplies, so extraction must not introduce a competing one.
    llm.on_step("extract_facts", contains="shipped").returns(
        extracted(fact("The team shipped the release", who="the team", entities=[]))
    )
    llm.on_step("consolidate").returns(consolidation())


async def _entities(client, bank: str) -> list[tuple[str, int]]:
    listing = await client.entities.list_entities(bank)
    return sorted((e.canonical_name, e.mention_count) for e in listing.items)


async def _seed_and_supply(client, bank: str, settled, name: str, *, resolve: bool | None = None) -> None:
    """Establish `EXISTING`, then retain a second document naming `name`."""
    await client.aretain(bank_id=bank, content="Alice Smith joined the team.")
    await settled(bank)

    item: dict = {"content": "The team shipped the release.", "entities": [{"text": name, "type": "person"}]}
    if resolve is not None:
        item["resolve_entities"] = resolve
    await client.aretain_batch(bank_id=bank, items=[item])
    await settled(bank)


async def test_by_default_a_variant_is_resolved_onto_the_existing_entity(client, bank_id, settled):
    """One person, two spellings, one history. Without this a bank accumulates a
    near-duplicate per misspelling and no entity ever looks well-attested."""
    await _seed_and_supply(client, bank_id, settled, FUZZY_VARIANT)

    assert await _entities(client, bank_id) == [(EXISTING, 2)]


async def test_opting_out_stores_the_name_exactly_as_written(client, bank_id, settled):
    """For callers whose names are authoritative — an id from their own system,
    or two genuinely different people who happen to look alike. Resolution is
    lossy and irreversible, so it has to be refusable."""
    await _seed_and_supply(client, bank_id, settled, FUZZY_VARIANT, resolve=False)

    assert await _entities(client, bank_id) == [(EXISTING, 1), (FUZZY_VARIANT, 1)]


async def test_case_differences_collapse_whichever_way_the_flag_is_set(client, bank_id, settled):
    """The subtlety this story exists for.

    `resolve_entities=False` means "do not merge this onto something that merely
    resembles it" — not "treat every byte as significant". A capitalisation
    difference is the same name, so it normalises either way, and a test that
    used only a case variant to check the flag would see no difference and
    conclude the flag works when it might not.
    """
    await _seed_and_supply(client, bank_id, settled, CASE_VARIANT, resolve=False)

    assert await _entities(client, bank_id) == [(EXISTING, 2)]


async def test_an_unrelated_name_is_never_merged(client, bank_id, settled):
    """Resolution has to be conservative in the other direction too: a name that
    resembles nothing in the bank becomes its own entity rather than drifting
    onto the nearest one."""
    await _seed_and_supply(client, bank_id, settled, "Bob Jones")

    assert await _entities(client, bank_id) == [(EXISTING, 1), ("Bob Jones", 1)]

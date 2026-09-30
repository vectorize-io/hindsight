"""A bank id is migrated in phases, with the API running the whole time.

Renaming a bank is the blunt version of this: it rewrites the id on every row and
the old id stops working at the commit, so every client has to be stopped and
repointed in one step. Anything missed gets a 404 — or worse, silently creates a
fresh empty bank under the old id and starts writing into it.

An alias is the same migration without the stop. The bank keeps its id and its
data and answers to a second one, so callers move across a few at a time.

The story is the sequence, and each step is where it can go wrong:

1. a bank is filled under its original id;
2. the new id is added as an alias, and reaches the *same* memories — not an
   empty bank that happens to answer;
3. both ids work at once, which is the entire point of a phased cutover;
4. writes through either id land in one bank, so a half-migrated fleet does not
   split a bank in two — the failure that would be invisible until someone
   noticed half their memories missing;
5. the alias is removed, and the old name goes back to meaning nothing.

Every step runs through the published client against a real server, because the
resolution happens at the HTTP edge: nothing below it ever sees an alias, and
only a blackbox call proves that seam is actually in the path.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
from hindsight_client_api.models.create_bank_alias_request import CreateBankAliasRequest
from hindsight_client_api.models.set_bank_alias_primary_request import SetBankAliasPrimaryRequest

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

OLD_FACT = "The deploy window is Tuesday | Involving: Ops"
NEW_FACT = "The rollback owner is Dana | Involving: Dana"


@pytest.fixture
async def migrating_bank(client, llm, settled, bank_id) -> AsyncIterator[tuple[str, str]]:
    """A bank holding one memory under its original id, plus the id it is moving to."""
    llm.on_step("extract_facts", contains="Tuesday").returns(
        extracted(fact("The deploy window is Tuesday", who="Ops", entities=["Ops"]))
    )
    llm.on_step("extract_facts", contains="Dana").returns(
        extracted(fact("The rollback owner is Dana", who="Dana", entities=["Dana"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content="The deploy window is Tuesday.", document_id="ops")
    await settled(bank_id)

    new_id = f"{bank_id}-v2"
    await client.banks.create_bank_alias(bank_id, CreateBankAliasRequest(alias=new_id))
    yield bank_id, new_id


async def _fact_texts(client, name: str) -> list[str]:
    memories = await client.memory.list_memories(name, limit=100)
    return sorted(m.text for m in memories.items if m.state == "valid")


async def test_the_new_id_reaches_the_existing_memories(client, migrating_bank):
    """The step that makes it a migration rather than a new bank.

    An alias that resolved to nothing would still answer 200 here — with an empty
    bank, auto-created under the new id — so the assertion is on the *content*.
    """
    old, new = migrating_bank

    assert await _fact_texts(client, new) == await _fact_texts(client, old)
    assert len(await _fact_texts(client, new)) == 1


async def test_both_ids_serve_at_once(client, migrating_bank):
    """The cutover window. If the old id stopped working the moment the alias
    existed, this would be a rename with extra steps.

    Asserted as the whole recall payload rather than "Tuesday appears somewhere":
    with the LLM, embedder and reranker all stubbed, recall is a pure function of
    its input, so the two ids must return byte-identical results — which is the
    actual claim, and what a keyword check would let slide."""
    old, new = migrating_bank

    expected = ["The deploy window is Tuesday | Involving: Ops"]
    for name in (old, new):
        recalled = await client.arecall(bank_id=name, query="When is the deploy window?")
        assert [r.text for r in recalled.results] == expected, f"{name} did not recall the bank's own memory"


async def test_a_write_through_the_new_id_lands_in_the_same_bank(client, settled, migrating_bank):
    """The failure this story exists to catch.

    Mid-migration, some clients still write to the old id and some to the new one.
    If the alias resolved anywhere else, both halves would succeed and nobody would
    see an error — the bank would just quietly become two.
    """
    old, new = migrating_bank

    await client.aretain(bank_id=new, content="The rollback owner is Dana.", document_id="rollback")
    await settled(old)

    from_old = await _fact_texts(client, old)
    assert len(from_old) == 2, "a write through the alias did not land in the aliased bank"
    assert from_old == await _fact_texts(client, new)

    # And the documents agree, so it is one bank by every measure, not just facts.
    docs = await client.documents.list_documents(old)
    assert sorted(d.id for d in docs.items) == ["ops", "rollback"]


async def test_the_bank_reports_its_own_id_through_the_alias(client, migrating_bank):
    """An alias adds a way in; it never renames the bank. Responses keep naming the
    real id, which is how a caller mid-migration can tell what it actually reached."""
    old, new = migrating_bank

    aliases = await client.banks.list_bank_aliases(new)
    assert aliases.bank_id == old
    # Nothing is promoted here, so the bank is still presented under its own id.
    assert [(a.alias, a.primary) for a in aliases.aliases] == [(new, False)]


async def test_the_old_id_keeps_working_after_the_alias_is_removed(client, settled, migrating_bank):
    """The end of the migration, run backwards: dropping the alias must close only
    that door. Losing the bank here would mean the cleanup step destroys the data
    the whole exercise was preserving."""
    old, new = migrating_bank

    await client.banks.delete_bank_alias(old, new)

    assert len(await _fact_texts(client, old)) == 1
    with pytest.raises(Exception):
        await client.memory.list_memories(new, limit=100)


async def test_the_new_id_cannot_be_claimed_by_a_second_bank(client, llm, migrating_bank):
    """Two banks answering to one name would route a caller to whichever the
    database happened to return. The name is a primary key, so the second claim
    loses rather than racing."""
    old, new = migrating_bank
    other = f"systest-{uuid.uuid4().hex[:12]}"

    try:
        await client.acreate_bank(bank_id=other)
        with pytest.raises(Exception):
            await client.banks.create_bank_alias(other, CreateBankAliasRequest(alias=new))

        # The original owner is unaffected.
        assert (await client.banks.list_bank_aliases(new)).bank_id == old
    finally:
        await client.banks.delete_bank(other)


async def test_the_bank_can_be_presented_under_the_new_id(client, migrating_bank):
    """The end of a migration: the bank is shown as the id everyone now uses, while
    every row it owns is still keyed on the id it was created with.

    That split is the whole point — a display that quietly became the identity
    would be a rename, which is the thing aliases exist to avoid.
    """
    old, new = migrating_bank

    await client.banks.set_bank_alias_primary(old, new, SetBankAliasPrimaryRequest(primary=True))

    listing = await client.banks.list_banks(q=old)
    row = next(b for b in listing.banks if b.bank_id == old)
    assert row.display_alias == new, "the bank is not presented under the promoted alias"
    assert row.bank_id == old, "promoting an alias must not touch the bank's identity"

    # And the data is reachable by both, exactly as before the promotion.
    for name in (old, new):
        recalled = await client.arecall(bank_id=name, query="When is the deploy window?")
        assert [r.text for r in recalled.results] == ["The deploy window is Tuesday | Involving: Ops"]


async def test_dropping_the_shown_alias_returns_the_bank_to_its_own_id(client, migrating_bank):
    """The flag lives on the alias row, so removing the alias removes the display
    with it — there is no way to leave a bank pointing at an id that stopped
    resolving."""
    old, new = migrating_bank
    await client.banks.set_bank_alias_primary(old, new, SetBankAliasPrimaryRequest(primary=True))

    await client.banks.delete_bank_alias(old, new)

    row = next(b for b in (await client.banks.list_banks(q=old)).banks if b.bank_id == old)
    assert row.display_alias is None

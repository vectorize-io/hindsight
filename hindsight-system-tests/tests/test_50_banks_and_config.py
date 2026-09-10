"""A bank is created with an identity, and configured a field at a time.

The identity part is simple. The configuration part has a sharp edge worth
pinning: a per-bank config update must be **additive**. Sending one field means
"change this one", not "this is now the entire configuration" — and the
difference is invisible at the moment it goes wrong. Nothing errors; the bank
simply reverts every other setting to its default and starts behaving like a
fresh one. Whoever notices is debugging a behaviour change with no event that
explains it.

So the assertions here are mostly about what *did not* change.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def configured_bank(client, bank_id) -> str:
    await client.acreate_bank(
        bank_id=bank_id,
        name="Alice",
        disposition_skepticism=4,
        disposition_empathy=2,
    )
    return bank_id


async def test_a_bank_is_created_with_the_identity_it_was_given(client, configured_bank):
    listing = await client.banks.list_banks(q=configured_bank)

    banks = {b.bank_id: b for b in listing.banks}
    assert configured_bank in banks
    assert banks[configured_bank].name == "Alice"


async def test_unspecified_disposition_traits_take_their_default(client, configured_bank):
    """Two traits were set, one was not — the third must land on the documented
    midpoint rather than zero or null. A trait that silently reads 0 makes an
    agent behave in a way nobody configured."""
    listing = await client.banks.list_banks(q=configured_bank)
    bank = next(b for b in listing.banks if b.bank_id == configured_bank)

    assert bank.disposition.skepticism == 4
    assert bank.disposition.empathy == 2
    assert bank.disposition.literalism == 3


async def test_a_config_update_changes_only_the_field_it_names(client, configured_bank):
    """The clobber guard.

    One field is sent; everything else must survive. A replace-semantics update
    passes any test that only checks the field it changed, which is why the
    assertions below are about the neighbours.
    """
    before = (await client.banks.get_bank_config(configured_bank)).config
    assert before["enable_reranking"] is True
    assert before["enable_temporal_retrieval"] is True

    await client.banks.update_bank_config(configured_bank, {"updates": {"enable_reranking": False}})

    after = (await client.banks.get_bank_config(configured_bank)).config
    assert after["enable_reranking"] is False, "the field asked for did not change"
    assert after["enable_temporal_retrieval"] is True, "a neighbouring setting was reset by an unrelated update"
    assert after["disposition_skepticism"] == before["disposition_skepticism"]


async def test_a_second_update_does_not_undo_the_first(client, configured_bank):
    """Two updates in sequence, different fields. Both have to stick — the
    accumulation is what makes a config editable at all."""
    await client.banks.update_bank_config(configured_bank, {"updates": {"enable_reranking": False}})
    await client.banks.update_bank_config(configured_bank, {"updates": {"enable_temporal_retrieval": False}})

    config = (await client.banks.get_bank_config(configured_bank)).config
    assert config["enable_reranking"] is False
    assert config["enable_temporal_retrieval"] is False


async def test_resetting_returns_the_bank_to_the_server_defaults(client, configured_bank):
    """The escape hatch. A bank that cannot be put back is a bank someone is
    afraid to experiment with."""
    await client.banks.update_bank_config(configured_bank, {"updates": {"enable_reranking": False}})
    assert (await client.banks.get_bank_config(configured_bank)).config["enable_reranking"] is False

    await client.banks.reset_bank_config(configured_bank)

    assert (await client.banks.get_bank_config(configured_bank)).config["enable_reranking"] is True


async def test_a_bank_that_does_not_exist_is_a_404_not_an_empty_answer(client):
    """An unknown bank has to be distinguishable from an empty one. Answering an
    empty result turns a typo'd bank id into "you have no memories", which is the
    same shape as data loss and sends people looking for the wrong thing.
    """
    from hindsight_client_api.exceptions import NotFoundException

    with pytest.raises(NotFoundException):
        await client.banks.get_bank_config("systest-no-such-bank")

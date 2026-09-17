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
    from aiohttp import ClientResponseError
    from hindsight_client_api.exceptions import ApiException, NotFoundException

    with pytest.raises(NotFoundException):
        await client.banks.get_bank_config("systest-no-such-bank")

    # Recall is the one that matters most here: it is the read people call in a loop, and it
    # used to run the whole retrieval fan-out and then answer 200 with an empty list (#4442).
    # NotFoundException subclasses ApiException; arecall may surface aiohttp's error instead.
    with pytest.raises((ApiException, ClientResponseError)) as recalled:
        await client.arecall(bank_id="systest-no-such-bank", query="anything at all")
    assert recalled.value.status == 404


async def test_a_bank_id_over_the_byte_limit_is_refused_before_the_bank_exists(client):
    """Bank ids are capped at 192 bytes of UTF-8 (#4391). Without the cap a long id
    was accepted and then broke every file write, because storage keys percent-encode
    it past S3's 1,024-byte limit. The refusal has to come at creation, from both
    the explicit create and the implicit one a first retain performs, and leave no
    bank behind.
    """
    from aiohttp import ClientResponseError
    from hindsight_client_api.exceptions import ApiException

    too_long = "systest-" + "中" * 62  # 8 + 186 = 194 bytes
    # acreate_bank surfaces aiohttp's error, aretain the generated SDK's; both carry .status.
    with pytest.raises((ApiException, ClientResponseError)) as created:
        await client.acreate_bank(bank_id=too_long, name="too long")
    assert created.value.status == 400
    with pytest.raises((ApiException, ClientResponseError)) as retained:
        await client.aretain(bank_id=too_long, content="Alice moved to Berlin.")
    assert retained.value.status == 400

    listing = await client.banks.list_banks(q="systest-中")
    assert [b.bank_id for b in listing.banks] == []


async def test_a_bank_id_at_the_byte_limit_is_accepted(client):
    """64 CJK characters is exactly 192 bytes — the budget the limit was sized for."""
    at_limit = "中" * 64
    try:
        await client.acreate_bank(bank_id=at_limit, name="at limit")
        listing = await client.banks.list_banks(q=at_limit)
        assert [b.bank_id for b in listing.banks] == [at_limit]
    finally:
        await client.banks.delete_bank(at_limit)

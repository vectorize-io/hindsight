"""Knowledge banks and memory banks share one id space and never cross into each other's API.

A knowledge bank is a row of the same ``banks`` table, so every memory endpoint that takes a bank
id could reach it. Each side must refuse the other's banks: a retain must not land in a knowledge
bank, the memory delete must not remove one (its files and its own access hooks belong to the
knowledge side), and neither may take an id or an alias the other already holds.
"""

from __future__ import annotations

import uuid

import pytest
from hindsight_client_api.exceptions import ApiException

from hindsight_system_tests import KnowledgeBanks

pytestmark = pytest.mark.asyncio


async def test_the_two_kinds_of_bank_refuse_each_others_api(client, bank_id):
    kb = KnowledgeBanks(client)
    knowledge = f"systest-kb-{uuid.uuid4().hex[:12]}"
    assert (await kb.create(knowledge)).status == 200
    try:
        await client.acreate_bank(bank_id)

        # A memory write or delete aimed at the knowledge bank is refused, and it is still there.
        with pytest.raises(ApiException) as retain:
            await client.aretain(bank_id=knowledge, content="This must not become a memory.")
        assert retain.value.status == 400 and "is a knowledge bank" in str(retain.value.body)
        with pytest.raises(ApiException) as delete:
            await client.banks.delete_bank(knowledge)
        assert delete.value.status == 400 and "is a knowledge bank" in str(delete.value.body)
        assert (await kb.get(knowledge)).status == 200

        # The memory bank list leaves knowledge banks out; the knowledge list leaves memory banks out.
        memory_ids = [b.bank_id for b in (await client.banks.list_banks(q="systest-", limit=1000)).banks]
        assert bank_id in memory_ids and knowledge not in memory_ids
        knowledge_ids = [b["bank_id"] for b in (await kb.call("GET", "/v1/default/knowledge-banks")).body["banks"]]
        assert knowledge in knowledge_ids and bank_id not in knowledge_ids

        # Neither may be created under an id, or an alias, the other already holds.
        r = await kb.create(bank_id)
        assert r.status == 409 and "already a memory bank" in r.body["detail"]
        alias = f"systest-alias-{uuid.uuid4().hex[:8]}"
        await client.banks.create_bank_alias(bank_id, {"alias": alias})
        r = await kb.create(alias)
        assert r.status == 409 and "is an alias" in r.body["detail"]
    finally:
        await kb.delete(knowledge)

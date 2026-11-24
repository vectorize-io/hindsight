"""A client that asks for more than a page gets one full page, then the rest.

Clients have asked list endpoints for ``limit=100000``. Answering in full reads
and sends the whole collection in one request; refusing with a 422 breaks the
client. So a list endpoint trims the limit to its cap (1000), says so in the
response's ``limit``, and the client pages on from there with ``offset``.

The story needs a bank that holds more than one page. Directives are the
cheapest rows to make that many of: no LLM, no background work.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.asyncio

CAP = 1000
COUNT = CAP + 1


async def test_an_oversized_limit_returns_one_full_page_and_the_rest_is_reachable(client, bank_id):
    await client.acreate_bank(bank_id)
    gate = asyncio.Semaphore(32)

    async def create(i: int) -> None:
        async with gate:
            await client.acreate_directive(bank_id=bank_id, name=f"d{i:04d}", content=f"Rule {i}.")

    await asyncio.gather(*(create(i) for i in range(COUNT)))

    first = await client.alist_directives(bank_id=bank_id, limit=100_000)
    assert (first.limit, first.offset, first.total, len(first.items)) == (CAP, 0, COUNT, CAP)

    rest = await client.alist_directives(bank_id=bank_id, limit=100_000, offset=first.limit)
    assert (rest.limit, rest.offset, rest.total, len(rest.items)) == (CAP, CAP, COUNT, 1)

    names = {d.name for d in first.items} | {d.name for d in rest.items}
    assert names == {f"d{i:04d}" for i in range(COUNT)}

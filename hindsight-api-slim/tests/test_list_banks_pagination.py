"""Tests for pagination and search on the bank list.

``GET /v1/default/banks`` used to return every bank in the system in one
response: no limit, no offset, and a SQL query with no LIMIT clause. On an
instance with many banks that is an unbounded payload, plus per-bank config
resolution (and a live store count for non-SQL stores) for every bank rather
than the ones actually being shown.

Runs via: uv run pytest tests/test_list_banks_pagination.py -v
"""

from __future__ import annotations

import uuid

import pytest


@pytest.fixture
async def three_banks(memory, request_context):
    """Three banks sharing a unique prefix, so the search is xdist-safe."""
    prefix = f"pagebank{uuid.uuid4().hex[:8]}"
    bank_ids = [f"{prefix}_{i}" for i in range(3)]
    for bank_id in bank_ids:
        await memory.get_bank_profile(bank_id, request_context=request_context)
    try:
        yield prefix, bank_ids
    finally:
        for bank_id in bank_ids:
            await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_pages_are_disjoint_and_cover_every_match(memory, request_context, three_banks):
    prefix, bank_ids = three_banks

    first = await memory.list_banks(search_query=prefix, limit=2, offset=0, request_context=request_context)
    second = await memory.list_banks(search_query=prefix, limit=2, offset=2, request_context=request_context)

    assert first["total"] == 3
    assert first["limit"] == 2
    assert first["offset"] == 0
    assert len(first["banks"]) == 2
    assert second["total"] == 3
    assert second["offset"] == 2
    assert len(second["banks"]) == 1

    paged = [bank["bank_id"] for bank in first["banks"] + second["banks"]]
    assert len(set(paged)) == 3, f"pages overlap: {paged}"
    assert set(paged) == set(bank_ids)


@pytest.mark.asyncio
async def test_offset_past_the_end_returns_no_banks_but_the_real_total(memory, request_context, three_banks):
    prefix, _ = three_banks

    page = await memory.list_banks(search_query=prefix, limit=10, offset=3, request_context=request_context)

    assert page["banks"] == []
    assert page["total"] == 3


@pytest.mark.asyncio
async def test_limit_zero_returns_no_banks(memory, request_context, three_banks):
    prefix, _ = three_banks

    page = await memory.list_banks(search_query=prefix, limit=0, request_context=request_context)

    assert page["banks"] == []
    assert page["total"] == 3


@pytest.mark.asyncio
async def test_negative_paging_values_are_clamped(memory, request_context, three_banks):
    """The MCP tool takes limit/offset straight from a model, and the page is a Python
    slice — a negative value must not silently trim the tail."""
    prefix, _ = three_banks

    page = await memory.list_banks(search_query=prefix, limit=-1, offset=-5, request_context=request_context)

    assert page["banks"] == []
    assert page["total"] == 3


@pytest.mark.asyncio
async def test_search_matches_bank_name_case_insensitively(memory, request_context):
    bank_id = f"searchname{uuid.uuid4().hex[:8]}"
    display_name = f"Zeta {uuid.uuid4().hex[:8]}"
    try:
        await memory.get_bank_profile(bank_id, request_context=request_context)
        await memory.update_bank(bank_id, name=display_name, request_context=request_context)

        page = await memory.list_banks(search_query=display_name.upper(), request_context=request_context)

        assert [bank["bank_id"] for bank in page["banks"]] == [bank_id]
        assert page["total"] == 1
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_http_endpoint_echoes_paging_and_filters(api_client, three_banks):
    prefix, _ = three_banks

    response = await api_client.get("/v1/default/banks", params={"q": prefix, "limit": 1, "offset": 1})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 3
    assert body["limit"] == 1
    assert body["offset"] == 1
    assert len(body["banks"]) == 1
    assert body["banks"][0]["bank_id"].startswith(prefix)

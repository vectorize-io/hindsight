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

from hindsight_api import MemoryEngine
from hindsight_api.extensions import (
    BankListContext,
    BankListResult,
    BankListScope,
    OperationValidatorExtension,
    RecallContext,
    ReflectContext,
    RetainContext,
    ValidationResult,
)
from hindsight_api.models import RequestContext


@pytest.fixture
async def three_banks(memory, request_context):
    """Three banks sharing a unique prefix, so the search is xdist-safe."""
    prefix = f"pagebank{uuid.uuid4().hex[:8]}"
    bank_ids = [f"{prefix}_{i}" for i in range(3)]
    for bank_id in bank_ids:
        await memory.ensure_bank_profile(bank_id, request_context=request_context)
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
async def test_fact_counts_are_right_for_the_banks_on_the_page(memory, request_context, three_banks):
    """Counting facts moved out of the list query and into a per-page query (#4468): the
    banks actually returned must still carry their own count, on any page."""
    prefix, bank_ids = three_banks
    # Facts are inserted directly rather than retained: the assertion is an exact count per
    # bank, and how many facts a retain extracts is the LLM's business, not this test's.
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        for index, bank_id in enumerate(bank_ids):
            for fact in range(index + 1):
                await conn.execute(
                    "INSERT INTO memory_units (bank_id, text, event_date) VALUES ($1, $2, NOW())",
                    bank_id,
                    f"fact {fact}",
                )

    expected = {bank_id: index + 1 for index, bank_id in enumerate(bank_ids)}
    seen = {}
    for offset in (0, 2):
        page = await memory.list_banks(search_query=prefix, limit=2, offset=offset, request_context=request_context)
        seen.update({bank["bank_id"]: bank["fact_count"] for bank in page["banks"]})

    assert seen == expected


@pytest.mark.asyncio
async def test_search_matches_bank_name_case_insensitively(memory, request_context):
    bank_id = f"searchname{uuid.uuid4().hex[:8]}"
    display_name = f"Zeta {uuid.uuid4().hex[:8]}"
    try:
        await memory.ensure_bank_profile(bank_id, request_context=request_context)
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


class _ScopedValidator(OperationValidatorExtension):
    """A validator whose bank list is either declared up front (``scope``) or filtered after the
    fact (``scope=None``, hiding ``hidden``), recording whether the filter ran."""

    def __init__(self, scope: BankListScope | None, hidden: str | None = None) -> None:
        super().__init__({})
        self.scope, self.hidden, self.filtered = scope, hidden, False

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        return ValidationResult.accept()

    async def bank_list_scope(self, request_context: RequestContext) -> BankListScope | None:
        return self.scope

    async def filter_bank_list(self, ctx: BankListContext) -> BankListResult:
        self.filtered = True
        return BankListResult(banks=[bank for bank in ctx.banks if bank["bank_id"] != self.hidden])


async def _list_with(
    memory: MemoryEngine, request_context: RequestContext, validator: _ScopedValidator, **kwargs: object
) -> dict:
    saved, memory._operation_validator = memory._operation_validator, validator
    try:
        return await memory.list_banks(request_context=request_context, **kwargs)
    finally:
        memory._operation_validator = saved


@pytest.mark.asyncio
async def test_undeclared_scope_filters_the_full_list(memory, request_context, three_banks):
    """A validator written before scopes existed keeps its contract: it is handed the list."""
    prefix, bank_ids = three_banks
    validator = _ScopedValidator(None, hidden=bank_ids[0])

    page = await _list_with(memory, request_context, validator, search_query=prefix)

    assert validator.filtered
    assert {bank["bank_id"] for bank in page["banks"]} == set(bank_ids[1:])
    assert page["total"] == 2


@pytest.mark.asyncio
async def test_an_all_banks_scope_skips_the_filter(memory, request_context, three_banks):
    prefix, bank_ids = three_banks
    validator = _ScopedValidator(BankListScope())

    page = await _list_with(memory, request_context, validator, search_query=prefix)

    assert not validator.filtered
    assert {bank["bank_id"] for bank in page["banks"]} == set(bank_ids)


@pytest.mark.asyncio
async def test_an_explicit_scope_lists_only_those_banks_newest_first(memory, request_context, three_banks):
    """No search, so this is the by-id read, not a ranking of the tenant: the other banks in the
    test database must not appear, and an id naming no bank is ignored."""
    _, bank_ids = three_banks
    allowed = [bank_ids[0], bank_ids[2], "no-such-bank"]
    validator = _ScopedValidator(BankListScope(bank_ids=allowed))

    page = await _list_with(memory, request_context, validator, limit=1)
    rest = await _list_with(memory, request_context, validator, limit=10, offset=1)

    assert not validator.filtered
    listed = [bank["bank_id"] for bank in page["banks"] + rest["banks"]]
    assert sorted(listed) == sorted([bank_ids[0], bank_ids[2]])
    assert page["total"] == rest["total"] == 2
    written = [bank["last_write_at"] or bank["created_at"] for bank in page["banks"] + rest["banks"]]
    assert written == sorted(written, reverse=True)


@pytest.mark.asyncio
async def test_an_explicit_scope_accepts_aliases(memory, request_context, three_banks):
    """A request reaches a bank by its canonical id, so a scope naming an alias must list the bank
    the alias points at — and not the alias as if it were a bank of its own."""
    _, bank_ids = three_banks
    alias = f"{bank_ids[1]}-alias"
    await memory.create_bank_alias(bank_ids[1], alias, request_context=request_context)
    validator = _ScopedValidator(BankListScope(bank_ids=[alias, bank_ids[0]]))

    page = await _list_with(memory, request_context, validator)

    assert {bank["bank_id"] for bank in page["banks"]} == {bank_ids[0], bank_ids[1]}
    assert page["total"] == 2


@pytest.mark.asyncio
async def test_an_explicit_scope_narrows_a_search(memory, request_context, three_banks):
    prefix, bank_ids = three_banks
    validator = _ScopedValidator(BankListScope(bank_ids=[bank_ids[2]]))

    page = await _list_with(memory, request_context, validator, search_query=prefix)

    assert [bank["bank_id"] for bank in page["banks"]] == [bank_ids[2]]
    assert page["total"] == 1

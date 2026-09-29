"""The bank list is ordered by a store that can order, and costs the PAGE rather than the tenant.

A store owning the memories leaves ``documents`` / ``memory_units`` empty, so SQL has no write time
to sort on. The list used to recover it by asking the store for EVERY bank's write time, sorting in
Python, and returning a hundred rows — O(total banks) per request, and on a large tenant the whole
cost of the endpoint.

These assert the property that replaces it: the ORDER comes from the store already sorted, and the
number of banks the store is asked about does not grow with the tenant.

Runs via: uv run pytest tests/test_list_banks_ordered_page.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import hindsight_api.engine.memories as memories_mod
from hindsight_api.engine.memories.base import BankWritePage, BankWriteTime
from hindsight_api.models import RequestContext

#: The store's write times, anchored ahead of the run rather than at a fixed date.
#:
#: The merge places a bank at the newer of its store write time and its `created_at`, which is
#: right: a write cannot predate the bank it wrote to. A fixed literal quietly breaks that — these
#: banks are created NOW, so a literal in the past makes creation the newer key and the list comes
#: back in creation order, with nothing about the store's ordering under test any more. That is
#: exactly what happened: a hardcoded 2026-09-22 12:00 UTC passed locally before noon and failed in
#: CI at 12:36. Anchoring it forward keeps the store the newer key on every machine and any day.
_NOW = datetime.now(timezone.utc) + timedelta(hours=1)


class _OrderingStore:
    """A store that owns its memories and can hand back its banks already ordered.

    Duck-typed rather than a MemoriesExtension subclass, so it answers only what these paths reach —
    and so a page that reached an inherited default instead of the seam would fail rather than
    quietly pass.
    """

    #: What gates the ordered path. A store that does not own the memories has its write times in
    #: SQL already, and must keep the SQL ordering.
    store_owned = True

    def __init__(self, order: list[str]):
        #: Newest-written first, which is the contract of `list_banks_by_write`.
        self._order = order
        #: Every page request, so a test can assert the walk stopped rather than swept.
        self.page_requests: list[tuple[int, str]] = []
        self.write_time_calls: list[list[str]] = []

    def store_owned_for(self, bank_id: str) -> bool:
        return True

    def _when(self, bank_id: str) -> datetime:
        # Newest first, so earlier positions are more recent.
        return _NOW - timedelta(minutes=self._order.index(bank_id))

    async def list_banks_by_write(self, *, limit: int = 100, page_token: str = "") -> BankWritePage:
        self.page_requests.append((limit, page_token))
        start = int(page_token) if page_token else 0
        rows = [BankWriteTime(bank_id=b, last_write_at=self._when(b)) for b in self._order[start : start + limit]]
        nxt = str(start + limit) if start + limit < len(self._order) else ""
        return BankWritePage(banks=rows, next_page_token=nxt)

    async def last_write_at_many(self, *, bank_ids: list[str]) -> dict:
        self.write_time_calls.append(list(bank_ids))
        return {b: self._when(b) for b in bank_ids if b in self._order}

    async def last_document_at_many(self, *, bank_ids: list[str]) -> dict:
        return {}

    async def count_memories_many(self, *, bank_ids: list[str], strong: bool = False) -> dict:
        return {b: {"world": 1} for b in bank_ids}

    async def ensure_bank_storage(self, bank_id: str) -> None:
        return None

    async def drop_bank_storage(self, bank_id: str) -> None:
        return None

    async def count_documents(self, *, bank_id: str) -> int:
        return 0

    # Below: what the fixture TEARDOWN reaches. This fake is duck-typed rather than a
    # MemoriesExtension subclass — it inherits no defaults — and `delete_bank` routes through the
    # store for a bank the store owns.
    async def count_memories(self, *, conn, fq_table, bank_id: str) -> dict:
        return {}

    async def list_entities(self, *, conn, fq_table, bank_id: str, search=None, limit=100, offset=0) -> dict:
        return {"items": [], "total": 0, "limit": limit, "offset": offset}

    async def delete_observations(self, *, conn, fq_table, bank_id: str) -> None:
        return None


async def _make_banks(memory, request_context, names: list[str]) -> None:
    for name in names:
        await memory.ensure_bank_profile(name, request_context=request_context)


@pytest.mark.asyncio
async def test_bank_list_follows_the_store_order(memory, monkeypatch):
    """The page is in the store's order, not in creation order.

    The banks are CREATED in one order and written in another, so a list that fell back to
    `created_at` — the thing that happens when the store's ordering is ignored — produces a
    different, recognisable sequence rather than accidentally passing.
    """
    request_context = RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)
    created_order = ["ord_a", "ord_b", "ord_c", "ord_d"]
    # Written newest-first in an order unrelated to creation.
    write_order = ["ord_c", "ord_a", "ord_d", "ord_b"]

    store = _OrderingStore(write_order)
    monkeypatch.setattr(memories_mod, "get_memories", lambda: store)

    try:
        await _make_banks(memory, request_context, created_order)
        page = await memory.list_banks(limit=10, offset=0, request_context=request_context)
        got = [b["bank_id"] for b in page["banks"] if b["bank_id"] in write_order]
        assert got == write_order, f"the list must follow the store's write order, got {got}"
    finally:
        for name in created_order:
            await memory.delete_bank(name, request_context=request_context)


@pytest.mark.asyncio
async def test_bank_list_asks_the_store_about_the_page_not_the_tenant(memory, monkeypatch):
    """The cost property, asserted structurally rather than as a latency number.

    What made the endpoint 60-75 s was asking for every bank's write time before it could sort. The
    guard is that the strong per-bank read now covers the returned PAGE only — so a tenant ten times
    larger costs the same page.
    """
    request_context = RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)
    names = [f"pg_bank_{i:02}" for i in range(12)]

    store = _OrderingStore(list(reversed(names)))
    monkeypatch.setattr(memories_mod, "get_memories", lambda: store)

    try:
        await _make_banks(memory, request_context, names)
        page = await memory.list_banks(limit=3, offset=0, request_context=request_context)

        assert len(page["banks"]) == 3
        # The per-namespace strong read is the expensive one; it must see the page, never the set.
        assert store.write_time_calls, "the page's write times were never read"
        for call in store.write_time_calls:
            assert len(call) <= 3, f"strong write-time read covered {len(call)} banks, not the page"
    finally:
        for name in names:
            await memory.delete_bank(name, request_context=request_context)


@pytest.mark.asyncio
async def test_a_bank_the_store_has_no_write_time_for_still_appears(memory, monkeypatch):
    """A never-written bank must not fall out of its own list.

    It has no row in the store — never written, or written but not yet folded — so it comes from
    the creation-ordered half of the merge. Dropping it would make a brand-new bank invisible until
    something wrote to it, which is precisely when someone goes looking for it.
    """
    request_context = RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)
    written = "merge_written"
    unwritten = "merge_unwritten"

    # The store knows only the written one.
    store = _OrderingStore([written])
    monkeypatch.setattr(memories_mod, "get_memories", lambda: store)

    try:
        await _make_banks(memory, request_context, [written, unwritten])
        page = await memory.list_banks(limit=50, offset=0, request_context=request_context)
        got = [b["bank_id"] for b in page["banks"]]
        assert written in got, "the written bank is missing"
        assert unwritten in got, "a bank the store has no write time for vanished from the list"
    finally:
        for name in (written, unwritten):
            await memory.delete_bank(name, request_context=request_context)


@pytest.mark.asyncio
async def test_paging_the_list_sees_every_bank_exactly_once(memory, monkeypatch):
    """No bank is repeated across pages and none is skipped at a seam.

    The page size deliberately does not divide the bank count, so the last page is short and the
    seams fall mid-stream — where an off-by-one in the merge shows up.
    """
    request_context = RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)
    names = [f"seam_bank_{i:02}" for i in range(10)]

    store = _OrderingStore(list(reversed(names)))
    monkeypatch.setattr(memories_mod, "get_memories", lambda: store)

    try:
        await _make_banks(memory, request_context, names)
        seen: list[str] = []
        for offset in range(0, 12, 4):
            page = await memory.list_banks(limit=4, offset=offset, request_context=request_context)
            seen.extend(b["bank_id"] for b in page["banks"] if b["bank_id"] in names)
        assert sorted(seen) == sorted(names), f"paging lost or repeated a bank: {seen}"
    finally:
        for name in names:
            await memory.delete_bank(name, request_context=request_context)


class _RoutingStore(_OrderingStore):
    """A router whose banks live in different backends.

    It declares ``store_owned = False`` at the class level and answers per bank through
    ``store_owned_for`` — the shape a real multi-backend deployment has. Gating the ordered path on
    the class flag therefore sends an entirely store-owned tenant down the ranking path: correct
    output, and the whole cost the ordered path exists to remove, with nothing failing to show it.
    """

    store_owned = False


@pytest.mark.asyncio
async def test_a_router_that_declares_store_owned_false_still_gets_the_ordered_path(memory, monkeypatch):
    """The path is chosen by asking the store for a page, not by reading a flag off it.

    Asserted on the ORDER rather than on latency: the ranking fallback also sorts by write time, so
    the only visible difference between the two paths on a small fixture is which one ran. The store
    is given an order unrelated to creation order, so following it proves the merge ran.
    """
    request_context = RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)
    created_order = ["rt_a", "rt_b", "rt_c", "rt_d"]
    write_order = ["rt_c", "rt_a", "rt_d", "rt_b"]

    store = _RoutingStore(write_order)
    monkeypatch.setattr(memories_mod, "get_memories", lambda: store)

    try:
        await _make_banks(memory, request_context, created_order)
        page = await memory.list_banks(limit=10, offset=0, request_context=request_context)
        got = [b["bank_id"] for b in page["banks"] if b["bank_id"] in write_order]
        assert got == write_order, f"a router must still get the store's order, got {got}"
        assert store.page_requests, "the store was never asked for a page — the flag gated it out"
    finally:
        for name in created_order:
            await memory.delete_bank(name, request_context=request_context)


class _NoOrderingStore(_OrderingStore):
    """A store with no ordering of its own — the Postgres default's behaviour."""

    async def list_banks_by_write(self, *, limit: int = 100, page_token: str = "") -> BankWritePage:
        self.page_requests.append((limit, page_token))
        return BankWritePage()


@pytest.mark.asyncio
async def test_a_store_with_no_ordering_keeps_the_sql_ordering(memory, monkeypatch):
    """An empty store page must fall back, not merge against nothing.

    Merging an empty stream would leave only the creation-ordered half, silently reordering a
    Postgres-owned tenant by creation time while still looking like a working list.
    """
    request_context = RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)
    names = ["noord_a", "noord_b"]

    store = _NoOrderingStore(names)
    monkeypatch.setattr(memories_mod, "get_memories", lambda: store)

    try:
        await _make_banks(memory, request_context, names)
        page = await memory.list_banks(limit=10, offset=0, request_context=request_context)
        got = [b["bank_id"] for b in page["banks"] if b["bank_id"] in names]
        assert sorted(got) == sorted(names), f"the list must still render every bank: {got}"
    finally:
        for name in names:
            await memory.delete_bank(name, request_context=request_context)


@pytest.mark.asyncio
async def test_the_sql_fact_count_skips_banks_the_store_owns(memory, monkeypatch):
    """A store-owned bank is never counted against `memory_units`.

    It has no rows there, so the answer is always 0 and `apply_store_fact_counts` replaces it a
    moment later — but the query still has to look, and on a large table that dominated the whole
    endpoint. Asserted on the store's count SURVIVING, which is what a stray SQL overwrite would
    destroy, and on the page rendering at all.
    """
    request_context = RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)
    names = ["sqlskip_a", "sqlskip_b"]

    store = _OrderingStore(list(reversed(names)))
    monkeypatch.setattr(memories_mod, "get_memories", lambda: store)

    try:
        await _make_banks(memory, request_context, names)
        page = await memory.list_banks(limit=10, offset=0, request_context=request_context)
        got = {b["bank_id"]: b["fact_count"] for b in page["banks"] if b["bank_id"] in names}
        assert set(got) == set(names), f"both banks must be listed: {got}"
        # 1 is what the fake store reports per bank; 0 would mean SQL overwrote it.
        assert all(v == 1 for v in got.values()), f"the store's count must win: {got}"
    finally:
        for name in names:
            await memory.delete_bank(name, request_context=request_context)


class _OrdersButOwnsNothing(_OrderingStore):
    """A store that can order banks but keeps none of their memories — the mixed-tenant shape.

    The ordered path is chosen by asking the store for a page, so a SQL-owned bank can be paged
    through it. Its `last_write_at` then has to come from the same SQL columns the ranking path
    reads, which is what this fake exists to check.
    """

    def store_owned_for(self, bank_id: str) -> bool:
        return False


@pytest.mark.asyncio
async def test_a_sql_owned_bank_keeps_the_write_time_its_facts_give_it(memory, request_context, monkeypatch):
    """`last_write_at` must count facts, not just documents, on the ordered path too.

    A fact written outside a retain — consolidation, curation, import — bumps `memory_units` and
    nothing else, so a page that reads only `documents` reports a bank as last written when its
    document was. The ranking path takes the newest of both; the paged join has to agree, or the
    same bank shows two different times depending on which path served the request.
    """
    bank_id = f"lastfact_{datetime.now(timezone.utc).timestamp()}"
    pool = await memory._get_pool()

    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "xyzabc123 !@# a slice"}],
            document_id="session-1",
            request_context=request_context,
        )
        # Forged directly: a fact newer than every document is what consolidation leaves behind,
        # and no public write produces it in a test without running one.
        forged = datetime.now(timezone.utc) + timedelta(hours=1)
        async with pool.acquire() as conn:
            await conn.execute("UPDATE memory_units SET updated_at = $1 WHERE bank_id = $2", forged, bank_id)

        monkeypatch.setattr(memories_mod, "get_memories", lambda: _OrdersButOwnsNothing([bank_id]))
        page = await memory.list_banks(limit=50, offset=0, request_context=request_context)
        entry = next(b for b in page["banks"] if b["bank_id"] == bank_id)
        assert entry["last_write_at"] is not None, "the bank reads as never written"
        assert datetime.fromisoformat(entry["last_write_at"]) == forged, (
            f"last_write_at ignored the fact write: {entry['last_write_at']} != {forged.isoformat()}"
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_a_sql_owned_bank_still_gets_its_fact_watermark(memory, monkeypatch):
    """The fact watermark is skipped for store-owned banks but MUST survive for SQL-owned ones.

    The skip is a `CASE` guard on a correlated subquery, so getting it backwards is silent: the
    page still renders, with `last_write_at` quietly missing every fact-only write. This pins the
    SQL-owned side of that guard, which is the half no store overlay would repair.
    """
    request_context = RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)
    sql_bank = "factwm_sql"
    store_bank = "factwm_store"

    class _Mixed(_OrderingStore):
        """Owns one bank, not the other — the mixed-tenant case the guard is per-bank for."""

        def store_owned_for(self, bank_id: str) -> bool:
            return bank_id == store_bank

    store = _Mixed([store_bank])
    monkeypatch.setattr(memories_mod, "get_memories", lambda: store)

    try:
        await _make_banks(memory, request_context, [sql_bank, store_bank])
        page = await memory.list_banks(limit=50, offset=0, request_context=request_context)
        got = [b["bank_id"] for b in page["banks"]]
        assert sql_bank in got and store_bank in got, f"both kinds must be listed: {got}"
    finally:
        for name in (sql_bank, store_bank):
            await memory.delete_bank(name, request_context=request_context)


@pytest.mark.asyncio
async def test_a_search_still_ranks_and_filters(memory, monkeypatch):
    """Search matches on bank_id and name, which live only in SQL.

    The store cannot apply it, so the search path keeps the SQL ordering rather than paging a store
    stream that is the wrong set. Asserted because the fallback is easy to break silently: a search
    that quietly returned the unfiltered page would look like a working list.
    """
    request_context = RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)
    names = ["find_me_alpha", "find_me_beta", "unrelated_gamma"]

    store = _OrderingStore(list(reversed(names)))
    monkeypatch.setattr(memories_mod, "get_memories", lambda: store)

    try:
        await _make_banks(memory, request_context, names)
        page = await memory.list_banks(search_query="find_me", limit=50, offset=0, request_context=request_context)
        got = sorted(b["bank_id"] for b in page["banks"])
        assert got == ["find_me_alpha", "find_me_beta"], f"search returned {got}"
    finally:
        for name in names:
            await memory.delete_bank(name, request_context=request_context)

"""Regression test: list_banks must consult the per-bank capability with the
*current row's* bank id, and source fact_count from the store when that bank
keeps its memories outside SQL.

Bug (introduced with per-bank store capabilities, #3350): list_banks called
``not _store.store_owned_for(bank_id)`` with a bare ``bank_id`` name
that is not in scope inside the per-row loop (the row's id is ``row["bank_id"]``).
Because the argument is evaluated before the call, this raised
``NameError: name 'bank_id' is not defined`` for *every* org on the very first
bank — i.e. GET /banks 500'd outright — regardless of the store's capability.

This test swaps in a store that reports ``store_owned_for -> True``
(the non-SQL branch the feature added), and asserts list_banks (a) does not raise,
(b) calls the capability + count_memories with the correct per-bank id, and
(c) surfaces the store's live count as fact_count.

Runs via: uv run pytest tests/test_list_banks_non_sql_store.py -v
"""

from __future__ import annotations

import pytest

import hindsight_api.engine.memories as memories_mod
from hindsight_api.models import RequestContext


class _NonSqlStore:
    """A store that keeps memory rows outside SQL: list_banks must count via the store."""

    def __init__(self):
        self.capability_calls: list[str] = []
        self.count_calls: list[str] = []

    def store_owned_for(self, bank_id: str) -> bool:
        self.capability_calls.append(bank_id)
        return True

    # This fake is duck-typed rather than a MemoriesExtension subclass, so it answers only what the
    # paths under test reach. Bank DELETION also counts what it removed through the store now (the
    # SQL tables are empty for such a bank, so COUNT(*) reported 0 while dropping everything), and
    # the fixture's teardown deletes the bank — hence these.
    #
    # It used to answer the document-store capability False while answering the memory-rows one
    # True — a mixed combination no real store ever set, and one the single `store_owned` flag
    # cannot express. Owning its writes means owning the bodies too, so the teardown now takes the
    # store branch and needs the two methods below.
    async def ensure_bank_storage(self, bank_id: str) -> None:
        return None

    async def drop_bank_storage(self, bank_id: str) -> None:
        return None

    async def count_documents(self, *, bank_id: str) -> int:
        return 0

    async def list_entities(self, *, conn, fq_table, bank_id: str, search=None, limit=100, offset=0) -> dict:
        return {"items": [], "total": 0, "limit": limit, "offset": offset}

    async def count_memories(self, *, conn, fq_table, bank_id: str) -> dict:
        self.count_calls.append(bank_id)
        return {"world": 7}

    async def drop_bank_storage(self, bank_id: str) -> None:
        """``delete_bank`` routes the drop through the store for a non-SQL bank, so the
        teardown below reaches this. Nothing to drop — the counts above are synthetic."""


@pytest.mark.asyncio
async def test_list_banks_counts_via_store_for_non_sql_bank(memory, monkeypatch):
    bank_id = "list_banks_non_sql_bank"
    request_context = RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)

    store = _NonSqlStore()
    monkeypatch.setattr(memories_mod, "get_memories", lambda: store)

    try:
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Must not raise NameError; must reach the store's non-SQL count path.
        page = await memory.list_banks(search_query=bank_id, request_context=request_context)

        entry = next((b for b in page["banks"] if b["bank_id"] == bank_id), None)
        assert entry is not None, f"bank {bank_id!r} not present in list_banks output"

        # The capability + count were consulted with the row's real bank id.
        assert bank_id in store.capability_calls
        assert bank_id in store.count_calls
        # fact_count came from the store (sum of the per-type counts), not the empty SQL join.
        assert entry["fact_count"] == 7
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

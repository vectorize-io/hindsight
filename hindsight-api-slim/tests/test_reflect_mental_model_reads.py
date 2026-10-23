"""Mental models reach reflect as snippets, and are read in full only when asked.

`search_mental_models` used to return five whole pages with no token budget of any
kind — 8.7-19k tokens measured on the refresh-cost eval, re-sent on every later
turn of the loop, the largest single item in a reflect's floor (#4533). It now
returns what the knowledge-page search returns, a snippet, and `read_mental_models`
fetches the pages the model actually wants, under a budget.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.reflect.tools import tool_read_mental_models


def _row(page_id: str, name: str, content: str) -> dict:
    return {"id": page_id, "name": name, "content": content, "tags": [], "last_refreshed_at": None}


def _conn(rows: list[dict]) -> MagicMock:
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)
    return conn


@pytest.mark.asyncio
async def test_pages_come_back_in_full_in_the_order_asked_for():
    conn = _conn([_row("mm-b", "Second", "second page"), _row("mm-a", "First", "first page")])

    out = await tool_read_mental_models(conn, "bank", ["mm-a", "mm-b"])

    assert [p["name"] for p in out["mental_models"]] == ["First", "Second"]
    assert [p["content"] for p in out["mental_models"]] == ["first page", "second page"]
    assert "not_read" not in out


@pytest.mark.asyncio
async def test_the_budget_stops_before_a_page_that_would_cross_it():
    """One enormous page must not swallow the reflect's context — the failure the
    unbounded search had. What did not fit is named, not silently missing."""
    conn = _conn([_row("mm-a", "Small", "tiny"), _row("mm-b", "Huge", "word " * 5000)])

    out = await tool_read_mental_models(conn, "bank", ["mm-a", "mm-b"], max_tokens=100)

    assert [p["name"] for p in out["mental_models"]] == ["Small"]
    assert out["not_read"] == ["mm-b"], "named by id, so the model can ask for it again on its own"


@pytest.mark.asyncio
async def test_the_first_page_is_read_even_when_it_alone_exceeds_the_budget():
    """Returning nothing would leave the model with a snippet it cannot act on."""
    conn = _conn([_row("mm-a", "Huge", "word " * 5000)])

    out = await tool_read_mental_models(conn, "bank", ["mm-a"], max_tokens=100)

    assert [p["name"] for p in out["mental_models"]] == ["Huge"]


@pytest.mark.asyncio
async def test_an_id_the_bank_does_not_have_is_reported_rather_than_dropped():
    out = await tool_read_mental_models(_conn([]), "bank", ["mm-gone"])

    assert out["mental_models"] == []
    assert out["not_read"] == ["mm-gone"]


@pytest.mark.asyncio
async def test_no_ids_is_a_usage_error():
    out = await tool_read_mental_models(_conn([]), "bank", [])

    assert "error" in out


@pytest.mark.asyncio
async def test_the_read_is_scoped_to_the_bank():
    """A mental model id is unique per bank, not globally: without the predicate a
    read would return another bank's page of the same id."""
    conn = _conn([])

    await tool_read_mental_models(conn, "bank-1", ["mm-a"])

    sql, *params = conn.fetch.await_args.args
    assert "bank_id = $1" in sql
    assert params[0] == "bank-1"


class TestSearchPacksWholePagesUnderABudget:
    """The compromise the measurements forced.

    Snippets alone cut a refresh 29% (gemini-2.5-flash-lite) and 55% (a local 35B),
    but the model sometimes answered from a snippet without ever reading the page —
    a graded failure in the system-evals story ``test_06_reflect_reads_pages``.
    Whole pages in rank order up to a token budget keep the best pages readable while
    bounding what five unbounded pages cost; one whole page under a fixed limit left
    banks with large pages, or answers spread across pages, with none (#5787).
    """

    @staticmethod
    async def _search(rows, **kwargs):
        from hindsight_api.engine.reflect.tools import tool_search_mental_models

        conn = MagicMock()
        conn.fetch = AsyncMock(return_value=rows)
        engine = MagicMock()
        engine.compute_mental_models_are_stale = AsyncMock(return_value={str(r["id"]): False for r in rows})
        store = MagicMock()
        store.store_owned_for = MagicMock(return_value=False)
        with patch("hindsight_api.engine.memories.get_memories", return_value=store):
            return await tool_search_mental_models(engine, conn, "bank", "q", [0.1], **kwargs)

    @staticmethod
    def _row(page_id: str, content: str, relevance: float):
        return {
            "id": page_id,
            "name": f"page {page_id}",
            "content": content,
            "tags": [],
            "relevance": relevance,
            "last_refreshed_at": None,
            "last_memory_seen_at": None,
            "trigger": {},
            "created_at": None,
        }

    @pytest.mark.asyncio
    async def test_pages_come_back_whole_in_rank_order_until_the_budget_and_the_rest_as_snippets(self):
        page = "word " * 100  # ~100 tokens
        rows = [self._row("mm-a", page, 0.9), self._row("mm-b", page, 0.8), self._row("mm-c", page, 0.7)]

        out = await self._search(rows, full_pages_max_tokens=250)

        first, second, third = out["mental_models"]
        assert first["content"] == page and second["content"] == page
        assert "snippet" not in first and "snippet" not in second
        assert third["snippet"] == page[:280].strip(), "past the budget, a snippet like the page-search API"
        assert third["content_chars"] == len(page), "so the model can see what it would be reading"
        assert "content" not in third

    @pytest.mark.asyncio
    async def test_the_best_hit_is_whole_even_alone_over_the_budget(self):
        """A snippet of it would leave the search with no page at all, the #5787 failure."""
        big = "word " * 5000
        rows = [self._row("mm-huge", big, 0.9), self._row("mm-small", "tiny", 0.5)]

        out = await self._search(rows, full_pages_max_tokens=100)

        top, small = out["mental_models"]
        assert top["content"] == big
        assert "content" not in small, "the budget is spent, so the rest are snippets"

    @pytest.mark.asyncio
    async def test_a_page_that_does_not_fit_does_not_stop_a_smaller_one_after_it(self):
        rows = [
            self._row("mm-a", "word " * 100, 0.9),
            self._row("mm-big", "word " * 1000, 0.8),
            self._row("mm-c", "small page", 0.7),
        ]

        out = await self._search(rows, full_pages_max_tokens=200)

        assert ["content" in m for m in out["mental_models"]] == [True, False, True]

"""Pagination contract for the mental-model and directive list endpoints.

Both used to return a bare ``items`` array, so a caller could not tell a full
page from the end of the collection and silently saw only the first 100 rows.
They now report ``total`` (every match, not just the page) alongside the
applied ``limit``/``offset``, like the documents/memories/tags endpoints.
"""

import uuid

import httpx
import pytest

from hindsight_api.engine.memory_engine import MemoryEngine

pytestmark = pytest.mark.asyncio


async def _make_mental_models(memory: MemoryEngine, bank_id: str, count: int, request_context) -> None:
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    for i in range(count):
        await memory.create_mental_model(
            bank_id=bank_id,
            name=f"Model {i}",
            source_query=f"Query {i}",
            content=f"Content {i}",
            tags=["even"] if i % 2 == 0 else ["odd"],
            request_context=request_context,
        )


async def _make_directives(memory: MemoryEngine, bank_id: str, count: int, request_context) -> None:
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    for i in range(count):
        await memory.create_directive(
            bank_id=bank_id,
            name=f"Directive {i}",
            content=f"Rule {i}",
            tags=["even"] if i % 2 == 0 else ["odd"],
            request_context=request_context,
        )


class TestMentalModelPagination:
    async def test_total_counts_every_match_not_the_page(self, memory: MemoryEngine, request_context):
        bank_id = f"test-mm-page-{uuid.uuid4().hex[:8]}"
        await _make_mental_models(memory, bank_id, 5, request_context)

        page = await memory.list_mental_models(bank_id=bank_id, limit=2, request_context=request_context)
        assert len(page.items) == 2
        assert page.total == 5

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_total_respects_the_tag_filter(self, memory: MemoryEngine, request_context):
        bank_id = f"test-mm-page-{uuid.uuid4().hex[:8]}"
        await _make_mental_models(memory, bank_id, 5, request_context)

        page = await memory.list_mental_models(bank_id=bank_id, tags=["odd"], limit=1, request_context=request_context)
        assert len(page.items) == 1
        assert page.total == 2

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_paging_covers_every_model_exactly_once(self, memory: MemoryEngine, request_context):
        """The models are created in one burst, so last_refreshed_at ties — without the
        id tie-break the pages would overlap and drop rows."""
        bank_id = f"test-mm-page-{uuid.uuid4().hex[:8]}"
        await _make_mental_models(memory, bank_id, 5, request_context)

        seen: list[str] = []
        offset = 0
        while True:
            page = await memory.list_mental_models(
                bank_id=bank_id, limit=2, offset=offset, request_context=request_context
            )
            seen.extend(m["id"] for m in page.items)
            offset += 2
            if offset >= page.total:
                break

        assert len(set(seen)) == 5
        assert len(seen) == 5

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_limit_none_returns_everything(self, memory: MemoryEngine, request_context):
        """The bank-template export/import path passes limit=None; it must see the
        whole set, not the first page."""
        bank_id = f"test-mm-page-{uuid.uuid4().hex[:8]}"
        await _make_mental_models(memory, bank_id, 5, request_context)

        page = await memory.list_mental_models(bank_id=bank_id, limit=None, request_context=request_context)
        assert len(page.items) == 5
        assert page.total == 5

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_http_response_reports_total_limit_offset(
        self, memory: MemoryEngine, api_client: httpx.AsyncClient, request_context
    ):
        bank_id = f"test-mm-page-{uuid.uuid4().hex[:8]}"
        await _make_mental_models(memory, bank_id, 5, request_context)

        resp = await api_client.get(
            f"/v1/default/banks/{bank_id}/mental-models", params={"limit": 2, "offset": 1, "detail": "metadata"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) == 2
        assert body["total"] == 5
        assert body["limit"] == 2
        assert body["offset"] == 1

        await memory.delete_bank(bank_id, request_context=request_context)


class TestDirectivePagination:
    async def test_total_counts_every_match_not_the_page(self, memory: MemoryEngine, request_context):
        bank_id = f"test-dir-page-{uuid.uuid4().hex[:8]}"
        await _make_directives(memory, bank_id, 5, request_context)

        page = await memory.list_directives(bank_id=bank_id, limit=2, request_context=request_context)
        assert len(page.items) == 2
        assert page.total == 5

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_total_respects_active_only(self, memory: MemoryEngine, request_context):
        bank_id = f"test-dir-page-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
        await memory.create_directive(
            bank_id=bank_id, name="Active", content="on", is_active=True, request_context=request_context
        )
        await memory.create_directive(
            bank_id=bank_id, name="Inactive", content="off", is_active=False, request_context=request_context
        )

        active = await memory.list_directives(bank_id=bank_id, active_only=True, request_context=request_context)
        assert active.total == 1
        every = await memory.list_directives(bank_id=bank_id, active_only=False, request_context=request_context)
        assert every.total == 2

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_limit_none_returns_everything(self, memory: MemoryEngine, request_context):
        bank_id = f"test-dir-page-{uuid.uuid4().hex[:8]}"
        await _make_directives(memory, bank_id, 5, request_context)

        page = await memory.list_directives(bank_id=bank_id, limit=None, request_context=request_context)
        assert len(page.items) == 5
        assert page.total == 5

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_http_response_reports_total_limit_offset(
        self, memory: MemoryEngine, api_client: httpx.AsyncClient, request_context
    ):
        bank_id = f"test-dir-page-{uuid.uuid4().hex[:8]}"
        await _make_directives(memory, bank_id, 5, request_context)

        resp = await api_client.get(f"/v1/default/banks/{bank_id}/directives", params={"limit": 2, "offset": 1})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) == 2
        assert body["total"] == 5
        assert body["limit"] == 2
        assert body["offset"] == 1

        await memory.delete_bank(bank_id, request_context=request_context)

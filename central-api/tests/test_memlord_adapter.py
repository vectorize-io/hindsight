"""MemlordAdapter unit tests — mock-safe (httpx.MockTransport, no real network)."""

import asyncio

import httpx

from app.adapters.memlord import MemlordAdapter

SEARCH_PAYLOAD = {
    "results": [
        {
            "id": 1,
            "content": "failover runbook",
            "memory_type": "agent",
            "workspace_id": 7,
            "workspace": "telecom",
            "score": 0.83,
        }
    ],
    "query": "failover",
}


# Two workspaces across two pages; page 2 is short → signals last page.
LIST_PAGES = {
    1: [
        {"id": 1, "content": "a", "memory_type": "fact", "workspace_id": 7, "created_at": "t"},
        {"id": 2, "content": "b", "memory_type": "fact", "workspace_id": 9, "created_at": "t"},
    ],
    2: [
        {"id": 3, "content": "c", "memory_type": "fact", "workspace_id": 7, "created_at": "t"},
    ],
}


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api-dev/memories/search":
        assert request.headers.get("x-api-key") == "mlk_test"
        return httpx.Response(200, json=SEARCH_PAYLOAD)
    if path == "/api-dev/memories":
        assert request.headers.get("x-api-key") == "mlk_test"
        import json

        body = json.loads(request.content)
        # page_size 100 in adapter, but our fixture pages are short → page 1 then 2 then empty.
        page = body.get("page", 1)
        return httpx.Response(200, json={"memories": LIST_PAGES.get(page, []), "page": page})
    if path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    return httpx.Response(404, json={"error": "not_found"})


def _adapter() -> MemlordAdapter:
    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler), base_url="http://memlord.test")
    return MemlordAdapter(api_key="mlk_test", client=client)


def test_search_maps_rrf_score_and_ids():
    hits = asyncio.run(_adapter().search("failover", k=5))
    assert len(hits) == 1
    hit = hits[0]
    assert hit.backend == "memlord"
    assert hit.memory_id == "memlord:1"
    assert hit.score == 0.83
    assert hit.metadata["rrf_score"] == 0.83
    assert hit.metadata["workspace_id"] == 7
    assert hit.citation["workspace"] == "telecom"


def test_unconfigured_adapter_makes_no_calls():
    adapter = MemlordAdapter()  # no api key, no client
    assert adapter.configured is False
    assert asyncio.run(adapter.search("anything")) == []


def test_health_ok_when_configured():
    health = asyncio.run(_adapter().health())
    assert health.backend == "memlord"
    assert health.status == "ok"


def test_health_stub_when_unconfigured():
    health = asyncio.run(MemlordAdapter().health())
    assert health.status == "stub"


def test_export_paginates_the_list_endpoint():
    # Export is built by paginating POST /api-dev/memories (the API-key list
    # surface), since the dedicated cookie-only export is unreachable by key.
    adapter = _adapter()
    adapter.export_page_size = 2  # force a real page-1 → page-2 → empty walk
    rows = asyncio.run(adapter.export(None))
    assert [r["id"] for r in rows] == [1, 2, 3]


def test_export_filters_by_workspace_client_side():
    adapter = _adapter()
    adapter.export_page_size = 2
    rows = asyncio.run(adapter.export("7"))  # list endpoint ignores ws → filter here
    assert [r["id"] for r in rows] == [1, 3]


def test_export_unconfigured_returns_empty():
    assert asyncio.run(MemlordAdapter().export()) == []


def test_adapter_is_read_only():
    adapter = MemlordAdapter()
    for coro in (adapter.store("x"), adapter.update("1", "x"), adapter.delete("1")):
        try:
            asyncio.run(coro)
            raise AssertionError("expected NotImplementedError")
        except NotImplementedError:
            pass

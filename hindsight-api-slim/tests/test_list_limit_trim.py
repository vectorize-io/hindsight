"""A list endpoint trims an oversized ``limit`` to its cap instead of rejecting it.

Production clients have asked for ``limit=100000``. Rejecting that would break them;
answering it in full is too much work for one request. So each list endpoint answers
with at most its cap and echoes the ``limit`` it actually used.
"""

import uuid

import pytest

from hindsight_api.pagination import MAX_GRAPH_LIMIT, MAX_PAGE_LIMIT

HUGE = 100_000
KB = "/v1/default/knowledge-banks"


@pytest.fixture
async def bank(api_client):
    bank_id = f"trim-{uuid.uuid4().hex[:8]}"
    await api_client.put(f"/v1/default/banks/{bank_id}", json={})
    yield bank_id
    await api_client.delete(f"/v1/default/banks/{bank_id}")


@pytest.fixture
async def knowledge_bank(api_client):
    bank_id = f"trim-kb-{uuid.uuid4().hex[:8]}"
    r = await api_client.post(KB, json={"bank_id": bank_id})
    assert r.status_code == 200, r.text
    yield bank_id
    await api_client.delete(f"{KB}/{bank_id}")


async def _used_limit(client, path: str) -> int:
    r = await client.get(path, params={"limit": HUGE})
    assert r.status_code == 200, f"{path}: {r.status_code} {r.text}"
    return r.json()["limit"]


async def test_memory_bank_lists_trim_an_oversized_limit(api_client, bank):
    base = f"/v1/default/banks/{bank}"
    expected = {
        "/v1/default/banks": MAX_PAGE_LIMIT,
        f"{base}/memories/list": MAX_PAGE_LIMIT,
        f"{base}/documents": MAX_PAGE_LIMIT,
        f"{base}/entities": MAX_PAGE_LIMIT,
        f"{base}/tags": MAX_PAGE_LIMIT,
        f"{base}/mental-models": MAX_PAGE_LIMIT,
        f"{base}/directives": MAX_PAGE_LIMIT,
        f"{base}/webhooks": MAX_PAGE_LIMIT,
        f"{base}/observations/scopes": MAX_PAGE_LIMIT,
        # Endpoints that already had a smaller cap keep it, and now trim to it too.
        f"{base}/operations": 100,
        f"{base}/audit-logs": 500,
        f"{base}/llm-requests": 500,
        # The graphs draw many edges at once (the control plane asks for 2000).
        f"{base}/graph": MAX_GRAPH_LIMIT,
        f"{base}/entities/graph": MAX_GRAPH_LIMIT,
    }
    used = {path: await _used_limit(api_client, path) for path in expected}
    assert used == expected


async def test_knowledge_bank_lists_trim_an_oversized_limit(api_client, knowledge_bank):
    expected = {
        KB: MAX_PAGE_LIMIT,
        f"{KB}/{knowledge_bank}/tags": MAX_PAGE_LIMIT,
    }
    used = {path: await _used_limit(api_client, path) for path in expected}
    assert used == expected
    # These two do not echo the limit; they answered 422 before trimming.
    for path in [f"{KB}/{knowledge_bank}/documents", f"{KB}/{knowledge_bank}/operations"]:
        r = await api_client.get(path, params={"limit": HUGE})
        assert r.status_code == 200, f"{path}: {r.status_code} {r.text}"


async def test_a_limit_under_the_cap_is_kept(api_client, bank):
    r = await api_client.get(f"/v1/default/banks/{bank}/memories/list", params={"limit": 7})
    assert r.status_code == 200, r.text
    assert r.json()["limit"] == 7


def test_every_endpoint_that_takes_a_limit_trims_it(memory):
    """Family guard: a new list endpoint written as a plain ``Query(ge=0)`` would be
    uncapped again. Every ``limit`` query param must carry a ``trim_to`` validator."""
    from fastapi.routing import APIRoute
    from pydantic.functional_validators import AfterValidator

    from hindsight_api.api import create_app

    app = create_app(memory, initialize_memory=False)
    routes = [
        (route.path, param)
        for route in app.routes
        if isinstance(route, APIRoute)
        for param in route.dependant.query_params
        if param.name == "limit"
    ]
    assert routes, "no route takes a limit; the guard is looking in the wrong place"
    untrimmed = [
        path for path, param in routes if not any(isinstance(m, AfterValidator) for m in param.field_info.metadata)
    ]
    assert untrimmed == []


async def test_webhook_deliveries_refuse_a_negative_limit(api_client, bank):
    # It was the one list endpoint with no lower bound: -1 reached the SQL LIMIT.
    r = await api_client.get(f"/v1/default/banks/{bank}/webhooks/any/deliveries", params={"limit": -1})
    assert r.status_code == 422, r.text

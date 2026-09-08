"""Regression (#4175): bank-scoped read endpoints must 404 for a bank that does not exist.

They used to answer 200 with an empty payload — zeroed counters from ``/stats``,
an empty page from ``/memories/list`` — which is byte-identical to a healthy,
freshly-created bank. Any monitor built on those endpoints therefore kept passing
after the bank it watched was renamed, deleted or recreated under another id, and
a typo in ``bank_id`` was never surfaced at all.

The check runs after each endpoint's own read authorization, so it widens nothing:
a caller that may not read the bank still gets its usual authorization error, not
a 404 that would leak the bank's existence.

The endpoint list is derived from the app's own routing table rather than written
out by hand, so a bank-scoped collection GET added later is covered the day it
lands instead of the day someone remembers to extend this file.
"""

import uuid

import httpx
import pytest
import pytest_asyncio
from fastapi.routing import APIRoute

from hindsight_api import RequestContext
from hindsight_api.api import create_app

_BANK_PREFIX = "/v1/default/banks/{bank_id}"

# Query params without which a route is a 422 before it ever reaches the handler.
_REQUIRED_PARAMS: dict[str, dict[str, str]] = {
    "/knowledge-base/search": {"q": "anything"},
}

# Bank-scoped GETs deliberately outside this contract.
_EXEMPT: dict[str, str] = {
    # Serves the bank's own stored files by storage key; a missing bank has no
    # keys, so the 404 already comes from the file lookup.
    "/files/{storage_key:path}": "404s on the missing file",
    # Withdrawn endpoints: 410 Gone for every bank, existing or not, pointing at
    # their replacements. There is nothing left to distinguish.
    "/document-transfer": "410 Gone regardless of the bank",
    "/profile": "410 Gone regardless of the bank",
}


def _bank_scoped_read_paths(app) -> list[str]:
    """Every GET under /banks/{bank_id} whose answer is a collection or aggregate.

    Sub-resource GETs (``/memories/{memory_id}``, ``/webhooks/{id}/deliveries``, …)
    are excluded: they already 404 on the missing child, so a missing bank was
    never mistaken for an empty one there. They are identified by carrying a
    second path parameter.
    """
    paths = []
    for route in app.routes:
        if not isinstance(route, APIRoute) or "GET" not in route.methods:
            continue
        if not route.path.startswith(_BANK_PREFIX):
            continue
        suffix = route.path[len(_BANK_PREFIX) :]
        if "{" in suffix or suffix in _EXEMPT:
            continue
        paths.append(suffix)
    return sorted(set(paths))


@pytest.fixture
def app(memory):
    return create_app(memory, initialize_memory=False)


@pytest_asyncio.fixture
async def api_client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def read_paths(app):
    return _bank_scoped_read_paths(app)


def test_the_route_scan_finds_the_endpoints_from_the_issue(read_paths):
    # Guards the scan itself: if a refactor moved these routes or changed the
    # prefix, every other test here would pass vacuously over an empty list.
    assert "/stats" in read_paths
    assert "/memories/list" in read_paths
    assert len(read_paths) >= 15


@pytest.mark.asyncio
async def test_missing_bank_returns_404(api_client, read_paths):
    bank_id = f"nosuch-{uuid.uuid4().hex[:8]}"
    for suffix in read_paths:
        resp = await api_client.get(
            f"{_BANK_PREFIX.format(bank_id=bank_id)}{suffix}", params=_REQUIRED_PARAMS.get(suffix)
        )
        assert resp.status_code == 404, f"{suffix}: {resp.status_code} {resp.text}"
        assert bank_id in resp.json()["detail"], suffix


@pytest.mark.asyncio
async def test_existing_empty_bank_still_returns_200(api_client, memory, read_paths):
    # Positive control: an empty bank that DOES exist must keep answering 200,
    # so the 404 above distinguishes "missing" from "empty" rather than replacing
    # one indistinguishable answer with another.
    bank_id = f"empty-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=RequestContext())
    for suffix in read_paths:
        resp = await api_client.get(
            f"{_BANK_PREFIX.format(bank_id=bank_id)}{suffix}", params=_REQUIRED_PARAMS.get(suffix)
        )
        assert resp.status_code == 200, f"{suffix}: {resp.status_code} {resp.text}"


@pytest.mark.asyncio
async def test_read_does_not_create_the_bank(api_client, memory):
    # The 404 must come from a read-only existence check: a client polling a
    # stale bank_id must not silently materialise that bank.
    bank_id = f"nosuch-{uuid.uuid4().hex[:8]}"
    assert (await api_client.get(f"{_BANK_PREFIX.format(bank_id=bank_id)}/stats")).status_code == 404
    profile = await memory.get_bank_profile(bank_id, request_context=RequestContext(), create_if_missing=False)
    assert profile is None

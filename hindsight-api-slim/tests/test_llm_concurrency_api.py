"""The process-wide LLM concurrency cap, read and changed at runtime.

Deterministic: no LLM call is made; the cap is a semaphore in this process.
"""

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.config import clear_config_cache
from hindsight_api.engine.llm_wrapper import get_global_llm_semaphore
from hindsight_api.extensions import AuthenticationError, Tenant, TenantContext, TenantExtension

URL = "/v1/default/llm-concurrency"


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture(autouse=True)
def _enable_api_and_restore_cap(monkeypatch):
    """The write API is off by default, so enable it here; the 'disabled' test turns it
    back off in its own body. The cap is process-global, so every test puts it back."""
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_LLM_CONCURRENCY_API", "true")
    clear_config_cache()
    sem = get_global_llm_semaphore()
    original = sem.capacity
    yield
    sem.set_capacity(original)
    clear_config_cache()


@pytest.mark.asyncio
async def test_get_reports_the_cap_and_live_usage(api_client):
    response = await api_client.get(URL)
    assert response.status_code == 200
    body = response.json()
    sem = get_global_llm_semaphore()
    assert set(body) == {"max_concurrent", "configured_max_concurrent", "in_flight", "waiting"}
    assert (body["max_concurrent"], body["in_flight"], body["waiting"]) == (sem.capacity, sem.in_flight, sem.waiting)


@pytest.mark.asyncio
async def test_patch_changes_the_live_cap_and_delete_restores_the_configured_one(api_client):
    configured = (await api_client.get(URL)).json()["configured_max_concurrent"]
    response = await api_client.patch(URL, json={"max_concurrent": configured + 3})
    assert response.status_code == 200
    assert response.json()["max_concurrent"] == configured + 3
    assert get_global_llm_semaphore().capacity == configured + 3

    reset = await api_client.delete(URL)
    assert reset.status_code == 200
    assert reset.json()["max_concurrent"] == configured
    assert get_global_llm_semaphore().capacity == configured


@pytest.mark.asyncio
async def test_writes_are_404_when_disabled_but_reads_still_work(api_client, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_LLM_CONCURRENCY_API", "false")
    clear_config_cache()
    before = get_global_llm_semaphore().capacity
    for response in (await api_client.patch(URL, json={"max_concurrent": before + 1}), await api_client.delete(URL)):
        assert response.status_code == 404
        assert "HINDSIGHT_API_ENABLE_LLM_CONCURRENCY_API" in response.json()["detail"]
    assert get_global_llm_semaphore().capacity == before
    assert (await api_client.get(URL)).status_code == 200


@pytest.mark.asyncio
async def test_a_cap_below_one_is_rejected(api_client):
    before = get_global_llm_semaphore().capacity
    assert (await api_client.patch(URL, json={"max_concurrent": 0})).status_code == 422
    assert get_global_llm_semaphore().capacity == before


class _RejectingTenant(TenantExtension):
    def __init__(self):
        super().__init__({})

    async def authenticate(self, context) -> TenantContext:
        raise AuthenticationError("no")

    async def list_tenants(self) -> list[Tenant]:
        return [Tenant(schema="public")]


@pytest.mark.asyncio
async def test_every_verb_goes_through_tenant_authentication(api_client, memory, monkeypatch):
    monkeypatch.setattr(memory, "_tenant_extension", _RejectingTenant())
    before = get_global_llm_semaphore().capacity
    for response in (
        await api_client.get(URL),
        await api_client.patch(URL, json={"max_concurrent": before + 1}),
        await api_client.delete(URL),
    ):
        assert response.status_code == 401
    assert get_global_llm_semaphore().capacity == before


@pytest.mark.asyncio
async def test_version_advertises_the_flag(api_client):
    assert (await api_client.get("/version")).json()["features"]["llm_concurrency_api"] is True

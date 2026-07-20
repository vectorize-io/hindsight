"""Tests for the server-level (instance) LLM configuration feature.

Covers the persistence store, the engine ``reconfigure_llm`` no-restart swap, the
config-resolver layering + credential redaction, and the HTTP endpoints. All
deterministic (mock / none providers only)."""

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.server_settings import ServerLlmConfig


@pytest_asyncio.fixture(autouse=True)
async def _isolate_server_settings(memory):
    """Keep the singleton server_settings row from leaking across tests.

    The pg0 database is session-scoped, so a persisted LLM config would otherwise
    reconfigure later tests' engines. Clear + reset the engine to the fixture's
    mock provider before each test, and clear again on teardown.
    """
    await memory._server_settings.clear_llm_config()
    memory._config_resolver.invalidate_server_settings(None)
    await memory.reconfigure_llm(provider="mock", model="mock", api_key="", base_url=None)
    yield
    await memory._server_settings.clear_llm_config()


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ---- Store -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_round_trip(memory):
    store = memory._server_settings
    assert await store.load_llm_config() is None

    await store.save_llm_config(provider="mock", model="m1", api_key="sk-test", base_url="http://x")
    loaded = await store.load_llm_config()
    assert loaded is not None
    assert (loaded.provider, loaded.model, loaded.api_key, loaded.base_url) == ("mock", "m1", "sk-test", "http://x")

    await store.clear_llm_config()
    assert await store.load_llm_config() is None


@pytest.mark.asyncio
async def test_store_rejects_invalid_provider(memory):
    with pytest.raises(ValueError, match="Invalid LLM provider"):
        await memory._server_settings.save_llm_config(provider="not-a-provider", api_key="x")


@pytest.mark.asyncio
async def test_store_encryption_round_trip(memory):
    from hindsight_api.server_settings import ServerSettingsStore

    store = ServerSettingsStore(memory._backend, enc_key="unit-test-secret")
    await store.save_llm_config(provider="mock", model="m", api_key="sk-secret", base_url=None)
    # Decrypts back for the app...
    assert (await store.load_llm_config()).api_key == "sk-secret"
    # ...but a store without the key cannot read the ciphertext and returns None.
    plain = ServerSettingsStore(memory._backend, enc_key=None)
    assert (await plain.load_llm_config()).api_key is None
    await store.clear_llm_config()


# ---- Engine reconfigure ----------------------------------------------------------


@pytest.mark.asyncio
async def test_reconfigure_rebuilds_all_configs(memory_no_llm_verify):
    mem = memory_no_llm_verify
    await mem.reconfigure_llm(provider="none", model=None, api_key="", base_url=None)
    assert mem._llm_config.provider == "none"
    assert mem._retain_llm_config.provider == "none"

    await mem.reconfigure_llm(provider="mock", model="mock", api_key="", base_url=None)
    assert mem._llm_config.provider == "mock"
    assert all(
        cfg.provider == "mock"
        for cfg in (mem._retain_llm_config, mem._reflect_llm_config, mem._consolidation_llm_config)
    )


# ---- Config-resolver layering + redaction ----------------------------------------


@pytest.mark.asyncio
async def test_resolver_applies_server_llm_overrides(memory):
    resolver = memory._config_resolver
    resolver.invalidate_server_settings(
        ServerLlmConfig(provider="openai", model="gpt-4.1-mini", api_key="sk-live", base_url=None)
    )
    resolved = await resolver.resolve_full_config("layering-bank")
    # The INTERNAL resolve path (used by operations) carries the base LLM + credential.
    assert resolved.llm_provider == "openai"
    assert resolved.llm_model == "gpt-4.1-mini"
    assert resolved.llm_api_key == "sk-live"


@pytest.mark.asyncio
async def test_bank_config_read_still_strips_llm_credentials(memory):
    resolver = memory._config_resolver
    resolver.invalidate_server_settings(
        ServerLlmConfig(provider="openai", model="gpt-4.1-mini", api_key="sk-live", base_url="http://x")
    )
    # The API-facing read must never expose provider/model/base_url/api_key.
    bank_config = await resolver.get_bank_config("redaction-bank")
    for leaked in ("llm_api_key", "llm_base_url", "llm_provider", "llm_model"):
        assert leaked not in bank_config


@pytest.mark.asyncio
async def test_bank_patch_still_rejects_llm_credentials(memory):
    resolver = memory._config_resolver
    with pytest.raises(ValueError, match="[Cc]redential"):
        await resolver.update_bank_config("reject-bank", {"llm_api_key": "sk-nope"})
    with pytest.raises(ValueError):
        await resolver.update_bank_config("reject-bank", {"llm_provider": "anthropic"})


# ---- HTTP endpoints --------------------------------------------------------------


@pytest.mark.asyncio
async def test_endpoint_put_get_delete(api_client, memory):
    # PUT persists and never echoes the key.
    put = await api_client.put(
        "/v1/default/server/llm-config",
        json={"provider": "mock", "model": "m-put", "api_key": "sk-abc", "base_url": None},
    )
    assert put.status_code == 200, put.text
    body = put.json()
    assert body["provider"] == "mock" and body["model"] == "m-put"
    assert body["api_key_is_set"] is True
    assert "api_key" not in body

    # GET reflects the stored config (still no key).
    got = (await api_client.get("/v1/default/server/llm-config")).json()
    assert got["provider"] == "mock" and got["api_key_is_set"] is True and "api_key" not in got

    # PUT without api_key preserves the stored one (write-only field).
    put2 = await api_client.put(
        "/v1/default/server/llm-config",
        json={"provider": "mock", "model": "m-put2"},
    )
    assert put2.status_code == 200
    assert put2.json()["model"] == "m-put2"
    assert put2.json()["api_key_is_set"] is True

    # DELETE clears the stored config.
    assert (await api_client.delete("/v1/default/server/llm-config")).status_code == 200
    assert await memory._server_settings.load_llm_config() is None


@pytest.mark.asyncio
async def test_endpoint_put_invalid_provider_returns_400(api_client):
    resp = await api_client.put(
        "/v1/default/server/llm-config",
        json={"provider": "totally-bogus", "api_key": "x"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_endpoint_health_probe(api_client):
    await api_client.put("/v1/default/server/llm-config", json={"provider": "mock", "model": "m"})
    resp = await api_client.post("/v1/default/server/llm-config/health")
    assert resp.status_code == 200
    ops = {o["operation"] for o in resp.json()["operations"]}
    assert ops == {"retain", "consolidation", "reflect"}
    # mock provider verifies successfully.
    assert all(o["ok"] for o in resp.json()["operations"])


@pytest.mark.asyncio
async def test_features_reports_server_llm_config(api_client):
    version = (await api_client.get("/version")).json()
    assert version["features"]["server_llm_config_api"] is True

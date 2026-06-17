"""Tests for the per-bank LLM connectivity probe (POST /health/llm).

Deterministic: the probe runs against the MockLLM provider (whose verify_connection
succeeds offline). No judge.
"""

import asyncio

import httpx
import pytest
import pytest_asyncio

import hindsight_api.engine.memory_engine as memory_engine
from hindsight_api.api import create_app
from hindsight_api.config import clear_config_cache


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture(autouse=True)
def _enable_bank_llm_health(monkeypatch):
    """The probe is off by default, so enable it for these tests. The 'disabled' test
    overrides this within its own body."""
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_BANK_LLM_HEALTH", "true")
    clear_config_cache()
    yield
    clear_config_cache()


# --------------------------------------------------------------------------- #
# POST /health/llm  (connectivity probe)
# --------------------------------------------------------------------------- #


def _statuses(body: dict) -> dict[str, str]:
    """Map operation -> status from a probe response."""
    return {op["operation"]: op["status"] for op in body["operations"]}


@pytest.mark.asyncio
async def test_bank_llm_connected_with_mock(api_client):
    response = await api_client.post("/v1/default/banks/llm-ok/health/llm")
    assert response.status_code == 200
    body = response.json()
    # All three operations share the mock config and should report connected.
    statuses = _statuses(body)
    assert statuses == {"retain": "connected", "consolidation": "connected", "reflect": "connected"}
    assert all(op["ok"] for op in body["operations"])
    assert all(op["latency_ms"] is not None for op in body["operations"])
    # Status only — no LLM identity must leak.
    assert all(set(op) == {"operation", "ok", "status", "latency_ms"} for op in body["operations"])
    assert "mock" not in response.text


@pytest.mark.asyncio
async def test_bank_llm_probes_shared_config_once(api_client, memory, monkeypatch):
    """retain/consolidation/reflect share one config in the mock fixture, so the probe
    must run exactly once and fan the result out to all three."""
    calls = 0

    async def counting_verify():
        nonlocal calls
        calls += 1

    for cfg in (memory._retain_llm_config, memory._consolidation_llm_config, memory._reflect_llm_config):
        monkeypatch.setattr(cfg, "verify_connection", counting_verify)

    body = (await api_client.post("/v1/default/banks/llm-dedup/health/llm")).json()
    assert len(body["operations"]) == 3
    assert calls == 1


@pytest.mark.asyncio
async def test_bank_llm_not_configured(api_client, memory, monkeypatch):
    for cfg in (memory._retain_llm_config, memory._consolidation_llm_config, memory._reflect_llm_config):
        monkeypatch.setattr(cfg, "provider", "none")
    body = (await api_client.post("/v1/default/banks/llm-none/health/llm")).json()
    assert all(op["status"] == "not_configured" and op["ok"] is False for op in body["operations"])
    # latency_ms is null when not configured; responses omit null fields, so use .get().
    assert all(op.get("latency_ms") is None for op in body["operations"])


@pytest.mark.asyncio
async def test_bank_llm_unreachable_does_not_leak_error(api_client, memory, monkeypatch):
    async def boom():
        raise RuntimeError("Connection refused to model gpt-4 at https://secret.internal/v1")

    for cfg in (memory._retain_llm_config, memory._consolidation_llm_config, memory._reflect_llm_config):
        monkeypatch.setattr(cfg, "verify_connection", boom)
    response = await api_client.post("/v1/default/banks/llm-bad/health/llm")
    body = response.json()
    assert all(op["status"] == "unreachable" and op["ok"] is False for op in body["operations"])
    # The raw provider error (which embeds endpoint/model) must NOT be returned.
    assert "secret.internal" not in response.text


@pytest.mark.asyncio
async def test_bank_llm_auth_failed(api_client, memory, monkeypatch):
    """A wrong API key (the most common failure) gets its own status, without leaking
    the raw provider error."""

    async def bad_key():
        raise RuntimeError("Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-secret'}}")

    for cfg in (memory._retain_llm_config, memory._consolidation_llm_config, memory._reflect_llm_config):
        monkeypatch.setattr(cfg, "verify_connection", bad_key)
    response = await api_client.post("/v1/default/banks/llm-badkey/health/llm")
    body = response.json()
    assert all(op["status"] == "auth_failed" and op["ok"] is False for op in body["operations"])
    assert "sk-secret" not in response.text


def test_is_auth_error_classifier():
    assert memory_engine._is_auth_error(RuntimeError("Error code: 401 Unauthorized")) is True
    assert memory_engine._is_auth_error(RuntimeError("Incorrect API key provided")) is True
    assert memory_engine._is_auth_error(RuntimeError("permission denied")) is True
    assert memory_engine._is_auth_error(RuntimeError("Connection refused")) is False
    assert memory_engine._is_auth_error(TimeoutError("slow")) is False

    class _StatusErr(Exception):
        status_code = 401

    assert memory_engine._is_auth_error(_StatusErr("nope")) is True


@pytest.mark.asyncio
async def test_bank_llm_timeout(api_client, memory, monkeypatch):
    monkeypatch.setattr(memory_engine, "_LLM_PROBE_TIMEOUT_SECONDS", 0.05)

    async def slow():
        await asyncio.sleep(0.5)

    for cfg in (memory._retain_llm_config, memory._consolidation_llm_config, memory._reflect_llm_config):
        monkeypatch.setattr(cfg, "verify_connection", slow)
    body = (await api_client.post("/v1/default/banks/llm-slow/health/llm")).json()
    assert all(op["status"] == "timeout" and op["ok"] is False for op in body["operations"])


@pytest.mark.asyncio
async def test_bank_llm_health_disabled_returns_404(api_client, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_BANK_LLM_HEALTH", "false")
    clear_config_cache()
    try:
        response = await api_client.post("/v1/default/banks/llm-off/health/llm")
        assert response.status_code == 404
    finally:
        monkeypatch.delenv("HINDSIGHT_API_ENABLE_BANK_LLM_HEALTH", raising=False)
        clear_config_cache()

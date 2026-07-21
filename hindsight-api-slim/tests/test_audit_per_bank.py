"""Tests for the per-bank ``audit_log_enabled`` override.

``audit_log_enabled`` resolves through env -> tenant -> bank, so a deployment
can audit some banks and not others. These tests cover both directions of the
override (a bank opting IN when the default is off, and a bank opting OUT when
the default is on), plus the fallbacks that decide behaviour when there is no
bank in scope or config resolution fails.
"""

import asyncio
from datetime import datetime

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.engine.audit import AuditLogger
from tests.conftest import enable_audit_default

# Audit writes are fire-and-forget; give the background task room to land.
_AUDIT_SETTLE_SECONDS = 1.0


@pytest_asyncio.fixture
async def client(memory):
    """HTTP client whose audit allowlist is open (all actions auditable)."""
    memory._audit_logger._allowed_actions = None
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _recall(client, bank_id: str) -> None:
    r = await client.post(
        f"/v1/default/banks/{bank_id}/memories/recall",
        json={"query": "per-bank audit test"},
    )
    assert r.status_code == 200, r.text
    await asyncio.sleep(_AUDIT_SETTLE_SECONDS)


async def _audited_actions(client, bank_id: str) -> list[str]:
    r = await client.get(f"/v1/default/banks/{bank_id}/audit-logs")
    assert r.status_code == 200, r.text
    return [e["action"] for e in r.json()["items"]]


def _bank(prefix: str) -> str:
    return f"{prefix}_{datetime.now().timestamp()}"


# ── the override, both directions ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_bank_opts_in_when_default_off(client, memory):
    """A bank override of True audits even though the deployment default is off."""
    enable_audit_default(memory, False)
    memory._audit_logger._enabled = False
    bank_id = _bank("audit_optin")

    await client.put(f"/v1/default/banks/{bank_id}", json={})
    await memory._config_resolver.update_bank_config(bank_id, {"audit_log_enabled": True})

    await _recall(client, bank_id)
    assert "recall" in await _audited_actions(client, bank_id)


@pytest.mark.asyncio
async def test_bank_opts_out_when_default_on(client, memory):
    """A bank override of False suppresses auditing that is otherwise on."""
    enable_audit_default(memory, True)
    memory._audit_logger._enabled = True
    bank_id = _bank("audit_optout")

    await client.put(f"/v1/default/banks/{bank_id}", json={})
    await memory._config_resolver.update_bank_config(bank_id, {"audit_log_enabled": False})

    await _recall(client, bank_id)
    # The create_bank above was audited (it ran before the override existed),
    # so assert on the action the override was meant to suppress.
    assert "recall" not in await _audited_actions(client, bank_id)


@pytest.mark.asyncio
async def test_banks_are_independent(client, memory):
    """One bank's override does not leak into another bank."""
    enable_audit_default(memory, False)
    memory._audit_logger._enabled = False
    audited, quiet = _bank("audit_on"), _bank("audit_off")

    for b in (audited, quiet):
        await client.put(f"/v1/default/banks/{b}", json={})
    await memory._config_resolver.update_bank_config(audited, {"audit_log_enabled": True})

    await _recall(client, audited)
    await _recall(client, quiet)

    assert "recall" in await _audited_actions(client, audited)
    assert await _audited_actions(client, quiet) == []


@pytest.mark.asyncio
async def test_no_override_uses_deployment_default(client, memory):
    """A bank with no explicit override follows the deployment default."""
    enable_audit_default(memory, True)
    memory._audit_logger._enabled = True
    bank_id = _bank("audit_inherit")

    await client.put(f"/v1/default/banks/{bank_id}", json={})
    await _recall(client, bank_id)
    assert "recall" in await _audited_actions(client, bank_id)


# ── AuditLogger decision logic (no DB) ─────────────────────────────────────


def _logger(*, enabled: bool, resolver=None, actions: list[str] | None = None) -> AuditLogger:
    return AuditLogger(
        pool_getter=lambda: None,
        schema_getter=lambda: "public",
        enabled=enabled,
        allowed_actions=actions or [],
        bank_enabled_resolver=resolver,
    )


@pytest.mark.asyncio
async def test_allowlist_short_circuits_before_resolution():
    """A disallowed action never triggers a per-bank lookup."""
    calls: list[str] = []

    async def resolver(bank_id, context=None):
        calls.append(bank_id)
        return True

    log = _logger(enabled=True, resolver=resolver, actions=["recall"])
    assert await log.should_log("reflect", "b1") is False
    assert calls == [], "allowlist must short-circuit before resolving bank config"


@pytest.mark.asyncio
async def test_no_bank_in_scope_uses_global_default():
    """With no bank_id there is nothing to resolve, so the global value decides."""

    async def resolver(bank_id, context=None):
        raise AssertionError("must not resolve without a bank_id")

    assert await _logger(enabled=True, resolver=resolver).should_log("recall", None) is True
    assert await _logger(enabled=False, resolver=resolver).should_log("recall", None) is False


@pytest.mark.asyncio
async def test_resolution_failure_falls_back_to_global_default():
    """A resolver error must not drop audit rows for an audited deployment.

    Fails OPEN (to the deployment default) rather than closed: a transient DB
    blip should not silently create an audit gap for a bank meant to be audited.
    """

    async def broken(bank_id, context=None):
        raise RuntimeError("config backend down")

    assert await _logger(enabled=True, resolver=broken).should_log("recall", "b1") is True
    assert await _logger(enabled=False, resolver=broken).should_log("recall", "b1") is False


@pytest.mark.asyncio
async def test_unwired_resolver_uses_global_default():
    """Without a resolver the logger behaves exactly as before this feature."""
    assert await _logger(enabled=True).should_log("recall", "b1") is True
    assert await _logger(enabled=False).should_log("recall", "b1") is False


@pytest.mark.asyncio
async def test_gating_ignores_config_permission_filter(memory, request_context):
    """A tenant permission filter must not suppress a bank's audit override.

    get_allowed_config_fields controls which fields a user may *modify* (and it
    filters the API-facing config read). Audit gating is an internal decision
    and must see the bank's true stored value: a deployment that makes
    audit_log_enabled read-only for some users must still audit banks that
    opted in. Regression guard for the resolve_full_config vs get_bank_config
    distinction in MemoryEngine._resolve_bank_audit_enabled.
    """
    from hindsight_api.config_resolver import ConfigResolver
    from hindsight_api.extensions.tenant import Tenant, TenantContext, TenantExtension

    bank_id = "audit-perm-filter"

    class RestrictiveExtension(TenantExtension):
        def __init__(self):
            pass  # skip Extension.__init__(config); nothing here needs config

        async def authenticate(self, context):
            return TenantContext(schema_name="public")

        async def list_tenants(self):
            return [Tenant(schema="public")]

        async def get_allowed_config_fields(self, context, bank_id):
            # audit_log_enabled deliberately absent: read-only for this user.
            return {"retain_chunk_size"}

    await memory.get_bank_profile(bank_id, request_context=request_context)

    # Store the opt-in with an allow-all resolver (the write is a separate
    # concern from gating), then swap in the restrictive extension.
    await memory._config_resolver.update_bank_config(bank_id, {"audit_log_enabled": True}, request_context)
    restrictive = ConfigResolver(backend=memory._backend, tenant_extension=RestrictiveExtension())
    memory._config_resolver = restrictive

    # The API-facing read is filtered (proving the filter is actually active)...
    api_view = await restrictive.get_bank_config(bank_id, request_context)
    assert "audit_log_enabled" not in api_view

    # ...but gating still sees the real value and audits the bank.
    assert await memory._resolve_bank_audit_enabled(bank_id, request_context) is True

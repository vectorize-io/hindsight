"""Tests for Supabase organization authz extensions."""

from __future__ import annotations

import hashlib
from typing import Literal
from unittest.mock import AsyncMock

import pytest

from hindsight_api.extensions.authz_profile import validate_authz_profile
from hindsight_api.extensions.builtin.supabase_org import (
    CallerPolicy,
    SupabaseAuthorizationExtension,
    SupabaseOrgTenantExtension,
    SupabasePolicyResolver,
)
from hindsight_api.extensions.operation_validator import (
    BankListContext,
    BankWriteContext,
    RecallContext,
    ReflectContext,
    RetainContext,
)
from hindsight_api.extensions.tenant import AuthenticationError, Tenant
from hindsight_api.models import RequestContext


class _MigrationContext:
    def __init__(self) -> None:
        self.run_migration = AsyncMock()


def _resolver_config() -> dict[str, str]:
    return {
        "supabase_url": "https://test.supabase.co",
        "supabase_service_key": "service-key",
        "policy_cache_ttl_seconds": "0",
    }


def _jwt_context() -> RequestContext:
    return RequestContext(api_key="header.payload.signature", selected_org_id="org_123")


def _policy(
    *,
    role: Literal["owner", "admin", "member"] = "owner",
    allowed_operations: frozenset[str] | None = None,
    allowed_bank_ids: frozenset[str] | None = None,
) -> CallerPolicy:
    return CallerPolicy(
        org_id="org_123",
        schema_name="org_org_123",
        user_id="user_123",
        api_key_id=None,
        role=role,
        allowed_bank_ids=allowed_bank_ids,
        allowed_operations=allowed_operations,
        tenant_config={},
    )


def _api_key_policy() -> CallerPolicy:
    return CallerPolicy(
        org_id="org_123",
        schema_name="org_org_123",
        user_id=None,
        api_key_id="key_123",
        role="member",
        allowed_bank_ids=frozenset({"bank_a"}),
        allowed_operations=frozenset({"recall"}),
        tenant_config={},
    )


def _admin_api_key_policy() -> CallerPolicy:
    return CallerPolicy(
        org_id="org_123",
        schema_name="org_org_123",
        user_id=None,
        api_key_id="key_123",
        role="admin",
        allowed_bank_ids=frozenset({"bank_a"}),
        allowed_operations=frozenset({"retain", "recall", "reflect"}),
        tenant_config={},
    )


@pytest.mark.asyncio
async def test_resolver_maps_supabase_jwt_to_org_schema() -> None:
    resolver = SupabasePolicyResolver(_resolver_config())
    resolver._verify_token = AsyncMock(return_value="user_123")  # type: ignore[method-assign]
    resolver._rest_get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            [{"org_id": "org_123", "user_id": "user_123", "role": "admin"}],
            [{"id": "org_123", "name": "Org", "config": {"llm_model": "gpt-4"}}],
        ]
    )

    policy = await resolver.resolve(_jwt_context())

    assert policy.org_id == "org_123"
    assert policy.schema_name == "org_org_123"
    assert policy.user_id == "user_123"
    assert policy.role == "admin"
    assert policy.allowed_bank_ids is None
    assert policy.tenant_config == {"llm_model": "gpt-4"}


@pytest.mark.asyncio
async def test_resolver_rejects_jwt_without_selected_org() -> None:
    resolver = SupabasePolicyResolver(_resolver_config())

    with pytest.raises(AuthenticationError, match="Missing X-Hindsight-Org-Id"):
        await resolver.resolve(RequestContext(api_key="header.payload.signature"))


@pytest.mark.asyncio
async def test_resolver_rejects_non_member_jwt() -> None:
    resolver = SupabasePolicyResolver(_resolver_config())
    resolver._verify_token = AsyncMock(return_value="user_123")  # type: ignore[method-assign]
    resolver._rest_get = AsyncMock(return_value=[])  # type: ignore[method-assign]

    with pytest.raises(AuthenticationError, match="not a member"):
        await resolver.resolve(_jwt_context())


@pytest.mark.asyncio
async def test_resolver_maps_hindsight_api_key_to_bank_scoped_policy() -> None:
    resolver = SupabasePolicyResolver(_resolver_config())
    api_key = "hs_test_secret_with_enough_entropy"
    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    resolver._rest_get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            [
                {
                    "id": "key_123",
                    "org_id": "org_123",
                    "role": "member",
                    "allowed_operations": ["recall", "reflect"],
                }
            ],
            [{"bank_id": "bank_a"}],
            [{"id": "org_123", "name": "Org", "config": {}}],
        ]
    )

    policy = await resolver.resolve(RequestContext(api_key=api_key))

    assert policy.org_id == "org_123"
    assert policy.api_key_id == "key_123"
    assert policy.allowed_bank_ids == frozenset({"bank_a"})
    assert policy.allowed_operations == frozenset({"recall", "reflect"})
    resolver._rest_get.assert_any_call(  # type: ignore[attr-defined]
        "hindsight_api_keys",
        select="id,org_id,role,allowed_operations,revoked_at,expires_at",
        key_hash=f"eq.{key_hash}",
        revoked_at="is.null",
        limit="1",
    )


@pytest.mark.asyncio
async def test_resolver_rejects_expired_hindsight_api_key() -> None:
    resolver = SupabasePolicyResolver(_resolver_config())
    resolver._rest_get = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            {
                "id": "key_123",
                "org_id": "org_123",
                "role": "member",
                "allowed_operations": ["recall"],
                "expires_at": "2000-01-01T00:00:00Z",
            }
        ]
    )

    with pytest.raises(AuthenticationError, match="expired"):
        await resolver.resolve(RequestContext(api_key="hs_test_secret_with_enough_entropy"))


@pytest.mark.asyncio
async def test_resolver_preserves_empty_api_key_operation_scope() -> None:
    resolver = SupabasePolicyResolver(_resolver_config())
    resolver._rest_get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            [
                {
                    "id": "key_123",
                    "org_id": "org_123",
                    "role": "member",
                    "allowed_operations": [],
                    "expires_at": "2099-01-01T00:00:00Z",
                }
            ],
            [],
            [{"id": "org_123", "name": "Org", "config": {}}],
        ]
    )

    policy = await resolver.resolve(RequestContext(api_key="hs_test_secret_with_enough_entropy"))

    assert policy.allowed_operations == frozenset()


@pytest.mark.asyncio
async def test_tenant_extension_populates_request_context() -> None:
    extension = SupabaseOrgTenantExtension(_resolver_config())
    context_api = _MigrationContext()
    extension.set_context(context_api)  # type: ignore[arg-type]
    extension.resolver.resolve = AsyncMock(return_value=_policy())  # type: ignore[method-assign]

    context = _jwt_context()
    tenant = await extension.authenticate(context)

    assert tenant.schema_name == "org_org_123"
    assert context.tenant_id == "org_123"
    assert context.user_id == "user_123"
    assert context.role == "owner"
    assert context.auth_policy is not None
    context_api.run_migration.assert_awaited_once_with("org_org_123")


@pytest.mark.asyncio
async def test_tenant_extension_runs_migration_on_first_schema_access() -> None:
    extension = SupabaseOrgTenantExtension(_resolver_config())
    context_api = _MigrationContext()
    extension.set_context(context_api)  # type: ignore[arg-type]
    extension.resolver.resolve = AsyncMock(return_value=_policy())  # type: ignore[method-assign]

    await extension.authenticate(_jwt_context())
    await extension.authenticate(_jwt_context())

    context_api.run_migration.assert_awaited_once_with("org_org_123")


@pytest.mark.asyncio
async def test_list_tenants_marks_startup_migrated_schemas_ready() -> None:
    extension = SupabaseOrgTenantExtension(_resolver_config())
    context_api = _MigrationContext()
    extension.set_context(context_api)  # type: ignore[arg-type]
    extension.resolver.list_tenants = AsyncMock(return_value=[Tenant(schema="org_org_123", tenant_id="org_123")])  # type: ignore[method-assign]
    extension.resolver.resolve = AsyncMock(return_value=_policy())  # type: ignore[method-assign]

    await extension.list_tenants()
    await extension.authenticate(_jwt_context())

    context_api.run_migration.assert_not_awaited()


@pytest.mark.asyncio
async def test_authorization_allows_admin_bank_write() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(return_value=_policy())  # type: ignore[method-assign]

    result = await validator.validate_bank_write(
        BankWriteContext(bank_id="bank_a", operation="delete_bank", request_context=_jwt_context())
    )

    assert result.allowed is True


@pytest.mark.asyncio
async def test_authorization_allows_admin_api_key_retain_with_scope() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(return_value=_admin_api_key_policy())  # type: ignore[method-assign]

    result = await validator.validate_retain(
        RetainContext(bank_id="bank_a", contents=[], request_context=_jwt_context())
    )

    assert result.allowed is True


@pytest.mark.asyncio
async def test_authorization_reuses_policy_from_request_context() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(side_effect=AssertionError("resolver should not be called"))  # type: ignore[method-assign]
    context = _jwt_context()
    context.auth_policy = _admin_api_key_policy()

    result = await validator.validate_retain(RetainContext(bank_id="bank_a", contents=[], request_context=context))

    assert result.allowed is True
    validator.resolver.resolve.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_authorization_denies_member_write() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(return_value=_policy(role="member"))  # type: ignore[method-assign]

    result = await validator.validate_retain(
        RetainContext(bank_id="bank_a", contents=[], request_context=_jwt_context())
    )

    assert result.allowed is False
    assert result.status_code == 403


@pytest.mark.asyncio
async def test_authorization_enforces_api_key_bank_scope_and_operation_scope() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(return_value=_api_key_policy())  # type: ignore[method-assign]

    allowed = await validator.validate_recall(
        RecallContext(bank_id="bank_a", query="q", request_context=_jwt_context())
    )
    wrong_operation = await validator.validate_reflect(
        ReflectContext(bank_id="bank_a", query="q", request_context=_jwt_context())
    )
    wrong_bank = await validator.validate_recall(
        RecallContext(bank_id="bank_b", query="q", request_context=_jwt_context())
    )

    assert allowed.allowed is True
    assert wrong_operation.allowed is False
    assert wrong_bank.allowed is False


@pytest.mark.asyncio
async def test_filter_bank_list_limits_member_api_key_scope() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(return_value=_api_key_policy())  # type: ignore[method-assign]

    result = await validator.filter_bank_list(
        BankListContext(
            banks=[{"id": "bank_a"}, {"id": "bank_b"}],
            request_context=_jwt_context(),
        )
    )

    assert result.banks == [{"id": "bank_a"}]


def test_validate_authz_profile_requires_both_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HINDSIGHT_API_AUTHZ_PROFILE", "supabase_org")

    with pytest.raises(RuntimeError, match="requires"):
        validate_authz_profile(SupabaseOrgTenantExtension(_resolver_config()), None)

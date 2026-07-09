"""Tests for Supabase organization authz extensions."""

from __future__ import annotations

import hashlib
from typing import Literal
from unittest.mock import AsyncMock

import pytest

from hindsight_api.extensions.auth_profile import validate_auth_profile
from hindsight_api.extensions.builtin.supabase_org import (
    ALL_DATAPLANE_OPERATIONS,
    BANK_READ_OPERATIONS,
    BANK_WRITE_OPERATIONS,
    SPECIAL_BANK_OPERATIONS,
    UNSCOPED_DATAPLANE_OPERATIONS,
    CallerPolicy,
    SupabaseAuthorizationExtension,
    SupabaseOrgTenantExtension,
    SupabasePolicyResolver,
)
from hindsight_api.extensions.operation_validator import (
    BankCreateResult,
    BankDeleteResult,
    BankListContext,
    BankReadContext,
    BankReadOperation,
    BankWriteContext,
    BankWriteOperation,
    ConsolidateContext,
    CreateBankContext,
    MentalModelGetContext,
    MentalModelRefreshContext,
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
    return RequestContext(api_key="header.payload.signature", selected_tenant_id="org_123")


def _policy(
    *,
    role: Literal["owner", "admin", "member"] = "owner",
    allowed_operations: frozenset[str] | None = None,
    operation_bank_scope_modes: dict[str, Literal["all", "selected"]] | None = None,
    operation_bank_internal_ids: dict[str, frozenset[str]] | None = None,
    api_key_id: str | None = None,
) -> CallerPolicy:
    return CallerPolicy(
        org_id="org_123",
        schema_name="org_org_123",
        user_id=None if api_key_id else "user_123",
        api_key_id=api_key_id,
        role=role,
        allowed_operations=allowed_operations,
        operation_bank_scope_modes=operation_bank_scope_modes,
        operation_bank_internal_ids=operation_bank_internal_ids,
        tenant_config={},
    )


def _api_key_policy() -> CallerPolicy:
    return CallerPolicy(
        org_id="org_123",
        schema_name="org_org_123",
        user_id=None,
        api_key_id="key_123",
        role="member",
        allowed_operations=frozenset({"recall"}),
        operation_bank_scope_modes={"recall": "all"},
        operation_bank_internal_ids={},
        tenant_config={},
    )


def _admin_api_key_policy() -> CallerPolicy:
    return CallerPolicy(
        org_id="org_123",
        schema_name="org_org_123",
        user_id=None,
        api_key_id="key_123",
        role="admin",
        allowed_operations=frozenset({"retain", "recall", "reflect"}),
        operation_bank_scope_modes={"retain": "all", "recall": "all", "reflect": "all"},
        operation_bank_internal_ids={},
        tenant_config={},
    )


def test_special_hook_operations_are_separate_from_bank_read_write_operations() -> None:
    assert SPECIAL_BANK_OPERATIONS == frozenset(
        {
            "mental_model_get",
            "mental_model_refresh",
            "recall",
            "reflect",
            "retain",
        }
    )
    assert SPECIAL_BANK_OPERATIONS.isdisjoint(BANK_READ_OPERATIONS)
    assert SPECIAL_BANK_OPERATIONS.isdisjoint(BANK_WRITE_OPERATIONS)


def test_manifest_operations_match_hook_enums() -> None:
    api_unreachable_read_operations = frozenset({BankReadOperation.GET_ENTITY_STATE})
    api_unreachable_write_operations = frozenset(
        {
            BankWriteOperation.RUN_CONSOLIDATION,
            BankWriteOperation.SET_BANK_MISSION,
        }
    )
    api_unreachable_special_operations = frozenset({"consolidate"})

    assert BANK_READ_OPERATIONS == frozenset(
        operation.value for operation in BankReadOperation if operation not in api_unreachable_read_operations
    )
    assert BANK_WRITE_OPERATIONS == frozenset(
        operation.value for operation in BankWriteOperation if operation not in api_unreachable_write_operations
    )
    assert api_unreachable_special_operations.isdisjoint(ALL_DATAPLANE_OPERATIONS)
    assert "get_entity_state" not in ALL_DATAPLANE_OPERATIONS
    assert "run_consolidation" not in ALL_DATAPLANE_OPERATIONS
    assert "set_bank_mission" not in ALL_DATAPLANE_OPERATIONS
    assert UNSCOPED_DATAPLANE_OPERATIONS == frozenset({"create_bank"})
    assert len(ALL_DATAPLANE_OPERATIONS) == 67


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
    assert policy.operation_bank_scope_modes is None
    assert policy.tenant_config == {"llm_model": "gpt-4"}


@pytest.mark.asyncio
async def test_resolver_rejects_jwt_without_selected_org() -> None:
    resolver = SupabasePolicyResolver(_resolver_config())

    with pytest.raises(AuthenticationError, match="Missing X-Hindsight-Tenant-Id"):
        await resolver.resolve(RequestContext(api_key="header.payload.signature"))


@pytest.mark.asyncio
async def test_resolver_rejects_non_member_jwt() -> None:
    resolver = SupabasePolicyResolver(_resolver_config())
    resolver._verify_token = AsyncMock(return_value="user_123")  # type: ignore[method-assign]
    resolver._rest_get = AsyncMock(return_value=[])  # type: ignore[method-assign]

    with pytest.raises(AuthenticationError, match="not a member"):
        await resolver.resolve(_jwt_context())


@pytest.mark.asyncio
async def test_resolver_maps_hindsight_api_key_to_operation_scoped_policy() -> None:
    resolver = SupabasePolicyResolver(_resolver_config())
    api_key = "hs_test_secret_with_enough_entropy"
    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    resolver._rest_get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            [
                {
                    "id": "key_123",
                    "org_id": "org_123",
                    "created_by_user_id": "user_123",
                    "role": "member",
                    "allowed_operations": ["recall", "reflect"],
                }
            ],
            [{"org_id": "org_123", "user_id": "user_123", "role": "member"}],
            [
                {"operation": "recall", "bank_scope_mode": "all"},
                {"operation": "reflect", "bank_scope_mode": "all"},
            ],
            [{"id": "org_123", "name": "Org", "config": {}}],
        ]
    )

    policy = await resolver.resolve(RequestContext(api_key=api_key))

    assert policy.org_id == "org_123"
    assert policy.api_key_id == "key_123"
    assert policy.allowed_operations == frozenset({"recall", "reflect"})
    assert policy.operation_bank_scope_modes == {"recall": "all", "reflect": "all"}
    resolver._rest_get.assert_any_call(  # type: ignore[attr-defined]
        "hindsight_api_keys",
        select="id,org_id,created_by_user_id,permission_mode,allowed_operations,revoked_at,expires_at",
        key_hash=f"eq.{key_hash}",
        revoked_at="is.null",
        limit="1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "can_create_bank"),
    [("admin", True), ("member", False)],
)
async def test_resolver_full_access_api_key_follows_creator_current_role(
    role: Literal["admin", "member"], can_create_bank: bool
) -> None:
    resolver = SupabasePolicyResolver(_resolver_config())
    resolver._rest_get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            [
                {
                    "id": "key_123",
                    "org_id": "org_123",
                    "created_by_user_id": "user_123",
                    "role": "admin",
                    "permission_mode": "full_access",
                    "allowed_operations": None,
                    "expires_at": "2099-01-01T00:00:00Z",
                }
            ],
            [{"org_id": "org_123", "user_id": "user_123", "role": role}],
            [{"id": "org_123", "name": "Org", "config": {}}],
        ]
    )

    policy = await resolver.resolve(RequestContext(api_key="hs_test_secret_with_enough_entropy"))

    assert policy.role == role
    expected_operations = (
        ALL_DATAPLANE_OPERATIONS if role == "admin" else ALL_DATAPLANE_OPERATIONS - UNSCOPED_DATAPLANE_OPERATIONS
    )
    assert policy.allowed_operations == expected_operations
    assert ("create_bank" in policy.allowed_operations) is can_create_bank
    assert policy.operation_bank_scope_modes is None
    assert policy.operation_bank_internal_ids is None
    assert resolver._rest_get.await_count == 3  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_resolver_rejects_expired_hindsight_api_key() -> None:
    resolver = SupabasePolicyResolver(_resolver_config())
    resolver._rest_get = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            {
                "id": "key_123",
                "org_id": "org_123",
                "created_by_user_id": "user_123",
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
                    "created_by_user_id": "user_123",
                    "role": "member",
                    "allowed_operations": [],
                    "expires_at": "2099-01-01T00:00:00Z",
                }
            ],
            [{"org_id": "org_123", "user_id": "user_123", "role": "member"}],
            [],
            [{"id": "org_123", "name": "Org", "config": {}}],
        ]
    )

    policy = await resolver.resolve(RequestContext(api_key="hs_test_secret_with_enough_entropy"))

    assert policy.allowed_operations == frozenset()


@pytest.mark.asyncio
async def test_resolver_intersects_api_key_operations_with_creator_current_role() -> None:
    resolver = SupabasePolicyResolver(_resolver_config())
    resolver._rest_get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            [
                {
                    "id": "key_123",
                    "org_id": "org_123",
                    "created_by_user_id": "user_123",
                    "role": "admin",
                    "allowed_operations": ["retain", "recall", "list_documents"],
                    "expires_at": "2099-01-01T00:00:00Z",
                }
            ],
            [{"org_id": "org_123", "user_id": "user_123", "role": "member"}],
            [
                {"operation": "retain", "bank_scope_mode": "all"},
                {"operation": "recall", "bank_scope_mode": "all"},
                {"operation": "list_documents", "bank_scope_mode": "all"},
            ],
            [{"id": "org_123", "name": "Org", "config": {}}],
        ]
    )

    policy = await resolver.resolve(RequestContext(api_key="hs_test_secret_with_enough_entropy"))

    assert policy.role == "member"
    assert policy.allowed_operations == frozenset({"retain", "recall", "list_documents"})


@pytest.mark.asyncio
async def test_resolver_preserves_unscoped_create_bank_api_key_operation() -> None:
    resolver = SupabasePolicyResolver(_resolver_config())
    resolver._rest_get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            [
                {
                    "id": "key_123",
                    "org_id": "org_123",
                    "created_by_user_id": "user_123",
                    "role": "admin",
                    "allowed_operations": ["create_bank", "recall"],
                    "expires_at": "2099-01-01T00:00:00Z",
                }
            ],
            [{"org_id": "org_123", "user_id": "user_123", "role": "admin"}],
            [{"operation": "recall", "bank_scope_mode": "all"}],
            [{"id": "org_123", "name": "Org", "config": {}}],
        ]
    )

    policy = await resolver.resolve(RequestContext(api_key="hs_test_secret_with_enough_entropy"))

    assert policy.allowed_operations == frozenset({"create_bank", "recall"})


@pytest.mark.asyncio
async def test_resolver_maps_selected_api_key_bank_scope() -> None:
    resolver = SupabasePolicyResolver(_resolver_config())
    resolver._rest_get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            [
                {
                    "id": "key_123",
                    "org_id": "org_123",
                    "created_by_user_id": "user_123",
                    "role": "admin",
                    "allowed_operations": ["retain", "recall"],
                    "expires_at": "2099-01-01T00:00:00Z",
                }
            ],
            [{"org_id": "org_123", "user_id": "user_123", "role": "admin"}],
            [
                {"operation": "retain", "bank_scope_mode": "selected"},
                {"operation": "recall", "bank_scope_mode": "all"},
            ],
            [{"operation": "retain", "bank_id": "bank_a", "bank_internal_id": "internal_a"}],
            [{"id": "org_123", "name": "Org", "config": {}}],
        ]
    )

    policy = await resolver.resolve(RequestContext(api_key="hs_test_secret_with_enough_entropy"))

    assert policy.operation_bank_scope_modes == {"retain": "selected", "recall": "all"}
    assert policy.operation_bank_internal_ids == {"retain": frozenset({"internal_a"})}


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
    assert context.subject_id == "user_123"
    assert context.subject_type == "user"
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
async def test_authorization_accepts_bank_operation_enums() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(
        return_value=_policy(
            role="admin",
            allowed_operations=frozenset({"get_bank_profile", "delete_bank", "run_consolidation"}),
        )
    )  # type: ignore[method-assign]

    read_result = await validator.validate_bank_read(
        BankReadContext(
            bank_id="bank_a",
            operation=BankReadOperation.GET_BANK_PROFILE,
            request_context=_jwt_context(),
        )
    )
    write_result = await validator.validate_bank_write(
        BankWriteContext(
            bank_id="bank_a",
            operation=BankWriteOperation.DELETE_BANK,
            request_context=_jwt_context(),
        )
    )
    consolidation_result = await validator.validate_bank_write(
        BankWriteContext(
            bank_id="bank_a",
            operation=BankWriteOperation.RUN_CONSOLIDATION,
            request_context=_jwt_context(),
        )
    )

    assert read_result.allowed is True
    assert write_result.allowed is True
    assert consolidation_result.allowed is True


@pytest.mark.asyncio
async def test_authorization_allows_admin_create_bank() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(return_value=_policy(role="admin"))  # type: ignore[method-assign]

    result = await validator.validate_create_bank(CreateBankContext(bank_id="bank_new", request_context=_jwt_context()))

    assert result.allowed is True


@pytest.mark.asyncio
async def test_authorization_denies_member_create_bank() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(return_value=_policy(role="member"))  # type: ignore[method-assign]

    result = await validator.validate_create_bank(CreateBankContext(bank_id="bank_new", request_context=_jwt_context()))

    assert result.allowed is False
    assert result.status_code == 403


@pytest.mark.asyncio
async def test_authorization_empty_scoped_key_does_not_inherit_creator_role() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(  # type: ignore[method-assign]
        return_value=_policy(
            role="admin",
            api_key_id="key_123",
            allowed_operations=frozenset(),
            operation_bank_scope_modes={},
        )
    )

    result = await validator.validate_create_bank(CreateBankContext(bank_id="bank_new", request_context=_jwt_context()))

    assert result.allowed is False
    assert result.status_code == 403


@pytest.mark.asyncio
async def test_authorization_create_bank_is_unscoped_for_selected_api_key() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(
        return_value=_policy(
            role="admin",
            allowed_operations=frozenset({"create_bank"}),
            operation_bank_scope_modes={},
            operation_bank_internal_ids={},
            api_key_id="key_123",
        )
    )  # type: ignore[method-assign]
    validator._get_bank_internal_id = AsyncMock(side_effect=AssertionError("create_bank must not check bank scope"))  # type: ignore[method-assign]

    result = await validator.validate_create_bank(CreateBankContext(bank_id="bank_new", request_context=_jwt_context()))

    assert result.allowed is True
    validator._get_bank_internal_id.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_mcp_filter_keeps_create_bank_when_current_bank_out_of_scope() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(
        return_value=_policy(
            role="admin",
            allowed_operations=frozenset({"create_bank", "recall"}),
            operation_bank_scope_modes={"recall": "selected"},
            operation_bank_internal_ids={"recall": frozenset({"internal_a"})},
            api_key_id="key_123",
        )
    )  # type: ignore[method-assign]
    validator._can_access_owned_bank = AsyncMock(return_value=False)  # type: ignore[method-assign]

    tools = await validator.filter_mcp_tools(
        "bank_b",
        _jwt_context(),
        frozenset({"create_bank", "recall"}),
    )

    assert tools == frozenset({"create_bank"})


@pytest.mark.asyncio
async def test_authorization_allows_admin_api_key_retain() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(return_value=_admin_api_key_policy())  # type: ignore[method-assign]

    result = await validator.validate_retain(
        RetainContext(bank_id="bank_a", contents=[], request_context=_jwt_context())
    )

    assert result.allowed is True


@pytest.mark.asyncio
async def test_authorization_allows_member_read_operations() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(return_value=_policy(role="member"))  # type: ignore[method-assign]

    bank_read = await validator.validate_bank_read(
        BankReadContext(bank_id="bank_a", operation="list_documents", request_context=_jwt_context())
    )
    recall = await validator.validate_recall(RecallContext(bank_id="bank_a", query="q", request_context=_jwt_context()))
    reflect = await validator.validate_reflect(
        ReflectContext(bank_id="bank_a", query="q", request_context=_jwt_context())
    )

    assert bank_read.allowed is True
    assert recall.allowed is True
    assert reflect.allowed is True


@pytest.mark.asyncio
async def test_authorization_enforces_specialized_validate_hooks() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    context = _jwt_context()
    context.auth_policy = CallerPolicy(
        org_id="org_123",
        schema_name="org_org_123",
        user_id=None,
        api_key_id="key_123",
        role="admin",
        allowed_operations=frozenset({"mental_model_get", "mental_model_refresh", "consolidate"}),
        operation_bank_scope_modes={
            "mental_model_get": "all",
            "mental_model_refresh": "all",
            "consolidate": "all",
        },
        operation_bank_internal_ids={},
        tenant_config={},
    )

    mental_model_get = await validator.validate_mental_model_get(
        MentalModelGetContext(bank_id="bank_a", mental_model_id="mm_1", request_context=context)
    )
    mental_model_refresh = await validator.validate_mental_model_refresh(
        MentalModelRefreshContext(bank_id="bank_a", mental_model_id="mm_1", request_context=context)
    )
    consolidate = await validator.validate_consolidate(ConsolidateContext(bank_id="bank_a", request_context=context))

    assert mental_model_get.allowed is True
    assert mental_model_refresh.allowed is True
    assert consolidate.allowed is True


@pytest.mark.asyncio
async def test_authorization_allows_all_scope_api_key_on_any_bank() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(return_value=_admin_api_key_policy())  # type: ignore[method-assign]

    result = await validator.validate_retain(
        RetainContext(bank_id="bank_b", contents=[], request_context=_jwt_context())
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
async def test_authorization_allows_member_usable_writes() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(return_value=_policy(role="member"))  # type: ignore[method-assign]

    retain = await validator.validate_retain(
        RetainContext(bank_id="bank_a", contents=[], request_context=_jwt_context())
    )
    mental_model = await validator.validate_bank_write(
        BankWriteContext(bank_id="bank_a", operation="create_mental_model", request_context=_jwt_context())
    )
    update_document = await validator.validate_bank_write(
        BankWriteContext(bank_id="bank_a", operation="update_document", request_context=_jwt_context())
    )

    assert retain.allowed is True
    assert mental_model.allowed is True
    assert update_document.allowed is True


@pytest.mark.asyncio
async def test_authorization_allows_member_bank_scoped_writes() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(return_value=_policy(role="member"))  # type: ignore[method-assign]

    result = await validator.validate_bank_write(
        BankWriteContext(bank_id="bank_a", operation="delete_bank", request_context=_jwt_context())
    )

    assert result.allowed is True


@pytest.mark.asyncio
async def test_authorization_enforces_api_key_operation_scope() -> None:
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
    assert wrong_bank.allowed is True


@pytest.mark.asyncio
async def test_authorization_denies_selected_api_key_unscoped_existing_bank() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(
        return_value=_policy(
            role="admin",
            allowed_operations=frozenset({"recall"}),
            operation_bank_scope_modes={"recall": "selected"},
            operation_bank_internal_ids={"recall": frozenset({"internal_a"})},
            api_key_id="key_123",
        )
    )  # type: ignore[method-assign]
    validator._get_bank_internal_id = AsyncMock(return_value="internal_b")  # type: ignore[method-assign]

    result = await validator.validate_recall(RecallContext(bank_id="bank_b", query="q", request_context=_jwt_context()))

    assert result.allowed is False
    assert result.status_code == 403


@pytest.mark.asyncio
async def test_authorization_cleans_deleted_bank_references() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.delete_bank_references = AsyncMock()  # type: ignore[method-assign]
    context = _jwt_context()

    await validator.on_bank_delete_complete(
        BankDeleteResult(
            bank_id="bank_deleted",
            bank_internal_id="internal_deleted",
            request_context=context,
        )
    )

    validator.resolver.delete_bank_references.assert_awaited_once_with("internal_deleted")  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_authorization_records_api_key_created_bank_from_completion_hook() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.record_api_key_created_bank = AsyncMock()  # type: ignore[method-assign]
    context = _jwt_context()
    context.auth_policy = _policy(
        role="admin",
        allowed_operations=frozenset({"create_bank"}),
        operation_bank_scope_modes={},
        operation_bank_internal_ids={},
        api_key_id="key_123",
    )

    await validator.on_bank_create_complete(
        BankCreateResult(
            bank_id="bank_created",
            bank_internal_id="internal_created",
            request_context=context,
        )
    )

    validator.resolver.record_api_key_created_bank.assert_awaited_once_with(  # type: ignore[attr-defined]
        api_key_id="key_123",
        bank_id="bank_created",
        bank_internal_id="internal_created",
    )


@pytest.mark.asyncio
async def test_resolver_deletes_all_authorization_references_for_bank() -> None:
    resolver = SupabasePolicyResolver(_resolver_config())
    resolver._rest_rpc = AsyncMock()  # type: ignore[method-assign]

    await resolver.delete_bank_references("internal_deleted")

    resolver._rest_rpc.assert_awaited_once_with(  # type: ignore[attr-defined]
        "delete_hindsight_bank_references",
        {"p_bank_internal_id": "internal_deleted"},
    )


@pytest.mark.asyncio
async def test_resolver_excludes_tombstoned_metadata_from_online_authorization() -> None:
    resolver = SupabasePolicyResolver(_resolver_config())
    resolver._rest_get = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await resolver.api_key_owns_bank_internal_id("key_123", "internal_deleted")

    resolver._rest_get.assert_any_call(  # type: ignore[attr-defined]
        "hindsight_api_key_created_banks",
        select="api_key_id,bank_id,bank_internal_id",
        api_key_id="eq.key_123",
        bank_internal_id="eq.internal_deleted",
        deleted_at="is.null",
        limit="1",
    )


@pytest.mark.asyncio
async def test_authorization_allows_api_key_full_access_to_owned_bank() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    context = _jwt_context()
    context.auth_policy = CallerPolicy(
        org_id="org_123",
        schema_name="org_org_123",
        user_id=None,
        api_key_id="key_123",
        role="admin",
        allowed_operations=frozenset({"create_bank"}),
        operation_bank_scope_modes={},
        operation_bank_internal_ids={},
        tenant_config={},
    )
    validator._can_access_owned_bank = AsyncMock(return_value=True)  # type: ignore[method-assign]

    result = await validator.validate_bank_write(
        BankWriteContext(bank_id="bank_owned", operation="delete_bank", request_context=context)
    )

    assert result.allowed is True
    validator._can_access_owned_bank.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_authorization_allows_specialized_hooks_on_owned_bank() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    context = _jwt_context()
    context.auth_policy = CallerPolicy(
        org_id="org_123",
        schema_name="org_org_123",
        user_id=None,
        api_key_id="key_123",
        role="admin",
        allowed_operations=frozenset({"create_bank"}),
        operation_bank_scope_modes={},
        operation_bank_internal_ids={},
        tenant_config={},
    )
    validator._can_access_owned_bank = AsyncMock(return_value=True)  # type: ignore[method-assign]

    mental_model_get = await validator.validate_mental_model_get(
        MentalModelGetContext(bank_id="bank_owned", mental_model_id="mm_1", request_context=context)
    )
    mental_model_refresh = await validator.validate_mental_model_refresh(
        MentalModelRefreshContext(bank_id="bank_owned", mental_model_id="mm_1", request_context=context)
    )
    consolidate = await validator.validate_consolidate(
        ConsolidateContext(bank_id="bank_owned", request_context=context)
    )

    assert mental_model_get.allowed is True
    assert mental_model_refresh.allowed is True
    assert consolidate.allowed is True


@pytest.mark.asyncio
async def test_authorization_denies_api_key_unowned_bank_without_operation_scope() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    context = _jwt_context()
    context.auth_policy = CallerPolicy(
        org_id="org_123",
        schema_name="org_org_123",
        user_id=None,
        api_key_id="key_123",
        role="admin",
        allowed_operations=frozenset({"create_bank"}),
        operation_bank_scope_modes={},
        operation_bank_internal_ids={},
        tenant_config={},
    )
    validator._can_access_owned_bank = AsyncMock(return_value=False)  # type: ignore[method-assign]

    result = await validator.validate_bank_write(
        BankWriteContext(bank_id="bank_b", operation="delete_bank", request_context=context)
    )

    assert result.allowed is False
    assert result.status_code == 403


@pytest.mark.asyncio
async def test_filter_bank_list_limits_member_api_key_selected_scope() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(
        return_value=CallerPolicy(
            org_id="org_123",
            schema_name="org_org_123",
            user_id=None,
            api_key_id="key_123",
            role="member",
            allowed_operations=frozenset({"recall"}),
            operation_bank_scope_modes={"recall": "selected"},
            operation_bank_internal_ids={"recall": frozenset({"internal_a"})},
            tenant_config={},
        )
    )  # type: ignore[method-assign]

    result = await validator.filter_bank_list(
        BankListContext(
            banks=[
                {"id": "bank_a", "internal_id": "internal_a"},
                {"id": "bank_b", "internal_id": "internal_b"},
            ],
            request_context=_jwt_context(),
        )
    )

    assert result.banks == [{"id": "bank_a", "internal_id": "internal_a"}]


@pytest.mark.asyncio
async def test_filter_bank_list_includes_owned_banks_for_selected_api_key() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(
        return_value=CallerPolicy(
            org_id="org_123",
            schema_name="org_org_123",
            user_id=None,
            api_key_id="key_123",
            role="admin",
            allowed_operations=frozenset({"create_bank"}),
            operation_bank_scope_modes={"recall": "selected"},
            operation_bank_internal_ids={"recall": frozenset({"internal_a"})},
            tenant_config={},
        )
    )  # type: ignore[method-assign]
    validator._get_owned_bank_internal_ids = AsyncMock(return_value=frozenset({"internal_b"}))  # type: ignore[method-assign]

    result = await validator.filter_bank_list(
        BankListContext(
            banks=[
                {"id": "bank_a", "internal_id": "internal_a"},
                {"id": "bank_b", "internal_id": "internal_b"},
                {"id": "bank_c", "internal_id": "internal_c"},
            ],
            request_context=_jwt_context(),
        )
    )

    assert result.banks == [
        {"id": "bank_a", "internal_id": "internal_a"},
        {"id": "bank_b", "internal_id": "internal_b"},
    ]


@pytest.mark.asyncio
async def test_mcp_filter_allows_all_tools_on_owned_bank() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    context = _jwt_context()
    context.auth_policy = CallerPolicy(
        org_id="org_123",
        schema_name="org_org_123",
        user_id=None,
        api_key_id="key_123",
        role="admin",
        allowed_operations=frozenset({"create_bank"}),
        operation_bank_scope_modes={},
        operation_bank_internal_ids={},
        tenant_config={},
    )
    validator._can_access_owned_bank = AsyncMock(return_value=True)  # type: ignore[method-assign]

    tools = await validator.filter_mcp_tools(
        "bank_owned",
        context,
        frozenset({"create_bank", "delete_bank", "recall"}),
    )

    assert tools == frozenset({"create_bank", "delete_bank", "recall"})


@pytest.mark.asyncio
async def test_filter_bank_list_limits_admin_api_key_selected_scope() -> None:
    validator = SupabaseAuthorizationExtension(_resolver_config())
    validator.resolver.resolve = AsyncMock(
        return_value=CallerPolicy(
            org_id="org_123",
            schema_name="org_org_123",
            user_id=None,
            api_key_id="key_123",
            role="admin",
            allowed_operations=frozenset({"retain", "recall", "reflect"}),
            operation_bank_scope_modes={
                "retain": "selected",
                "recall": "selected",
                "reflect": "selected",
            },
            operation_bank_internal_ids={
                "retain": frozenset({"internal_a"}),
                "recall": frozenset({"internal_a"}),
                "reflect": frozenset({"internal_a"}),
            },
            tenant_config={},
        )
    )  # type: ignore[method-assign]

    result = await validator.filter_bank_list(
        BankListContext(
            banks=[
                {"id": "bank_a", "internal_id": "internal_a"},
                {"id": "bank_b", "internal_id": "internal_b"},
            ],
            request_context=_jwt_context(),
        )
    )

    assert result.banks == [{"id": "bank_a", "internal_id": "internal_a"}]


def test_validate_auth_profile_requires_both_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HINDSIGHT_API_AUTH_PROFILE", "supabase_org")

    with pytest.raises(ValueError, match="operation_validator"):
        validate_auth_profile(SupabaseOrgTenantExtension(_resolver_config()), None)

"""Supabase organization-based authn/authz extensions for Cloud-like deployments."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from jwt import PyJWK
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from hindsight_api.extensions.operation_validator import (
    BankCreateResult,
    BankDeleteResult,
    BankListContext,
    BankListResult,
    BankReadContext,
    BankReadOperation,
    BankWriteContext,
    BankWriteOperation,
    ConsolidateContext,
    CreateBankContext,
    MentalModelGetContext,
    MentalModelRefreshContext,
    OperationValidatorExtension,
    RecallContext,
    ReflectContext,
    RetainContext,
    ValidationResult,
)
from hindsight_api.extensions.tenant import AuthenticationError, Tenant, TenantContext, TenantExtension
from hindsight_api.models import RequestContext

from .supabase_authz_operations import operation_names_for_scope, operation_names_for_source
from .supabase_jwks import SupabaseJwksVerifierMixin

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 10.0
METADATA_WRITE_ATTEMPTS = 3
MIN_TOKEN_LENGTH = 20

_UUID_OR_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_JWT_SHAPE_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")
_SCHEMA_PREFIX_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

UNSCOPED_DATAPLANE_OPERATIONS = operation_names_for_scope("unscoped")
BANK_READ_OPERATIONS = operation_names_for_source("bank_read")
BANK_WRITE_OPERATIONS = operation_names_for_source("bank_write")
SPECIAL_BANK_OPERATIONS = operation_names_for_source("special_bank")
ALL_BANK_OPERATIONS = operation_names_for_scope("bank")
ALL_DATAPLANE_OPERATIONS = ALL_BANK_OPERATIONS | UNSCOPED_DATAPLANE_OPERATIONS
BankOperationName = str | BankReadOperation | BankWriteOperation
MEMBER_BANK_OPERATIONS = ALL_BANK_OPERATIONS


class OrganizationRow(BaseModel):
    """Organization row returned by Supabase PostgREST."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class MemberRow(BaseModel):
    """Organization membership row returned by Supabase PostgREST."""

    model_config = ConfigDict(extra="ignore")

    org_id: str
    user_id: str
    role: Literal["owner", "admin", "member"]


class ApiKeyRow(BaseModel):
    """Hindsight scoped API key row returned by Supabase PostgREST."""

    model_config = ConfigDict(extra="ignore")

    id: str
    org_id: str
    created_by_user_id: str | None = None
    permission_mode: Literal["scoped", "full_access"] = "scoped"
    allowed_operations: list[str] | None = None
    revoked_at: str | None = None
    expires_at: str | None = None

    @field_validator("allowed_operations", mode="before")
    @classmethod
    def coerce_allowed_operations(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return [str(item) for item in value]
        return None


class ApiKeyOperationScopeRow(BaseModel):
    """Effective bank scope mode row for one API key operation."""

    model_config = ConfigDict(extra="ignore")

    operation: str
    bank_scope_mode: Literal["all", "selected"] = "all"


class ApiKeyOperationBankScopeRow(BaseModel):
    """Selected bank scope row for one API key operation."""

    model_config = ConfigDict(extra="ignore")

    operation: str
    bank_id: str
    bank_internal_id: str | None = None


class ApiKeyCreatedBankRow(BaseModel):
    """Bank ownership row for a Hindsight scoped API key."""

    model_config = ConfigDict(extra="ignore")

    api_key_id: str
    bank_id: str
    bank_internal_id: str


@dataclass(frozen=True)
class CallerPolicy:
    """Resolved caller policy shared by tenant and operation validator extensions."""

    org_id: str
    schema_name: str
    user_id: str | None
    api_key_id: str | None
    role: Literal["owner", "admin", "member"]
    allowed_operations: frozenset[str] | None
    # None means this is a JWT/member or full-access API-key policy, neither of
    # which uses the scoped-key per-operation bank-scope model.
    operation_bank_scope_modes: dict[str, Literal["all", "selected"]] | None = None
    operation_bank_internal_ids: dict[str, frozenset[str]] | None = None
    tenant_config: dict[str, Any] = field(default_factory=dict)

    @property
    def is_admin(self) -> bool:
        return self.role in {"owner", "admin"}


@dataclass(frozen=True)
class _CachedPolicy:
    policy: CallerPolicy
    expires_at_monotonic: float


class SupabasePolicyResolver(SupabaseJwksVerifierMixin):
    """Resolve Supabase JWTs and Hindsight scoped API keys into caller policies."""

    def __init__(self, config: dict[str, str]):
        self.supabase_url = (config.get("supabase_url") or "").rstrip("/")
        self.supabase_service_key = config.get("supabase_service_key")
        self.schema_prefix = config.get("schema_prefix", "org")
        self.cache_ttl_seconds = int(config.get("policy_cache_ttl_seconds", "5"))
        self._http_client: httpx.AsyncClient | None = None
        self._jwks_keys: dict[str, PyJWK] = {}
        self._jwks_last_fetched = 0.0
        self._use_jwks = False
        self._cache: dict[str, _CachedPolicy] = {}

        if not self.supabase_url:
            raise ValueError("HINDSIGHT_AUTH_SUPABASE_URL or HINDSIGHT_API_*_SUPABASE_URL is required")
        if not self.supabase_service_key:
            raise ValueError("HINDSIGHT_AUTH_SUPABASE_SERVICE_KEY or HINDSIGHT_API_*_SUPABASE_SERVICE_KEY is required")
        if not _SCHEMA_PREFIX_RE.match(self.schema_prefix):
            raise ValueError("schema_prefix must be a valid PostgreSQL identifier component")

    async def on_startup(self) -> None:
        await self._ensure_http_client()
        await self._try_init_jwks()

    async def on_shutdown(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def resolve(self, context: RequestContext) -> CallerPolicy:
        token = context.api_key
        if not token:
            raise AuthenticationError("Missing Authorization header. Expected: Bearer <supabase_jwt|hindsight_api_key>")
        if len(token) < MIN_TOKEN_LENGTH:
            raise AuthenticationError("Invalid credential format")

        cache_key = f"{token}:{context.selected_tenant_id or ''}"
        cached = self._cache.get(cache_key)
        now = time.monotonic()
        if cached and cached.expires_at_monotonic > now:
            return cached.policy

        if _JWT_SHAPE_RE.match(token):
            policy = await self._resolve_jwt(token, context.selected_tenant_id)
        else:
            policy = await self._resolve_api_key(token)

        self._cache[cache_key] = _CachedPolicy(policy=policy, expires_at_monotonic=now + self.cache_ttl_seconds)
        return policy

    async def list_tenants(self) -> list[Tenant]:
        rows = await self._rest_get("organizations", select="id")
        organizations = TypeAdapter(list[OrganizationRow]).validate_python(rows)
        return [Tenant(schema=self._schema_for_org(row.id), tenant_id=row.id) for row in organizations]

    async def _resolve_jwt(self, token: str, selected_tenant_id: str | None) -> CallerPolicy:
        if not selected_tenant_id:
            raise AuthenticationError("Missing X-Hindsight-Tenant-Id header for Supabase JWT requests")
        self._validate_org_id(selected_tenant_id)
        user_id = await self._verify_token(token)
        member = await self._get_member(selected_tenant_id, user_id)
        if member is None:
            raise AuthenticationError("User is not a member of the selected organization")
        organization = await self._get_organization(selected_tenant_id)
        if organization is None:
            raise AuthenticationError("Selected organization does not exist")
        return CallerPolicy(
            org_id=selected_tenant_id,
            schema_name=self._schema_for_org(selected_tenant_id),
            user_id=user_id,
            api_key_id=None,
            role=member.role,
            allowed_operations=_operations_for_role(member.role),
            tenant_config=organization.config,
        )

    async def _resolve_api_key(self, api_key: str) -> CallerPolicy:
        key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        rows = await self._rest_get(
            "hindsight_api_keys",
            select="id,org_id,created_by_user_id,permission_mode,allowed_operations,revoked_at,expires_at",
            key_hash=f"eq.{key_hash}",
            revoked_at="is.null",
            limit="1",
        )
        keys = TypeAdapter(list[ApiKeyRow]).validate_python(rows)
        if not keys:
            raise AuthenticationError("Invalid Hindsight API key")
        key = keys[0]
        if key.expires_at and _is_past_timestamp(key.expires_at):
            raise AuthenticationError("Hindsight API key has expired")
        creator_member = await self._get_member(key.org_id, key.created_by_user_id) if key.created_by_user_id else None
        if creator_member is None:
            raise AuthenticationError("API key creator is no longer a member of the organization")
        creator_operations = _operations_for_role(creator_member.role)
        if key.permission_mode == "full_access":
            # Full-access keys are delegated user credentials: they intentionally
            # gain and lose permissions as their creator's current role changes.
            effective_operations = creator_operations
            operation_scope_modes = None
            operation_bank_internal_ids = None
        else:
            key_operations = frozenset(
                operation for operation in (key.allowed_operations or []) if operation in ALL_DATAPLANE_OPERATIONS
            )
            effective_operations = key_operations & creator_operations
            operation_scope_modes = {}
            operation_bank_internal_ids = {}
            scope_rows = await self._rest_get(
                "hindsight_api_key_operation_scopes",
                select="operation,bank_scope_mode",
                api_key_id=f"eq.{key.id}",
            )
            operation_scopes = TypeAdapter(list[ApiKeyOperationScopeRow]).validate_python(scope_rows)
            operation_scope_modes = {
                scope.operation: scope.bank_scope_mode
                for scope in operation_scopes
                if scope.operation in effective_operations
            }
            selected_operations = [operation for operation, mode in operation_scope_modes.items() if mode == "selected"]
            if selected_operations:
                bank_scope_rows = await self._rest_get(
                    "hindsight_api_key_operation_bank_scopes",
                    select="operation,bank_id,bank_internal_id",
                    api_key_id=f"eq.{key.id}",
                    operation=f"in.({','.join(selected_operations)})",
                )
                bank_scopes = TypeAdapter(list[ApiKeyOperationBankScopeRow]).validate_python(bank_scope_rows)
                by_operation: dict[str, set[str]] = {}
                for scope in bank_scopes:
                    if scope.operation not in effective_operations or not scope.bank_internal_id:
                        continue
                    by_operation.setdefault(scope.operation, set()).add(scope.bank_internal_id)
                operation_bank_internal_ids = {
                    operation: frozenset(internal_ids) for operation, internal_ids in by_operation.items()
                }
        organization = await self._get_organization(key.org_id)
        if organization is None:
            raise AuthenticationError("API key organization does not exist")
        return CallerPolicy(
            org_id=key.org_id,
            schema_name=self._schema_for_org(key.org_id),
            user_id=None,
            api_key_id=key.id,
            role=creator_member.role,
            allowed_operations=effective_operations,
            operation_bank_scope_modes=operation_scope_modes,
            operation_bank_internal_ids=operation_bank_internal_ids,
            tenant_config=organization.config,
        )

    async def _get_member(self, org_id: str, user_id: str) -> MemberRow | None:
        rows = await self._rest_get(
            "organization_members",
            select="org_id,user_id,role",
            org_id=f"eq.{org_id}",
            user_id=f"eq.{user_id}",
            removed_at="is.null",
            limit="1",
        )
        members = TypeAdapter(list[MemberRow]).validate_python(rows)
        return members[0] if members else None

    async def _get_organization(self, org_id: str) -> OrganizationRow | None:
        rows = await self._rest_get("organizations", select="id,name,config", id=f"eq.{org_id}", limit="1")
        organizations = TypeAdapter(list[OrganizationRow]).validate_python(rows)
        return organizations[0] if organizations else None

    async def _rest_get(self, table: str, **params: str) -> Any:
        await self._ensure_http_client()
        assert self._http_client is not None
        response = await self._http_client.get(
            f"{self.supabase_url}/rest/v1/{table}",
            params=params,
            headers={
                "apikey": self.supabase_service_key or "",
                "Authorization": f"Bearer {self.supabase_service_key}",
            },
        )
        if response.status_code >= 400:
            raise AuthenticationError(f"Supabase policy lookup failed: {response.status_code}")
        return response.json()

    async def _rest_rpc(self, function_name: str, body: dict[str, Any]) -> None:
        """Run an idempotent metadata transaction with bounded retries."""
        for attempt in range(METADATA_WRITE_ATTEMPTS):
            await self._ensure_http_client()
            assert self._http_client is not None
            try:
                response = await self._http_client.post(
                    f"{self.supabase_url}/rest/v1/rpc/{function_name}",
                    json=body,
                    headers={
                        "apikey": self.supabase_service_key or "",
                        "Authorization": f"Bearer {self.supabase_service_key}",
                        "Content-Type": "application/json",
                    },
                )
                if response.status_code < 500:
                    if response.status_code >= 400:
                        raise AuthenticationError(f"Supabase policy transaction failed: {response.status_code}")
                    return
            except httpx.TransportError:
                if attempt + 1 == METADATA_WRITE_ATTEMPTS:
                    raise
            await asyncio.sleep(0.1 * (2**attempt))
        raise AuthenticationError("Supabase policy transaction failed after retries")

    async def _rest_post(self, table: str, body: dict[str, Any]) -> None:
        """Insert idempotent metadata after a data-plane lifecycle commit."""
        await self._ensure_http_client()
        assert self._http_client is not None
        response = await self._http_client.post(
            f"{self.supabase_url}/rest/v1/{table}",
            json=body,
            headers={
                "apikey": self.supabase_service_key or "",
                "Authorization": f"Bearer {self.supabase_service_key}",
                "Content-Type": "application/json",
            },
        )
        if response.status_code == 409:
            return
        if response.status_code >= 400:
            raise AuthenticationError(f"Supabase policy update failed: {response.status_code}")

    async def record_api_key_created_bank(
        self,
        *,
        api_key_id: str,
        bank_id: str,
        bank_internal_id: str,
    ) -> None:
        await self._rest_post(
            "hindsight_api_key_created_banks",
            {
                "api_key_id": api_key_id,
                "bank_id": bank_id,
                "bank_internal_id": bank_internal_id,
            },
        )

    async def list_api_key_created_bank_internal_ids(self, api_key_id: str) -> frozenset[str]:
        rows = await self._rest_get(
            "hindsight_api_key_created_banks",
            select="api_key_id,bank_id,bank_internal_id",
            api_key_id=f"eq.{api_key_id}",
            deleted_at="is.null",
        )
        created_banks = TypeAdapter(list[ApiKeyCreatedBankRow]).validate_python(rows)
        return frozenset(row.bank_internal_id for row in created_banks)

    async def api_key_owns_bank_internal_id(self, api_key_id: str, bank_internal_id: str) -> bool:
        rows = await self._rest_get(
            "hindsight_api_key_created_banks",
            select="api_key_id,bank_id,bank_internal_id",
            api_key_id=f"eq.{api_key_id}",
            bank_internal_id=f"eq.{bank_internal_id}",
            deleted_at="is.null",
            limit="1",
        )
        return bool(TypeAdapter(list[ApiKeyCreatedBankRow]).validate_python(rows))

    async def delete_bank_references(self, bank_internal_id: str) -> None:
        """Delete all scoped-key references to a deleted bank."""
        await self._rest_rpc(
            "delete_hindsight_bank_references",
            {"p_bank_internal_id": bank_internal_id},
        )

    async def _try_init_jwks(self) -> None:
        try:
            await self._fetch_jwks()
            if self._jwks_keys:
                self._use_jwks = True
                return
        except Exception as exc:
            logger.warning("Could not fetch Supabase JWKS; falling back to /auth/v1/user: %s", exc)
        self._use_jwks = False

    async def _verify_token(self, token: str) -> str:
        return await self._verify_token_jwks(token) if self._use_jwks else await self._verify_token_legacy(token)

    async def _verify_token_legacy(self, token: str) -> str:
        await self._ensure_http_client()
        assert self._http_client is not None
        response = await self._http_client.get(
            f"{self.supabase_url}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": self.supabase_service_key or ""},
        )
        if response.status_code == 401:
            raise AuthenticationError("Invalid or expired token")
        if response.status_code != 200:
            raise AuthenticationError(f"Authentication failed: {response.status_code}")
        user_id = response.json().get("id")
        if not user_id:
            raise AuthenticationError("Token valid but no user ID found")
        return str(user_id)

    async def _ensure_http_client(self) -> None:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)

    def _schema_for_org(self, org_id: str) -> str:
        self._validate_org_id(org_id)
        return f"{self.schema_prefix}_{org_id.replace('-', '_')}"

    @staticmethod
    def _validate_org_id(org_id: str) -> None:
        if not _UUID_OR_SLUG_RE.match(org_id):
            raise AuthenticationError("Invalid organization ID")


def _resolver_config(config: dict[str, str]) -> dict[str, str]:
    """Accept both grouped auth env names and current extension-prefixed env names."""

    import os

    resolved = dict(config)
    if "supabase_url" not in resolved and os.getenv("HINDSIGHT_AUTH_SUPABASE_URL"):
        resolved["supabase_url"] = os.environ["HINDSIGHT_AUTH_SUPABASE_URL"]
    if "supabase_service_key" not in resolved and os.getenv("HINDSIGHT_AUTH_SUPABASE_SERVICE_KEY"):
        resolved["supabase_service_key"] = os.environ["HINDSIGHT_AUTH_SUPABASE_SERVICE_KEY"]
    if "schema_prefix" not in resolved and os.getenv("HINDSIGHT_AUTH_SCHEMA_PREFIX"):
        resolved["schema_prefix"] = os.environ["HINDSIGHT_AUTH_SCHEMA_PREFIX"]
    return resolved


def _is_past_timestamp(value: str) -> bool:
    normalized = value.replace("Z", "+00:00")
    try:
        expires_at = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise AuthenticationError("Hindsight API key has invalid expiration timestamp") from exc
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at <= datetime.now(UTC)


def _operations_for_role(role: Literal["owner", "admin", "member"]) -> frozenset[str]:
    if role == "member":
        return MEMBER_BANK_OPERATIONS
    return ALL_DATAPLANE_OPERATIONS


def _allowed_operations_for_policy(policy: CallerPolicy) -> frozenset[str]:
    # An empty scoped-key operation set means deny all; only None delegates the
    # decision to the caller role (used by policies without an explicit set).
    if policy.allowed_operations is not None:
        return policy.allowed_operations
    return _operations_for_role(policy.role)


def _operation_name(operation: BankOperationName) -> str:
    return operation.value if isinstance(operation, BankReadOperation | BankWriteOperation) else str(operation)


class SupabaseOrgTenantExtension(TenantExtension):
    """Tenant extension that maps Supabase users or Hindsight API keys to organization schemas."""

    auth_profile = "supabase_org"
    auth_profile_component = "tenant"
    required_auth_profile_components = frozenset({"tenant", "operation_validator"})

    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        self.resolver = SupabasePolicyResolver(_resolver_config(config))
        self._initialized_schemas: set[str] = set()
        self._schema_locks: dict[str, asyncio.Lock] = {}

    async def on_startup(self) -> None:
        await self.resolver.on_startup()

    async def on_shutdown(self) -> None:
        await self.resolver.on_shutdown()

    async def authenticate(self, context: RequestContext) -> TenantContext:
        policy = await self.resolver.resolve(context)
        await self._ensure_schema_ready(policy.schema_name)
        context.tenant_id = policy.org_id
        context.selected_tenant_id = policy.org_id
        context.subject_id = policy.user_id or policy.api_key_id
        context.subject_type = "user" if policy.user_id is not None else "api_key"
        context.api_key_id = policy.api_key_id
        context.role = policy.role
        context.allowed_bank_ids = None
        context.allowed_operations = (
            sorted(policy.allowed_operations) if policy.allowed_operations is not None else None
        )
        context.auth_policy = policy
        return TenantContext(schema_name=policy.schema_name)

    async def list_tenants(self) -> list[Tenant]:
        tenants = await self.resolver.list_tenants()
        self._initialized_schemas.update(tenant.schema for tenant in tenants)
        return tenants

    async def _ensure_schema_ready(self, schema_name: str) -> None:
        if schema_name in self._initialized_schemas:
            return
        lock = self._schema_locks.setdefault(schema_name, asyncio.Lock())
        async with lock:
            if schema_name in self._initialized_schemas:
                return
            logger.info("Initializing organization schema: %s", schema_name)
            try:
                await self.context.run_migration(schema_name)
            except Exception as exc:
                logger.error("Organization schema initialization failed for %s: %s", schema_name, exc)
                raise AuthenticationError(f"Failed to initialize organization schema: {exc!s}") from exc
            self._initialized_schemas.add(schema_name)
            logger.info("Organization schema ready: %s", schema_name)

    async def get_tenant_config(self, context: RequestContext) -> dict[str, Any]:
        policy = await self.resolver.resolve(context)
        return policy.tenant_config


class SupabaseAuthorizationExtension(OperationValidatorExtension):
    """Operation validator enforcing Cloud-like organization roles and operation-scoped API keys."""

    auth_profile = "supabase_org"
    auth_profile_component = "operation_validator"
    required_auth_profile_components = frozenset({"tenant", "operation_validator"})

    def __init__(self, config: dict[str, str]):
        super().__init__(config)
        self.resolver = SupabasePolicyResolver(_resolver_config(config))

    async def on_startup(self) -> None:
        await self.resolver.on_startup()

    async def on_shutdown(self) -> None:
        await self.resolver.on_shutdown()

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        return await self._validate_bank_operation(ctx.request_context, ctx.bank_id, "retain")

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        return await self._validate_bank_operation(ctx.request_context, ctx.bank_id, "recall")

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        return await self._validate_bank_operation(ctx.request_context, ctx.bank_id, "reflect")

    async def validate_consolidate(self, ctx: ConsolidateContext) -> ValidationResult:
        return await self._validate_bank_operation(ctx.request_context, ctx.bank_id, "consolidate")

    async def validate_mental_model_get(self, ctx: MentalModelGetContext) -> ValidationResult:
        return await self._validate_bank_operation(ctx.request_context, ctx.bank_id, "mental_model_get")

    async def validate_mental_model_refresh(self, ctx: MentalModelRefreshContext) -> ValidationResult:
        return await self._validate_bank_operation(ctx.request_context, ctx.bank_id, "mental_model_refresh")

    async def validate_bank_read(self, ctx: BankReadContext) -> ValidationResult:
        return await self._validate_bank_operation(ctx.request_context, ctx.bank_id, ctx.operation)

    async def validate_bank_write(self, ctx: BankWriteContext) -> ValidationResult:
        return await self._validate_bank_operation(ctx.request_context, ctx.bank_id, ctx.operation)

    async def on_bank_delete_complete(self, result: BankDeleteResult) -> None:
        await self.resolver.delete_bank_references(result.bank_internal_id)

    async def on_bank_create_complete(self, result: BankCreateResult) -> None:
        policy = await self._resolve_policy(result.request_context)
        if policy.api_key_id is None or "create_bank" not in _allowed_operations_for_policy(policy):
            return
        await self.resolver.record_api_key_created_bank(
            api_key_id=policy.api_key_id,
            bank_id=result.bank_id,
            bank_internal_id=result.bank_internal_id,
        )

    async def validate_create_bank(self, ctx: CreateBankContext) -> ValidationResult:
        policy = await self._resolve_policy(ctx.request_context)
        allowed_operations = _allowed_operations_for_policy(policy)
        if "create_bank" not in allowed_operations:
            return ValidationResult.reject("Caller is not allowed to create banks", status_code=403)
        return ValidationResult.accept()

    async def filter_bank_list(self, ctx: BankListContext) -> BankListResult:
        policy = await self._resolve_policy(ctx.request_context)
        if policy.operation_bank_scope_modes is None:
            return BankListResult(banks=ctx.banks)
        visible_internal_ids = await self._get_visible_bank_internal_ids(policy, ctx.request_context)
        if visible_internal_ids is None:
            return BankListResult(banks=ctx.banks)
        filtered = [bank for bank in ctx.banks if str(bank.get("internal_id") or "") in visible_internal_ids]
        return BankListResult(banks=filtered)

    async def _get_visible_bank_internal_ids(
        self,
        policy: CallerPolicy,
        request_context: RequestContext,
    ) -> frozenset[str] | None:
        if policy.operation_bank_scope_modes is None:
            return None
        if any(
            mode == "all" for operation, mode in policy.operation_bank_scope_modes.items() if operation != "create_bank"
        ):
            return None
        owned_internal_ids = await self._get_owned_bank_internal_ids(policy, request_context)
        selected_internal_ids = frozenset().union(*(policy.operation_bank_internal_ids or {}).values())
        return selected_internal_ids | owned_internal_ids

    async def filter_mcp_tools(
        self,
        bank_id: str,
        request_context: RequestContext,
        tools: frozenset[str],
    ) -> frozenset[str]:
        policy = await self._resolve_policy(request_context)
        if await self._can_access_owned_bank(policy, bank_id, request_context):
            return tools
        if policy.allowed_operations is None:
            allowed_tools = tools
        else:
            allowed_tools = frozenset(tool for tool in tools if self._tool_allowed(tool, policy.allowed_operations))
        bank_tools = allowed_tools - frozenset({"create_bank"})
        create_tools = allowed_tools & frozenset({"create_bank"})
        if not bank_tools:
            return create_tools
        if not await self._can_access_bank_for_operation(policy, bank_id, "get_bank_profile", request_context):
            return create_tools
        return allowed_tools

    async def _validate_bank_operation(
        self,
        request_context: RequestContext,
        bank_id: str,
        operation: BankOperationName,
    ) -> ValidationResult:
        operation_name = _operation_name(operation)
        policy = await self._resolve_policy(request_context)
        if await self._can_access_owned_bank(policy, bank_id, request_context):
            return ValidationResult.accept()
        allowed_operations = _allowed_operations_for_policy(policy)
        if operation_name not in allowed_operations:
            return ValidationResult.reject("Caller is not allowed to perform this operation", status_code=403)
        if not await self._can_access_bank_for_operation(policy, bank_id, operation_name, request_context):
            return ValidationResult.reject("Caller is not allowed to access this bank", status_code=403)
        return ValidationResult.accept()

    async def _resolve_policy(self, request_context: RequestContext) -> CallerPolicy:
        cached = request_context.auth_policy
        if isinstance(cached, CallerPolicy):
            return cached
        policy = await self.resolver.resolve(request_context)
        request_context.auth_policy = policy
        return policy

    async def _can_access_bank_for_operation(
        self,
        policy: CallerPolicy,
        bank_id: str,
        operation: BankOperationName,
        request_context: RequestContext,
    ) -> bool:
        operation_name = _operation_name(operation)
        if policy.operation_bank_scope_modes is None:
            return True
        mode = policy.operation_bank_scope_modes.get(operation_name)
        if mode is None:
            return False
        if mode == "all":
            return True
        current_internal_id = await self._get_bank_internal_id(bank_id, request_context)
        return bool(
            current_internal_id
            and current_internal_id in (policy.operation_bank_internal_ids or {}).get(operation_name, frozenset())
        )

    async def _can_access_owned_bank(
        self,
        policy: CallerPolicy,
        bank_id: str,
        request_context: RequestContext,
    ) -> bool:
        if policy.api_key_id is None:
            return False
        allowed_operations = _allowed_operations_for_policy(policy)
        if "create_bank" not in allowed_operations:
            return False
        current_internal_id = await self._get_bank_internal_id(bank_id, request_context)
        if not current_internal_id:
            return False
        return await self.resolver.api_key_owns_bank_internal_id(policy.api_key_id, current_internal_id)

    async def _get_owned_bank_internal_ids(
        self,
        policy: CallerPolicy,
        request_context: RequestContext,
    ) -> frozenset[str]:
        if policy.api_key_id is None:
            return frozenset()
        allowed_operations = _allowed_operations_for_policy(policy)
        if "create_bank" not in allowed_operations:
            return frozenset()
        return await self.resolver.list_api_key_created_bank_internal_ids(policy.api_key_id)

    async def _get_bank_internal_id(self, bank_id: str, request_context: RequestContext) -> str | None:
        memory = self.context.get_memory_engine()
        return await memory.get_bank_internal_id(bank_id, request_context=request_context)

    @staticmethod
    def _tool_allowed(tool_name: str, allowed_operations: frozenset[str]) -> bool:
        tool_operations: dict[str, frozenset[str]] = {
            "retain": frozenset({"retain"}),
            "sync_retain": frozenset({"retain"}),
            "recall": frozenset({"recall"}),
            "reflect": frozenset({"reflect"}),
            "list_banks": ALL_BANK_OPERATIONS,
            "create_bank": frozenset({"create_bank"}),
            "get_bank": frozenset({"get_bank_profile"}),
            "get_bank_stats": frozenset({"get_bank_stats"}),
            "update_bank": frozenset({"update_bank"}),
            "delete_bank": frozenset({"delete_bank"}),
            "clear_memories": frozenset({"delete_bank"}),
            "list_mental_models": frozenset({"list_mental_models"}),
            "get_mental_model": frozenset({"mental_model_get"}),
            "create_mental_model": frozenset({"create_mental_model"}),
            "update_mental_model": frozenset({"update_mental_model"}),
            "delete_mental_model": frozenset({"delete_mental_model"}),
            "refresh_mental_model": frozenset({"mental_model_refresh"}),
            "clear_mental_model": frozenset({"clear_mental_model"}),
            "list_directives": frozenset({"list_directives"}),
            "create_directive": frozenset({"create_directive"}),
            "delete_directive": frozenset({"delete_directive"}),
            "list_memories": frozenset({"list_memory_units"}),
            "get_memory": frozenset({"get_memory_unit"}),
            "update_memory": frozenset({"update_memory_unit"}),
            "invalidate_memory": frozenset({"update_memory_unit"}),
            "list_documents": frozenset({"list_documents"}),
            "get_document": frozenset({"get_document"}),
            "delete_document": frozenset({"delete_document"}),
            "list_operations": frozenset({"list_operations"}),
            "get_operation": frozenset({"get_operation_status"}),
            "cancel_operation": frozenset({"cancel_operation"}),
            "list_tags": frozenset({"list_tags"}),
        }
        required = tool_operations.get(tool_name)
        return required is not None and not required.isdisjoint(allowed_operations)

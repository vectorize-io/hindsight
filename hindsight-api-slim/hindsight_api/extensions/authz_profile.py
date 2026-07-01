"""Deployment profile validation for grouped authn/authz integrations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

ENV_AUTHZ_PROFILE = "HINDSIGHT_API_AUTHZ_PROFILE"
SUPABASE_ORG_PROFILE = "supabase_org"
SUPABASE_ORG_TENANT_CLASS = "SupabaseOrgTenantExtension"
SUPABASE_ORG_VALIDATOR_CLASS = "SupabaseAuthorizationExtension"


@dataclass(frozen=True)
class AuthzProfileInfo:
    """Authz deployment state exposed to control-plane health/version checks."""

    authz_profile: str
    tenant_extension: str | None
    operation_validator_extension: str | None
    supabase_org_ready: bool


def _class_name(instance: Any | None) -> str | None:
    return instance.__class__.__name__ if instance is not None else None


def get_authz_profile_info(tenant_extension: Any | None, operation_validator: Any | None) -> AuthzProfileInfo:
    """Return normalized authz profile information for diagnostics and feature gates."""

    profile = os.getenv(ENV_AUTHZ_PROFILE, "disabled").strip() or "disabled"
    tenant_name = _class_name(tenant_extension)
    validator_name = _class_name(operation_validator)
    return AuthzProfileInfo(
        authz_profile=profile,
        tenant_extension=tenant_name,
        operation_validator_extension=validator_name,
        supabase_org_ready=(
            profile == SUPABASE_ORG_PROFILE
            and tenant_name == SUPABASE_ORG_TENANT_CLASS
            and validator_name == SUPABASE_ORG_VALIDATOR_CLASS
        ),
    )


def validate_authz_profile(tenant_extension: Any | None, operation_validator: Any | None) -> None:
    """Fail fast when a deployment profile is only partially configured."""

    info = get_authz_profile_info(tenant_extension, operation_validator)

    has_supabase_org_piece = (
        info.tenant_extension == SUPABASE_ORG_TENANT_CLASS
        or info.operation_validator_extension == SUPABASE_ORG_VALIDATOR_CLASS
    )
    if info.authz_profile == SUPABASE_ORG_PROFILE:
        if not info.supabase_org_ready:
            raise RuntimeError(
                "HINDSIGHT_API_AUTHZ_PROFILE=supabase_org requires "
                "HINDSIGHT_API_TENANT_EXTENSION=...:SupabaseOrgTenantExtension and "
                "HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION=...:SupabaseAuthorizationExtension. "
                f"Got tenant_extension={info.tenant_extension!r}, "
                f"operation_validator_extension={info.operation_validator_extension!r}."
            )
        return

    if has_supabase_org_piece:
        raise RuntimeError(
            "Supabase org authz extensions are a grouped deployment profile. "
            "Set HINDSIGHT_API_AUTHZ_PROFILE=supabase_org and configure both "
            "SupabaseOrgTenantExtension and SupabaseAuthorizationExtension."
        )

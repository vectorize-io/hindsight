"""Deployment-profile validation for grouped authn/authz extensions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

AUTH_PROFILE_ENV = "HINDSIGHT_API_AUTH_PROFILE"
DISABLED_PROFILE = "disabled"


@dataclass(frozen=True)
class AuthProfileInfo:
    """Normalized auth profile diagnostics for startup checks and /version."""

    auth_profile: str
    tenant_extension: str | None
    operation_validator_extension: str | None
    auth_profile_ready: bool
    missing_components: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ComponentSlot:
    name: str
    extension: Any | None


def get_configured_auth_profile() -> str:
    """Return the configured auth profile name, defaulting to disabled."""

    return (os.getenv(AUTH_PROFILE_ENV) or DISABLED_PROFILE).strip() or DISABLED_PROFILE


def get_auth_profile_info(tenant_extension: Any | None, operation_validator: Any | None) -> AuthProfileInfo:
    """Return normalized auth profile information for diagnostics and feature gates."""

    profile = get_configured_auth_profile()
    components = _component_slots(tenant_extension, operation_validator)
    required = _required_components_for_profile(profile, components)
    missing = tuple(name for name in sorted(required) if not _has_component(components, profile, name))

    return AuthProfileInfo(
        auth_profile=profile,
        tenant_extension=_class_name(tenant_extension),
        operation_validator_extension=_class_name(operation_validator),
        auth_profile_ready=not missing,
        missing_components=missing,
    )


def validate_auth_profile(tenant_extension: Any | None, operation_validator: Any | None) -> None:
    """Validate grouped auth profile configuration.

    A profile is opt-in. When disabled, existing deployment modes remain
    compatible. When enabled, any loaded extension that declares itself part of
    that profile can require sibling components so incomplete profiles fail
    during startup instead of partially enforcing authn/authz.
    """

    info = get_auth_profile_info(tenant_extension, operation_validator)
    if info.auth_profile == DISABLED_PROFILE:
        return

    components = _component_slots(tenant_extension, operation_validator)
    has_profile_extension = any(
        getattr(slot.extension, "auth_profile", None) == info.auth_profile for slot in components
    )
    if not has_profile_extension:
        raise ValueError(
            f"{AUTH_PROFILE_ENV}={info.auth_profile} is configured, but no loaded extension declares that profile"
        )

    if not info.auth_profile_ready:
        missing = ", ".join(info.missing_components)
        raise ValueError(
            f"{AUTH_PROFILE_ENV}={info.auth_profile} requires missing auth profile component(s): {missing}"
        )


def _component_slots(tenant_extension: Any | None, operation_validator: Any | None) -> tuple[_ComponentSlot, ...]:
    return (
        _ComponentSlot("tenant", tenant_extension),
        _ComponentSlot("operation_validator", operation_validator),
    )


def _required_components_for_profile(profile: str, components: tuple[_ComponentSlot, ...]) -> set[str]:
    if profile == DISABLED_PROFILE:
        return set()

    required: set[str] = set()
    for slot in components:
        ext = slot.extension
        if getattr(ext, "auth_profile", None) == profile:
            component = getattr(ext, "auth_profile_component", None)
            if component:
                required.add(component)
            required.update(getattr(ext, "required_auth_profile_components", frozenset()))
    return required


def _has_component(components: tuple[_ComponentSlot, ...], profile: str, name: str) -> bool:
    return any(
        getattr(slot.extension, "auth_profile", None) == profile
        and getattr(slot.extension, "auth_profile_component", None) == name
        for slot in components
    )


def _class_name(value: Any | None) -> str | None:
    return value.__class__.__name__ if value is not None else None

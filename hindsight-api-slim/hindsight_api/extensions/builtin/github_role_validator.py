"""
GitHub role-based OperationValidator for Hindsight.

Companion to
:class:`hindsight_api.extensions.builtin.github_tenant.GitHubTenantExtension`.

The tenant extension resolves the user's GitHub team-based role and encodes it
into ``RequestContext.tenant_id`` as ``gh_<id>:<role>``. This validator reads
that role and gates the core memory operations accordingly:

    - ``admin``  : all operations allowed.
    - ``member`` : all operations allowed (read/write).
    - ``viewer`` : recall allowed; retain and reflect rejected (read-only).
    - unknown    : fail closed — retain/reflect rejected.

The role is read directly from ``tenant_id`` (no additional GitHub API call),
so this validator stays cheap and stateless.

Enable alongside the GitHub tenant extension:
    HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION=hindsight_api.extensions.builtin.github_role_validator:GitHubRoleOperationValidator

License: MIT
"""

from __future__ import annotations

import logging

from hindsight_api.extensions.builtin.github_tenant import (
    ROLE_ADMIN,
    ROLE_MEMBER,
    parse_role_from_tenant_id,
)
from hindsight_api.extensions.operation_validator import (
    OperationValidatorExtension,
    RecallContext,
    ReflectContext,
    RetainContext,
    ValidationResult,
)

logger = logging.getLogger(__name__)

__all__ = ["GitHubRoleOperationValidator"]

# Roles permitted to perform write/compute operations (retain, reflect).
_WRITE_ROLES = {ROLE_ADMIN, ROLE_MEMBER}


class GitHubRoleOperationValidator(OperationValidatorExtension):
    """Gate retain/recall/reflect based on the GitHub team-derived role."""

    def __init__(self, config: dict[str, str] | None = None) -> None:
        super().__init__(config or {})

    def _can_write(self, tenant_id: str | None) -> bool:
        return parse_role_from_tenant_id(tenant_id) in _WRITE_ROLES

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        if self._can_write(ctx.request_context.tenant_id):
            return ValidationResult.accept()
        return ValidationResult.reject("Read-only role: retain is not permitted", status_code=403)

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        if self._can_write(ctx.request_context.tenant_id):
            return ValidationResult.accept()
        return ValidationResult.reject("Read-only role: reflect is not permitted", status_code=403)

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        # All roles (including viewer) may read.
        return ValidationResult.accept()

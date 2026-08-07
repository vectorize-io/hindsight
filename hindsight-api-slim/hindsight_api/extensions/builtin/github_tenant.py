"""
GitHub OAuth Tenant Extension for Hindsight.

A purpose-built configuration of :class:`OidcTenantExtension` for GitHub OAuth.

GitHub OAuth user access tokens are **opaque** (not JWTs), so this variant does
not use JWKS verification. Instead it validates tokens against the GitHub REST
API and resolves the user's identity and team/organization membership.

Two things this extension provides:

1. **Per-user schema isolation** — identical to the generic OIDC extension. Each
   GitHub user gets their own PostgreSQL schema ``{prefix}_{github_user_id}``
   (the numeric, immutable user id), so multiple users share one Hindsight
   instance with complete data separation.

2. **Team-based roles** — a user's GitHub team membership in a configured org is
   mapped to a Hindsight role (``admin`` / ``member`` / ``viewer``). The role
   **gates capabilities** (it does not change the schema):
       - ``admin``  : full access, may modify all bank config fields.
       - ``member`` : read/write memory, may modify a limited set of config fields.
       - ``viewer`` : read-only (recall allowed; retain/reflect rejected).

   Capability enforcement is split across two hooks:
       - bank-config write permissions via :meth:`get_allowed_config_fields`
         (handled here in the tenant extension), and
       - retain/recall/reflect gating via
         :class:`hindsight_api.extensions.builtin.github_role_validator.GitHubRoleOperationValidator`
         (a companion OperationValidator). The role is propagated to the
         validator through ``RequestContext.tenant_id`` encoded as
         ``gh_<id>:<role>`` so no second GitHub call is needed.

Configuration via environment variables:
    HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.github_tenant:GitHubTenantExtension
    HINDSIGHT_API_TENANT_GITHUB_ORG=my-org

    # Team slugs (comma-separated) mapped to each role
    HINDSIGHT_API_TENANT_GITHUB_ADMIN_TEAMS=platform-admins
    HINDSIGHT_API_TENANT_GITHUB_MEMBER_TEAMS=engineering,data
    HINDSIGHT_API_TENANT_GITHUB_VIEWER_TEAMS=analysts

    # Optional
    HINDSIGHT_API_TENANT_GITHUB_DEFAULT_ROLE=viewer   # role for org members in no mapped team; empty = deny
    HINDSIGHT_API_TENANT_GITHUB_API_URL=https://api.github.com  # set for GHES
    HINDSIGHT_API_TENANT_GITHUB_ROLE_CACHE_TTL=300    # seconds
    HINDSIGHT_API_TENANT_SCHEMA_PREFIX=user

Companion validator (recommended) for read/write gating:
    HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION=hindsight_api.extensions.builtin.github_role_validator:GitHubRoleOperationValidator

License: MIT
"""

from __future__ import annotations

import hashlib
import logging
import time

import httpx

from hindsight_api.extensions.builtin.oidc_tenant import (
    REQUEST_TIMEOUT_SECONDS,
    Identity,
    OidcTenantExtension,
)
from hindsight_api.extensions.tenant import AuthenticationError
from hindsight_api.models import RequestContext

logger = logging.getLogger(__name__)

__all__ = [
    "ROLE_ADMIN",
    "ROLE_MEMBER",
    "ROLE_VIEWER",
    "GitHubTenantExtension",
    "encode_tenant_id",
    "parse_role_from_tenant_id",
]

ROLE_ADMIN = "admin"
ROLE_MEMBER = "member"
ROLE_VIEWER = "viewer"

# Role precedence: a user in multiple mapped teams gets the strongest role.
_ROLE_RANK = {ROLE_VIEWER: 0, ROLE_MEMBER: 1, ROLE_ADMIN: 2}

# Config fields that the "member" role is allowed to modify on a bank.
# Subset of HindsightConfig.get_configurable_fields(); kept conservative.
_MEMBER_ALLOWED_FIELDS: set[str] = {
    "retain_chunk_size",
    "retain_custom_instructions",
    "recall_default_budget",
}

_TENANT_ID_SEP = ":"


def encode_tenant_id(github_id: str, role: str) -> str:
    """Encode the GitHub id and resolved role into a RequestContext.tenant_id."""
    return f"gh_{github_id}{_TENANT_ID_SEP}{role}"


def parse_role_from_tenant_id(tenant_id: str | None) -> str | None:
    """Extract the role from a tenant_id encoded by :func:`encode_tenant_id`."""
    if not tenant_id or _TENANT_ID_SEP not in tenant_id:
        return None
    return tenant_id.split(_TENANT_ID_SEP, 1)[1] or None


class GitHubTenantExtension(OidcTenantExtension):
    """OIDC tenant extension specialized for GitHub OAuth with team-based roles."""

    def __init__(self, config: dict[str, str]) -> None:
        # GitHub uses opaque tokens validated via the API, not JWKS/discovery.
        # Pre-fill config so the base validator does not require an OIDC issuer.
        config = dict(config)
        self.github_api_url = (config.get("github_api_url") or "https://api.github.com").rstrip("/")
        # Mark issuer present (logical) so base _validate_config passes; we never
        # actually perform OIDC discovery or JWKS for GitHub.
        config.setdefault("oidc_issuer", self.github_api_url)

        super().__init__(config)

        self.github_org = (config.get("github_org") or "").strip()
        if not self.github_org:
            raise ValueError("HINDSIGHT_API_TENANT_GITHUB_ORG is required when using GitHubTenantExtension")

        self.admin_teams = self._parse_teams(config.get("github_admin_teams"))
        self.member_teams = self._parse_teams(config.get("github_member_teams"))
        self.viewer_teams = self._parse_teams(config.get("github_viewer_teams"))

        # Role for org members not in any mapped team. Empty string = deny.
        self.default_role = (config.get("github_default_role", ROLE_VIEWER) or "").strip().lower()
        if self.default_role and self.default_role not in _ROLE_RANK:
            raise ValueError(
                f"Invalid github_default_role '{self.default_role}'. "
                f"Must be one of: {', '.join(_ROLE_RANK)} (or empty to deny)."
            )

        try:
            self.role_cache_ttl = float(config.get("github_role_cache_ttl", "300"))
        except ValueError:
            raise ValueError("HINDSIGHT_API_TENANT_GITHUB_ROLE_CACHE_TTL must be a number (seconds)")

        # token-hash -> (expires_at, role). Avoids a GitHub round-trip per request.
        self._role_cache: dict[str, tuple[float, str]] = {}

    @staticmethod
    def _parse_teams(raw: str | None) -> set[str]:
        if not raw:
            return set()
        return {t.strip().lower() for t in raw.split(",") if t.strip()}

    def _validate_config(self) -> None:
        # GitHub does not use schema_prefix-only validation differently, but we
        # do not require an OIDC issuer (the base would). schema_prefix is still
        # validated by the base via the regex check below.
        from hindsight_api.extensions.builtin.oidc_tenant import _SCHEMA_PREFIX_RE

        if not _SCHEMA_PREFIX_RE.match(self.schema_prefix):
            raise ValueError(
                f"Invalid schema_prefix '{self.schema_prefix}'. "
                "Must be a valid Postgres identifier (letters, digits, underscores, "
                "starting with a letter or underscore)."
            )

    # ------------------------------------------------------------------
    # Lifecycle (no OIDC discovery / JWKS for GitHub)
    # ------------------------------------------------------------------

    async def on_startup(self) -> None:
        logger.info("Initializing GitHub tenant extension (org=%s)", self.github_org)
        logger.info("Schema prefix: %s_", self.schema_prefix)
        self._http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)

    # ------------------------------------------------------------------
    # Identity (GitHub REST API instead of JWKS)
    # ------------------------------------------------------------------

    def _headers(self, token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _resolve_identity(self, token: str) -> Identity:
        """Validate the opaque token via GET /user and return the GitHub identity."""
        if self._http_client is None:
            raise AuthenticationError("Extension not initialized")
        try:
            response = await self._http_client.get(f"{self.github_api_url}/user", headers=self._headers(token))
        except httpx.TimeoutException:
            raise AuthenticationError("GitHub authentication timeout - please retry")
        except httpx.RequestError as e:
            raise AuthenticationError(f"GitHub connection error: {e!s}")

        if response.status_code == 401:
            raise AuthenticationError("Invalid or expired GitHub token")
        if response.status_code != 200:
            raise AuthenticationError(f"GitHub authentication failed: {response.status_code}")

        user = response.json()
        user_id = user.get("id")
        login = user.get("login")
        if user_id is None:
            raise AuthenticationError("GitHub token valid but no user id returned")
        # Use the immutable numeric id as the subject (login can change).
        return Identity(subject=str(user_id), claims={"login": login, "id": user_id})

    def _schema_for_subject(self, subject: str) -> str:
        """Per-user schema based on the immutable GitHub numeric id."""
        return f"{self.schema_prefix}_{subject}"

    # ------------------------------------------------------------------
    # Role resolution + propagation
    # ------------------------------------------------------------------

    async def _load_roles(self, token: str, identity: Identity, context: RequestContext) -> None:
        """Resolve the user's role from team membership and propagate it.

        The role is stored on ``identity.roles`` and encoded into
        ``context.tenant_id`` (``gh_<id>:<role>``) so the companion
        OperationValidator can read it without another GitHub call.
        """
        role = await self._resolve_role(token, identity)
        if role is None:
            raise AuthenticationError(f"User is not a member of any authorized team in org '{self.github_org}'")
        identity.roles = [role]
        context.tenant_id = encode_tenant_id(identity.subject, role)

    async def _resolve_role(self, token: str, identity: Identity) -> str | None:
        """Return the strongest mapped role for the user, or the default role."""
        cache_key = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = time.monotonic()
        cached = self._role_cache.get(cache_key)
        if cached and cached[0] > now:
            return cached[1] or None

        teams = await self._fetch_user_team_slugs(token)

        role: str | None = None
        for team in teams:
            if team in self.admin_teams:
                role = self._stronger(role, ROLE_ADMIN)
            elif team in self.member_teams:
                role = self._stronger(role, ROLE_MEMBER)
            elif team in self.viewer_teams:
                role = self._stronger(role, ROLE_VIEWER)

        if role is None:
            role = self.default_role or None

        self._role_cache[cache_key] = (now + self.role_cache_ttl, role or "")
        return role

    @staticmethod
    def _stronger(current: str | None, candidate: str) -> str:
        if current is None:
            return candidate
        return current if _ROLE_RANK[current] >= _ROLE_RANK[candidate] else candidate

    async def _fetch_user_team_slugs(self, token: str) -> set[str]:
        """Return the set of team slugs (in the configured org) the user belongs to."""
        if self._http_client is None:
            raise AuthenticationError("Extension not initialized")

        slugs: set[str] = set()
        page = 1
        while True:
            try:
                response = await self._http_client.get(
                    f"{self.github_api_url}/user/teams",
                    headers=self._headers(token),
                    params={"per_page": 100, "page": page},
                )
            except httpx.TimeoutException:
                raise AuthenticationError("GitHub authentication timeout - please retry")
            except httpx.RequestError as e:
                raise AuthenticationError(f"GitHub connection error: {e!s}")

            if response.status_code != 200:
                raise AuthenticationError(f"Failed to read GitHub teams: {response.status_code}")

            batch = response.json()
            if not isinstance(batch, list) or not batch:
                break
            for team in batch:
                org_login = ((team.get("organization") or {}).get("login") or "").lower()
                if org_login == self.github_org.lower():
                    slug = (team.get("slug") or "").lower()
                    if slug:
                        slugs.add(slug)
            if len(batch) < 100:
                break
            page += 1

        return slugs

    # ------------------------------------------------------------------
    # Capability gating: bank-config write permissions per role
    # ------------------------------------------------------------------

    async def get_allowed_config_fields(self, context: RequestContext, bank_id: str) -> set[str] | None:
        """Restrict which bank-config fields a role may modify.

        - admin  -> None (all configurable fields)
        - member -> a curated subset
        - viewer -> empty set (read-only)
        - unknown -> empty set (fail closed)
        """
        role = parse_role_from_tenant_id(context.tenant_id)
        if role == ROLE_ADMIN:
            return None
        if role == ROLE_MEMBER:
            return set(_MEMBER_ALLOWED_FIELDS)
        # viewer or unknown -> no config writes
        return set()

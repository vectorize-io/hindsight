"""Tests for the GitHub OAuth Tenant Extension."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from hindsight_api.extensions.builtin.github_tenant import (
    ROLE_ADMIN,
    ROLE_MEMBER,
    ROLE_VIEWER,
    GitHubTenantExtension,
    encode_tenant_id,
    parse_role_from_tenant_id,
)
from hindsight_api.extensions.tenant import AuthenticationError, TenantContext, TenantExtension
from hindsight_api.models import RequestContext


def _make_extension(**overrides) -> GitHubTenantExtension:
    config = {
        "github_org": "acme",
        "github_admin_teams": "platform-admins",
        "github_member_teams": "engineering, data",
        "github_viewer_teams": "analysts",
    }
    config.update(overrides)
    return GitHubTenantExtension(config)


def _resp(status_code=200, json_data=None):
    r = MagicMock(spec=httpx.Response)
    r.status_code = status_code
    r.json.return_value = json_data if json_data is not None else {}
    return r


def _user_resp(user_id=12345678, login="octocat"):
    return _resp(200, {"id": user_id, "login": login})


def _teams_resp(slugs, org="acme"):
    return _resp(200, [{"slug": s, "organization": {"login": org}} for s in slugs])


class TestInit:
    def test_requires_org(self):
        with pytest.raises(ValueError, match="GITHUB_ORG is required"):
            GitHubTenantExtension({})

    def test_is_tenant_extension(self):
        assert isinstance(_make_extension(), TenantExtension)

    def test_parses_team_config(self):
        ext = _make_extension()
        assert ext.admin_teams == {"platform-admins"}
        assert ext.member_teams == {"engineering", "data"}
        assert ext.viewer_teams == {"analysts"}

    def test_default_role(self):
        assert _make_extension().default_role == ROLE_VIEWER
        assert _make_extension(github_default_role="").default_role == ""

    def test_rejects_bad_default_role(self):
        with pytest.raises(ValueError, match="Invalid github_default_role"):
            _make_extension(github_default_role="superuser")

    def test_does_not_require_oidc_issuer(self):
        # GitHub uses opaque tokens; no issuer needed.
        ext = _make_extension()
        assert ext.github_api_url == "https://api.github.com"

    def test_ghes_api_url(self):
        ext = _make_extension(github_api_url="https://ghe.acme.com/api/v3/")
        assert ext.github_api_url == "https://ghe.acme.com/api/v3"


class TestTenantIdEncoding:
    def test_round_trip(self):
        tid = encode_tenant_id("12345678", ROLE_ADMIN)
        assert tid == "gh_12345678:admin"
        assert parse_role_from_tenant_id(tid) == ROLE_ADMIN

    def test_parse_none(self):
        assert parse_role_from_tenant_id(None) is None
        assert parse_role_from_tenant_id("gh_123") is None


class TestIdentityAndSchema:
    @pytest.mark.asyncio
    async def test_resolve_identity_uses_numeric_id(self):
        ext = _make_extension()
        ext._http_client = AsyncMock()
        ext._http_client.get.return_value = _user_resp(user_id=42, login="alice")
        identity = await ext._resolve_identity("tok")
        assert identity.subject == "42"
        assert identity.claims["login"] == "alice"

    @pytest.mark.asyncio
    async def test_resolve_identity_401(self):
        ext = _make_extension()
        ext._http_client = AsyncMock()
        ext._http_client.get.return_value = _resp(401)
        with pytest.raises(AuthenticationError, match="Invalid or expired GitHub token"):
            await ext._resolve_identity("tok")

    def test_schema_per_user(self):
        ext = _make_extension()
        assert ext._schema_for_subject("42") == "user_42"


class TestRoleResolution:
    @pytest.mark.asyncio
    async def test_admin_team_wins(self):
        ext = _make_extension()
        ext._http_client = AsyncMock()
        ext._http_client.get.return_value = _teams_resp(["platform-admins", "engineering"])
        role = await ext._resolve_role("tok", MagicMock())
        assert role == ROLE_ADMIN

    @pytest.mark.asyncio
    async def test_member_role(self):
        ext = _make_extension()
        ext._http_client = AsyncMock()
        ext._http_client.get.return_value = _teams_resp(["engineering"])
        assert await ext._resolve_role("tok", MagicMock()) == ROLE_MEMBER

    @pytest.mark.asyncio
    async def test_viewer_role(self):
        ext = _make_extension()
        ext._http_client = AsyncMock()
        ext._http_client.get.return_value = _teams_resp(["analysts"])
        assert await ext._resolve_role("tok", MagicMock()) == ROLE_VIEWER

    @pytest.mark.asyncio
    async def test_default_role_when_no_mapped_team(self):
        ext = _make_extension()
        ext._http_client = AsyncMock()
        ext._http_client.get.return_value = _teams_resp(["random-team"])
        assert await ext._resolve_role("tok", MagicMock()) == ROLE_VIEWER

    @pytest.mark.asyncio
    async def test_deny_when_default_empty(self):
        ext = _make_extension(github_default_role="")
        ext._http_client = AsyncMock()
        ext._http_client.get.return_value = _teams_resp(["random-team"])
        assert await ext._resolve_role("tok", MagicMock()) is None

    @pytest.mark.asyncio
    async def test_other_org_teams_ignored(self):
        ext = _make_extension()
        ext._http_client = AsyncMock()
        ext._http_client.get.return_value = _teams_resp(["platform-admins"], org="other-org")
        # Team belongs to a different org -> falls back to default role.
        assert await ext._resolve_role("tok", MagicMock()) == ROLE_VIEWER

    @pytest.mark.asyncio
    async def test_role_is_cached(self):
        ext = _make_extension()
        ext._http_client = AsyncMock()
        ext._http_client.get.return_value = _teams_resp(["engineering"])
        await ext._resolve_role("tok", MagicMock())
        await ext._resolve_role("tok", MagicMock())
        # Only one teams fetch despite two resolutions.
        assert ext._http_client.get.await_count == 1


class TestAuthenticateEndToEnd:
    @pytest.mark.asyncio
    async def test_authenticate_sets_schema_and_role(self):
        ext = _make_extension()
        ext._http_client = AsyncMock()

        def get(url, **kwargs):
            if url.endswith("/user"):
                return _user_resp(user_id=999, login="bob")
            return _teams_resp(["platform-admins"])

        ext._http_client.get.side_effect = get
        mock_ctx = MagicMock()
        mock_ctx.run_migration = AsyncMock()
        ext.set_context(mock_ctx)

        req = RequestContext(api_key="a" * 40)
        result = await ext.authenticate(req)
        assert isinstance(result, TenantContext)
        assert result.schema_name == "user_999"
        assert req.tenant_id == "gh_999:admin"
        mock_ctx.run_migration.assert_awaited_once_with("user_999")

    @pytest.mark.asyncio
    async def test_authenticate_denies_unauthorized_user(self):
        ext = _make_extension(github_default_role="")
        ext._http_client = AsyncMock()

        def get(url, **kwargs):
            if url.endswith("/user"):
                return _user_resp(user_id=7, login="nobody")
            return _teams_resp(["unrelated"])

        ext._http_client.get.side_effect = get
        ext.set_context(MagicMock())

        with pytest.raises(AuthenticationError, match="not a member of any authorized team"):
            await ext.authenticate(RequestContext(api_key="a" * 40))


class TestAllowedConfigFields:
    @pytest.mark.asyncio
    async def test_admin_all_fields(self):
        ext = _make_extension()
        ctx = RequestContext(api_key="x", tenant_id=encode_tenant_id("1", ROLE_ADMIN))
        assert await ext.get_allowed_config_fields(ctx, "bank") is None

    @pytest.mark.asyncio
    async def test_member_subset(self):
        ext = _make_extension()
        ctx = RequestContext(api_key="x", tenant_id=encode_tenant_id("1", ROLE_MEMBER))
        fields = await ext.get_allowed_config_fields(ctx, "bank")
        assert isinstance(fields, set) and fields

    @pytest.mark.asyncio
    async def test_viewer_readonly(self):
        ext = _make_extension()
        ctx = RequestContext(api_key="x", tenant_id=encode_tenant_id("1", ROLE_VIEWER))
        assert await ext.get_allowed_config_fields(ctx, "bank") == set()

    @pytest.mark.asyncio
    async def test_unknown_fails_closed(self):
        ext = _make_extension()
        ctx = RequestContext(api_key="x", tenant_id=None)
        assert await ext.get_allowed_config_fields(ctx, "bank") == set()

"""Tests for StaticKeysTenantExtension (env-configured per-user API keys)."""

from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.extensions.builtin.multi_key_tenant import StaticKeysTenantExtension
from hindsight_api.extensions.context import ExtensionContext
from hindsight_api.extensions.loader import load_extension
from hindsight_api.extensions.tenant import AuthenticationError, Tenant, TenantContext, TenantExtension
from hindsight_api.models import RequestContext


def _make_config(**overrides) -> dict[str, str]:
    """Build a minimal valid config, overridable per test."""
    config = {
        "users": "rafael:key-a,sophie:key-b",
    }
    config.update(overrides)
    return config


def _make_extension(**overrides) -> StaticKeysTenantExtension:
    return StaticKeysTenantExtension(_make_config(**overrides))


class TestStaticKeysTenantExtensionInit:
    """Tests for initialization and configuration parsing."""

    def test_init_with_valid_config(self):
        ext = _make_extension()
        assert ext.schema_prefix == "user"
        assert ext._key_to_user == {
            "key-a": ("rafael", "user_rafael"),
            "key-b": ("sophie", "user_sophie"),
        }
        assert ext._users == {"rafael": "user_rafael", "sophie": "user_sophie"}
        assert ext.mcp_auth_disabled is False

    def test_init_missing_users(self):
        with pytest.raises(ValueError, match="HINDSIGHT_API_TENANT_USERS is required"):
            _make_extension(users="")

    def test_init_default_schema_prefix(self):
        ext = _make_extension()
        assert ext.schema_prefix == "user"

    def test_init_custom_schema_prefix(self):
        ext = _make_extension(schema_prefix="tenant")
        assert ext._users == {"rafael": "tenant_rafael", "sophie": "tenant_sophie"}

    def test_init_rejects_invalid_schema_prefix(self):
        with pytest.raises(ValueError, match="Invalid schema_prefix"):
            _make_extension(schema_prefix="1bad")

    def test_init_rejects_schema_prefix_with_dash(self):
        with pytest.raises(ValueError, match="Invalid schema_prefix"):
            _make_extension(schema_prefix="bad-prefix")

    def test_init_accepts_underscore_prefix(self):
        ext = _make_extension(schema_prefix="my_user")
        assert ext.schema_prefix == "my_user"

    def test_init_rejects_entry_without_colon(self):
        with pytest.raises(ValueError, match="Invalid HINDSIGHT_API_TENANT_USERS entry"):
            _make_extension(users="rafael")

    def test_init_rejects_empty_user_id(self):
        with pytest.raises(ValueError, match="user_id and api_key must be non-empty"):
            _make_extension(users=":key-a")

    def test_init_rejects_empty_api_key(self):
        with pytest.raises(ValueError, match="user_id and api_key must be non-empty"):
            _make_extension(users="rafael:")

    def test_init_rejects_sql_injection_user_id(self):
        with pytest.raises(ValueError, match="Invalid user_id"):
            _make_extension(users='rafael"; DROP TABLE memory_units;--:key-a')

    def test_init_rejects_user_id_starting_with_digit(self):
        with pytest.raises(ValueError, match="Invalid user_id"):
            _make_extension(users="1rafael:key-a")

    def test_init_normalizes_dashes_in_user_id(self):
        ext = _make_extension(users="my-user-1:key-a")
        assert ext._users == {"my-user-1": "user_my_user_1"}

    def test_init_multiple_keys_same_user(self):
        ext = _make_extension(users="rafael:key-a,rafael:key-b")
        assert ext._key_to_user == {
            "key-a": ("rafael", "user_rafael"),
            "key-b": ("rafael", "user_rafael"),
        }
        assert ext._users == {"rafael": "user_rafael"}

    def test_is_tenant_extension_subclass(self):
        ext = _make_extension()
        assert isinstance(ext, TenantExtension)


class TestStaticKeysTenantExtensionAuthenticate:
    """Tests for authentication."""

    @pytest.mark.asyncio
    async def test_authenticate_valid_key(self):
        ext = _make_extension()
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        result = await ext.authenticate(RequestContext(api_key="key-a"))

        assert isinstance(result, TenantContext)
        assert result.schema_name == "user_rafael"
        mock_context.run_migration.assert_called_once_with("user_rafael")

    @pytest.mark.asyncio
    async def test_authenticate_missing_key(self):
        ext = _make_extension()
        with pytest.raises(AuthenticationError, match="Missing Authorization header"):
            await ext.authenticate(RequestContext(api_key=None))

    @pytest.mark.asyncio
    async def test_authenticate_unknown_key(self):
        ext = _make_extension()
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            await ext.authenticate(RequestContext(api_key="wrong-key"))

    @pytest.mark.asyncio
    async def test_authenticate_sets_usage_metering_fields(self):
        ext = _make_extension()
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        ctx = RequestContext(api_key="key-a")
        await ext.authenticate(ctx)

        assert ctx.tenant_id == "rafael"
        assert ctx.api_key_id == "rafael"

    @pytest.mark.asyncio
    async def test_authenticate_provisions_schema_once(self):
        ext = _make_extension()
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        await ext.authenticate(RequestContext(api_key="key-a"))
        await ext.authenticate(RequestContext(api_key="key-a"))

        mock_context.run_migration.assert_called_once_with("user_rafael")

    @pytest.mark.asyncio
    async def test_authenticate_schema_init_failure_not_cached(self):
        ext = _make_extension()
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock(side_effect=RuntimeError("Migration failed"))
        ext._context = mock_context

        with pytest.raises(AuthenticationError, match="Failed to initialize tenant"):
            await ext.authenticate(RequestContext(api_key="key-a"))

        assert "user_rafael" not in ext._initialized_schemas


class TestStaticKeysTenantExtensionMcp:
    """Tests for MCP auth."""

    @pytest.mark.asyncio
    async def test_authenticate_mcp_delegates(self):
        ext = _make_extension()
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        with patch.object(ext, "authenticate") as mock_authenticate:
            mock_authenticate.return_value = TenantContext(schema_name="user_rafael")
            result = await ext.authenticate_mcp(RequestContext(api_key="key-a"))
            mock_authenticate.assert_awaited_once()
        assert result.schema_name == "user_rafael"

    @pytest.mark.asyncio
    async def test_authenticate_mcp_disabled_skips_auth(self):
        ext = _make_extension(mcp_auth_disabled="true")
        with patch.object(ext, "authenticate") as mock_authenticate:
            result = await ext.authenticate_mcp(RequestContext(api_key="anything"))
            mock_authenticate.assert_not_awaited()
        assert result.schema_name == "public"


class TestStaticKeysTenantExtensionListTenants:
    """Tests for list_tenants."""

    @pytest.mark.asyncio
    async def test_list_tenants_returns_all_configured(self):
        ext = _make_extension()
        tenants = await ext.list_tenants()
        assert tenants == [
            Tenant(schema="user_rafael", tenant_id="rafael"),
            Tenant(schema="user_sophie", tenant_id="sophie"),
        ]

    @pytest.mark.asyncio
    async def test_list_tenants_includes_tenant_id(self):
        ext = _make_extension()
        tenants = await ext.list_tenants()
        assert all(t.tenant_id is not None for t in tenants)
        assert {t.tenant_id for t in tenants} == {"rafael", "sophie"}


class TestStaticKeysTenantExtensionLoader:
    """Tests for loading via the extension loader."""

    def test_load_via_extension_loader(self, monkeypatch):
        monkeypatch.setenv(
            "HINDSIGHT_API_TENANT_EXTENSION",
            "hindsight_api.extensions.builtin.multi_key_tenant:StaticKeysTenantExtension",
        )
        monkeypatch.setenv("HINDSIGHT_API_TENANT_USERS", "rafael:key-a,sophie:key-b")
        monkeypatch.setenv("HINDSIGHT_API_TENANT_SCHEMA_PREFIX", "tenant")

        ext = load_extension("TENANT", TenantExtension)

        assert ext is not None
        assert isinstance(ext, StaticKeysTenantExtension)
        assert ext.schema_prefix == "tenant"
        assert ext._users == {"rafael": "tenant_rafael", "sophie": "tenant_sophie"}


@pytest.fixture
def memory_with_static_keys(memory):
    """Memory engine with StaticKeysTenantExtension wired in."""
    tenant_ext = StaticKeysTenantExtension({"users": "rafael:key-a,sophie:key-b"})
    # The extension needs its ExtensionContext set for schema provisioning —
    # mirror exactly what load_extension() does in production.
    tenant_ext.set_context(memory._ext_ctx)
    memory._tenant_extension = tenant_ext
    return memory


class TestStaticKeysTenantEngineAuth:
    """Tests for tenant authentication enforced by the MemoryEngine."""

    @pytest.mark.asyncio
    async def test_retain_requires_request_context(self, memory_with_static_keys):
        with pytest.raises(AuthenticationError, match="RequestContext is required"):
            await memory_with_static_keys.retain_batch_async(
                bank_id="test-bank",
                contents=[{"content": "test"}],
                request_context=None,
            )

    @pytest.mark.asyncio
    async def test_retain_fails_with_invalid_key(self, memory_with_static_keys):
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            await memory_with_static_keys.retain_batch_async(
                bank_id="test-bank",
                contents=[{"content": "test"}],
                request_context=RequestContext(api_key="wrong-key"),
            )

    @pytest.mark.asyncio
    async def test_retain_succeeds_with_valid_key(self, memory_with_static_keys):
        # Should not raise — first call provisions the schema via run_migration.
        await memory_with_static_keys.retain_batch_async(
            bank_id="test-bank-tenant",
            contents=[{"content": "test content"}],
            request_context=RequestContext(api_key="key-a"),
        )

    @pytest.mark.asyncio
    async def test_recall_fails_with_invalid_key(self, memory_with_static_keys):
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            await memory_with_static_keys.recall_async(
                bank_id="test-bank",
                query="test query",
                fact_type=["world"],
                request_context=RequestContext(api_key="wrong-key"),
            )

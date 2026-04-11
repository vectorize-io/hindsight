"""Tests for API key schema tenant extension."""

import pytest

from hindsight_api.extensions.builtin.bank_scoped_tenant import (
    ApiKeySchemaTenantExtension,
    _parse_key_map,
)
from hindsight_api.extensions.tenant import AuthenticationError
from hindsight_api.models import RequestContext

# =========================================================================
# _parse_key_map tests
# =========================================================================


class TestParseKeyMap:
    """Tests for the key map parser."""

    def test_single_entry(self):
        result = _parse_key_map("key1:schema_a")
        assert result == {"key1": "schema_a"}

    def test_multiple_entries(self):
        result = _parse_key_map("key1:schema_a;key2:schema_b")
        assert result == {"key1": "schema_a", "key2": "schema_b"}

    def test_whitespace_handling(self):
        result = _parse_key_map("  key1 : schema_a ; key2 : schema_b  ")
        assert result == {"key1": "schema_a", "key2": "schema_b"}

    def test_invalid_no_colon(self):
        with pytest.raises(ValueError, match="Expected format"):
            _parse_key_map("key1-schema_a")

    def test_invalid_empty_key(self):
        with pytest.raises(ValueError, match="Empty API key"):
            _parse_key_map(":schema_a")

    def test_invalid_empty_schema(self):
        with pytest.raises(ValueError, match="Empty schema name"):
            _parse_key_map("key1:")

    def test_invalid_schema_not_postgres_identifier(self):
        with pytest.raises(ValueError, match="valid Postgres identifier"):
            _parse_key_map("key1:bad-schema")


# =========================================================================
# ApiKeySchemaTenantExtension tests
# =========================================================================


class TestApiKeySchemaTenantExtension:
    """Tests for the schema-isolating tenant extension."""

    def _make_ext(self, key_map: str, **kwargs) -> ApiKeySchemaTenantExtension:
        config = {"key_map": key_map, **kwargs}
        return ApiKeySchemaTenantExtension(config)

    def test_init_requires_key_map(self):
        with pytest.raises(ValueError, match="HINDSIGHT_API_TENANT_KEY_MAP is required"):
            ApiKeySchemaTenantExtension({})

    def test_init_invalid_schema_prefix(self):
        with pytest.raises(ValueError, match="Invalid schema_prefix"):
            ApiKeySchemaTenantExtension(
                {
                    "key_map": "key1:schema1",
                    "schema_prefix": "bad-prefix",
                }
            )

    def test_schema_names_without_prefix(self):
        ext = self._make_ext("key1:team_alpha;key2:team_beta")
        assert ext._key_to_schema["key1"] == "team_alpha"
        assert ext._key_to_schema["key2"] == "team_beta"

    def test_schema_names_with_prefix(self):
        ext = self._make_ext("key1:alpha;key2:beta", schema_prefix="hs")
        assert ext._key_to_schema["key1"] == "hs_alpha"
        assert ext._key_to_schema["key2"] == "hs_beta"

    @pytest.mark.asyncio
    async def test_authenticate_valid_key(self):
        ext = self._make_ext("secret123:tenant_a")
        ext._initialized_schemas.add("tenant_a")
        ctx = RequestContext(api_key="secret123")
        result = await ext.authenticate(ctx)
        assert result.schema_name == "tenant_a"

    @pytest.mark.asyncio
    async def test_authenticate_missing_key(self):
        ext = self._make_ext("secret:tenant_a")
        with pytest.raises(AuthenticationError, match="Missing API key"):
            await ext.authenticate(RequestContext(api_key=None))

    @pytest.mark.asyncio
    async def test_authenticate_wrong_key(self):
        ext = self._make_ext("secret:tenant_a")
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            await ext.authenticate(RequestContext(api_key="wrong-key"))

    @pytest.mark.asyncio
    async def test_different_keys_different_schemas(self):
        """Core isolation test: two keys resolve to different schemas."""
        ext = self._make_ext("key_a:schema_a;key_b:schema_b")
        ext._initialized_schemas.update(["schema_a", "schema_b"])

        result_a = await ext.authenticate(RequestContext(api_key="key_a"))
        result_b = await ext.authenticate(RequestContext(api_key="key_b"))

        assert result_a.schema_name == "schema_a"
        assert result_b.schema_name == "schema_b"
        assert result_a.schema_name != result_b.schema_name

    @pytest.mark.asyncio
    async def test_mcp_auth_disabled_falls_back_to_default(self):
        ext = self._make_ext("secret:tenant_a", mcp_auth_disabled="true")
        result = await ext.authenticate_mcp(RequestContext(api_key=None))
        assert result.schema_name is not None

    @pytest.mark.asyncio
    async def test_mcp_auth_enabled_rejects_missing_key(self):
        ext = self._make_ext("secret:tenant_a")
        with pytest.raises(AuthenticationError):
            await ext.authenticate_mcp(RequestContext(api_key=None))


# =========================================================================
# Prompt injection defense tests
# =========================================================================


class TestPromptInjectionDefense:
    """
    Validates the core security property: an API key can only access its
    own schema, regardless of what the agent requests. This is the defense
    against prompt injection where a compromised agent tries to access
    another tenant's memories.
    """

    @pytest.mark.asyncio
    async def test_attacker_key_cannot_reach_victim_schema(self):
        """The schema is determined solely by the API key, not the request."""
        ext = ApiKeySchemaTenantExtension(
            {
                "key_map": "victim_key:victim_schema;attacker_key:attacker_schema",
            }
        )
        ext._initialized_schemas.update(["victim_schema", "attacker_schema"])

        result = await ext.authenticate(RequestContext(api_key="attacker_key"))
        assert result.schema_name == "attacker_schema"
        assert result.schema_name != "victim_schema"

    @pytest.mark.asyncio
    async def test_unknown_key_rejected_not_defaulted(self):
        """Unknown keys must be rejected, never mapped to a default schema."""
        ext = ApiKeySchemaTenantExtension(
            {
                "key_map": "real_key:real_schema",
            }
        )

        with pytest.raises(AuthenticationError, match="Invalid API key"):
            await ext.authenticate(RequestContext(api_key="guessed_key"))

    @pytest.mark.asyncio
    async def test_empty_key_rejected(self):
        """Empty and None keys must be rejected."""
        ext = ApiKeySchemaTenantExtension(
            {
                "key_map": "real_key:real_schema",
            }
        )

        with pytest.raises(AuthenticationError, match="Missing API key"):
            await ext.authenticate(RequestContext(api_key=None))

        with pytest.raises(AuthenticationError, match="Missing API key"):
            await ext.authenticate(RequestContext(api_key=""))

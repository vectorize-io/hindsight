"""
Tests for hierarchical configuration system.

Tests config resolution hierarchy (global → tenant → bank),
key normalization, API endpoints, validation, and caching.
"""

import os

import pytest

from hindsight_api import MemoryEngine
from hindsight_api.config import HindsightConfig, normalize_config_dict, normalize_config_key
from hindsight_api.config_resolver import ConfigResolver
from hindsight_api.extensions.tenant import TenantExtension
from hindsight_api.models import RequestContext

# Enable bank config API for all tests in this module
os.environ["HINDSIGHT_API_ENABLE_BANK_CONFIG_API"] = "true"


class MockTenantExtension(TenantExtension):
    """Mock tenant extension for testing tenant-level config."""

    def __init__(self, tenant_config: dict):
        self.tenant_config = tenant_config

    async def authenticate(self, context):
        from hindsight_api.extensions.tenant import TenantContext

        return TenantContext(schema_name="public")

    async def list_tenants(self):
        from hindsight_api.extensions.tenant import Tenant

        return [Tenant(schema="public")]

    async def get_tenant_config(self, context):
        """Return mock tenant config."""
        return self.tenant_config


@pytest.mark.asyncio
async def test_config_key_normalization():
    """Test that env var keys are normalized to Python field names."""
    # Test basic normalization
    assert normalize_config_key("HINDSIGHT_API_LLM_PROVIDER") == "llm_provider"
    assert normalize_config_key("HINDSIGHT_API_LLM_MODEL") == "llm_model"
    assert normalize_config_key("HINDSIGHT_API_RETAIN_LLM_PROVIDER") == "retain_llm_provider"

    # Test already normalized keys
    assert normalize_config_key("llm_provider") == "llm_provider"
    assert normalize_config_key("llm_model") == "llm_model"

    # Test dict normalization
    input_dict = {
        "HINDSIGHT_API_LLM_PROVIDER": "openai",
        "HINDSIGHT_API_LLM_MODEL": "gpt-4",
        "llm_base_url": "https://api.openai.com",
    }
    expected = {"llm_provider": "openai", "llm_model": "gpt-4", "llm_base_url": "https://api.openai.com"}
    assert normalize_config_dict(input_dict) == expected


@pytest.mark.asyncio
async def test_hierarchical_fields_categorization():
    """Test that fields are correctly categorized as configurable, credentials, or static."""
    configurable = HindsightConfig.get_configurable_fields()
    credentials = HindsightConfig.get_credential_fields()
    static = HindsightConfig.get_static_fields()

    # Verify no overlap between configurable and credentials
    assert len(configurable & credentials) == 0, "Configurable fields should not include credentials"

    # Verify configurable fields include behavioral settings (safe to modify)
    assert "retain_extraction_mode" in configurable
    assert "enable_observations" in configurable
    assert "retain_chunk_size" in configurable
    assert "retain_custom_instructions" in configurable

    # Verify count is correct (only 4 fields)
    assert len(configurable) == 4

    # Verify credential fields (NEVER exposed)
    assert "llm_api_key" in credentials
    assert "llm_base_url" in credentials
    assert "retain_llm_api_key" in credentials
    assert "reflect_llm_api_key" in credentials

    # Verify static fields include server settings AND non-configurable LLM fields
    assert "database_url" in static
    assert "port" in static
    assert "host" in static
    assert "embeddings_provider" in static
    assert "reranker_provider" in static
    assert "worker_enabled" in static
    assert "llm_provider" in static  # Not configurable (needs presets)
    assert "llm_model" in static  # Not configurable (needs presets)
    assert "graph_retriever" in static  # Performance tuning, not configurable
    assert "llm_max_concurrent" in static  # Performance tuning, not configurable


@pytest.mark.asyncio
async def test_config_hierarchy_resolution(memory, request_context):
    """Test that config resolution follows global → tenant → bank hierarchy."""
    bank_id = "test-hierarchy-bank"

    try:
        # Ensure bank exists in database
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Set up mock tenant extension with tenant-level config (use configurable fields only)
        tenant_config = {"retain_chunk_size": 5000, "retain_extraction_mode": "tenant-mode"}
        mock_tenant = MockTenantExtension(tenant_config)

        # Create config resolver with mock tenant extension
        resolver = ConfigResolver(pool=memory._pool, tenant_extension=mock_tenant)

        # Test 1: Global config only (no overrides)
        context = RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)
        config = await resolver.get_bank_config(bank_id, context)

        # Should have configurable fields from global config (NOT credentials or llm_provider/model)
        assert "retain_chunk_size" in config  # Configurable field
        assert "llm_api_key" not in config  # Credential - never exposed
        assert "llm_provider" not in config  # Not configurable (needs presets)

        # Test 2: Add tenant-level overrides
        config = await resolver.get_bank_config(bank_id, context)

        # Should apply tenant overrides (only configurable fields)
        assert config["retain_chunk_size"] == 5000  # Tenant override
        assert config["retain_extraction_mode"] == "tenant-mode"  # Tenant override

        # Test 3: Add bank-level overrides (should take precedence)
        await resolver.update_bank_config(
            bank_id,
            {"retain_chunk_size": 2000, "retain_extraction_mode": "bank-mode"},  # Override tenant settings
            context,
        )

        # Config should reflect changes immediately (no caching)
        config = await resolver.get_bank_config(bank_id, context)

        # Bank overrides should take precedence over tenant
        assert config["retain_chunk_size"] == 2000  # Bank override wins
        assert config["retain_extraction_mode"] == "bank-mode"  # Bank override wins

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_config_validation_rejects_static_fields(memory, request_context):
    """Test that attempting to override static fields raises ValueError."""
    bank_id = "test-validation-bank"

    try:
        # Ensure bank exists in database
        await memory.get_bank_profile(bank_id, request_context=request_context)

        resolver = ConfigResolver(pool=memory._pool)

        # Test 1: Configurable fields should work
        await resolver.update_bank_config(bank_id, {"retain_chunk_size": 4000, "retain_extraction_mode": "verbose"})

        # Test 2: Static fields should raise ValueError
        with pytest.raises(ValueError, match="Cannot override static"):
            await resolver.update_bank_config(bank_id, {"port": 9000})

        with pytest.raises(ValueError, match="Cannot override static"):
            await resolver.update_bank_config(bank_id, {"database_url": "postgresql://fake"})

        with pytest.raises(ValueError, match="Cannot override static"):
            await resolver.update_bank_config(bank_id, {"embeddings_provider": "openai"})

        # Test 3: Credential fields should raise ValueError
        with pytest.raises(ValueError, match="Cannot set credential fields"):
            await resolver.update_bank_config(bank_id, {"llm_api_key": "sk-fake"})

        # Test 4: Non-configurable LLM fields should raise ValueError (need presets)
        with pytest.raises(ValueError, match="Cannot override static"):
            await resolver.update_bank_config(bank_id, {"llm_model": "gpt-4"})

        # Test 5: Mix of configurable and static should fail
        with pytest.raises(ValueError, match="Cannot override static"):
            await resolver.update_bank_config(bank_id, {"retain_chunk_size": 4000, "port": 9000})

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_config_freshness_across_updates(memory, request_context):
    """Test that config changes are immediately visible (no stale cache)."""
    bank1 = "freshness-test-1"

    try:
        # Ensure bank exists in database
        await memory.get_bank_profile(bank1, request_context=request_context)

        resolver = ConfigResolver(pool=memory._pool)

        # Test 1: Initial config reflects global defaults
        config1 = await resolver.get_bank_config(bank1, None)
        initial_chunk_size = config1["retain_chunk_size"]

        # Test 2: Update config
        await resolver.update_bank_config(bank1, {"retain_chunk_size": 4000})

        # Test 3: Next call should see updated value immediately (no stale cache)
        config2 = await resolver.get_bank_config(bank1, None)
        assert config2["retain_chunk_size"] == 4000

        # Test 4: Multiple updates are all immediately visible
        await resolver.update_bank_config(bank1, {"retain_chunk_size": 4500})
        config3 = await resolver.get_bank_config(bank1, None)
        assert config3["retain_chunk_size"] == 4500

        # Test 5: Reset restores global defaults immediately
        await resolver.reset_bank_config(bank1)
        config4 = await resolver.get_bank_config(bank1, None)
        assert config4["retain_chunk_size"] == initial_chunk_size  # Back to global default

        # Test 6: Each call returns a fresh config dict (not a cached reference)
        config5 = await resolver.get_bank_config(bank1, None)
        config6 = await resolver.get_bank_config(bank1, None)
        assert config5 is not config6  # Different object instances

    finally:
        await memory.delete_bank(bank1, request_context=request_context)


@pytest.mark.asyncio
async def test_config_reset_to_defaults(memory, request_context):
    """Test that resetting config removes all bank-specific overrides."""
    bank_id = "test-reset-bank"

    try:
        # Ensure bank exists in database
        await memory.get_bank_profile(bank_id, request_context=request_context)

        resolver = ConfigResolver(pool=memory._pool)

        # Add bank-specific overrides
        await resolver.update_bank_config(
            bank_id,
            {
                "retain_chunk_size": 5500,
                "retain_extraction_mode": "custom",
                "retain_custom_instructions": "Custom instructions",
            },
        )

        # Verify overrides applied
        config = await resolver.get_bank_config(bank_id, None)
        assert config["retain_chunk_size"] == 5500
        assert config["retain_extraction_mode"] == "custom"
        assert config["retain_custom_instructions"] == "Custom instructions"

        # Reset to defaults
        await resolver.reset_bank_config(bank_id)

        # Verify overrides removed (back to global defaults)
        config_reset = await resolver.get_bank_config(bank_id, None)
        assert config_reset["retain_chunk_size"] != 5500  # Should be global default
        assert config_reset["retain_extraction_mode"] != "custom"  # Should be global default

        # Verify bank_config is empty
        bank_overrides = await resolver._load_bank_config(bank_id)
        assert bank_overrides == {}

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_config_supports_both_key_formats(memory, request_context):
    """Test that API accepts both env var and Python field formats."""
    bank_id = "test-key-format-bank"

    try:
        # Ensure bank exists in database
        await memory.get_bank_profile(bank_id, request_context=request_context)

        resolver = ConfigResolver(pool=memory._pool)

        # Test 1: Python field format
        await resolver.update_bank_config(bank_id, {"retain_chunk_size": 7000})

        config = await resolver.get_bank_config(bank_id, None)
        assert config["retain_chunk_size"] == 7000

        # Test 2: Env var format (should be normalized)
        await resolver.update_bank_config(bank_id, {"HINDSIGHT_API_RETAIN_CHUNK_SIZE": 8000})

        config = await resolver.get_bank_config(bank_id, None)
        assert config["retain_chunk_size"] == 8000

        # Test 3: Mixed format in same request
        await resolver.update_bank_config(
            bank_id,
            {
                "retain_chunk_size": 9000,  # Python format
                "HINDSIGHT_API_RETAIN_EXTRACTION_MODE": "verbose",  # Env format
            },
        )

        config = await resolver.get_bank_config(bank_id, None)
        assert config["retain_chunk_size"] == 9000
        assert config["retain_extraction_mode"] == "verbose"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_config_only_configurable_fields_stored(memory, request_context):
    """Test that only configurable fields are stored in bank config."""
    bank_id = "test-filter-bank"

    try:
        # Ensure bank exists in database
        await memory.get_bank_profile(bank_id, request_context=request_context)

        resolver = ConfigResolver(pool=memory._pool)

        # Add valid configurable field
        await resolver.update_bank_config(bank_id, {"retain_chunk_size": 3500})

        # Load bank config and verify only configurable fields present
        bank_overrides = await resolver._load_bank_config(bank_id)

        for key in bank_overrides.keys():
            assert key in HindsightConfig.get_configurable_fields(), f"Non-configurable field {key} in bank config"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_config_get_bank_config_no_static_or_credential_fields_leak(memory, request_context):
    """
    SECURITY TEST: Verify get_bank_config() only returns configurable fields (no static/credentials).

    This prevents leaking sensitive system configuration like database URLs,
    API keys, LLM providers/models, worker counts, etc. when retrieving bank configuration.
    """
    bank_id = "test-security-bank"

    try:
        # Ensure bank exists in database
        await memory.get_bank_profile(bank_id, request_context=request_context)

        resolver = ConfigResolver(pool=memory._pool)

        # Get bank config
        config = await resolver.get_bank_config(bank_id, None)

        # Get field categorizations
        configurable_fields = HindsightConfig.get_configurable_fields()
        credential_fields = HindsightConfig.get_credential_fields()
        static_fields = HindsightConfig.get_static_fields()

        # SECURITY: Verify ONLY configurable fields are returned (NO static, NO credentials)
        for key in config.keys():
            assert key in configurable_fields, (
                f"SECURITY VIOLATION: Non-configurable field '{key}' returned by get_bank_config(). "
                f"Only configurable fields should be returned to prevent leaking system config."
            )
            assert key not in credential_fields, (
                f"SECURITY VIOLATION: Credential field '{key}' returned by get_bank_config(). "
                f"Credentials must NEVER be exposed via API."
            )

        # SECURITY: Verify specific sensitive fields are NOT present
        sensitive_fields = [
            "database_url", "api_port", "host", "worker_count",  # Infrastructure
            "llm_api_key", "llm_base_url",  # Credentials
            "retain_llm_api_key", "reflect_llm_api_key",  # More credentials
            "llm_provider", "llm_model",  # Not configurable (need presets)
        ]
        for field in sensitive_fields:
            assert field not in config, (
                f"SECURITY VIOLATION: Sensitive field '{field}' returned by get_bank_config(). "
                f"Must not be exposed via bank config API."
            )

        # Verify we have the expected configurable fields (small set)
        expected_configurable = ["retain_chunk_size", "retain_extraction_mode", "enable_observations"]
        for field in expected_configurable:
            assert field in config, f"Expected configurable field '{field}' missing from config"

        # Should have a small number of configurable fields (not hundreds)
        assert len(config) < 20, f"Too many fields returned: {len(config)}"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_config_permissions_system(memory, request_context):
    """
    Test that tenant extension can control which fields banks are allowed to modify.

    Tests get_allowed_config_fields() permission system.
    """
    bank_id = "test-permissions-bank"

    class PermissionTenantExtension(TenantExtension):
        """Mock tenant extension with configurable permissions."""

        def __init__(self, allowed_fields: set[str] | None):
            self.allowed_fields = allowed_fields

        async def authenticate(self, context):
            from hindsight_api.extensions.tenant import TenantContext

            return TenantContext(schema_name="public")

        async def list_tenants(self):
            from hindsight_api.extensions.tenant import Tenant

            return [Tenant(schema="public")]

        async def get_allowed_config_fields(self, context, bank_id):
            """Return configured allowed fields."""
            return self.allowed_fields

    try:
        # Ensure bank exists in database
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Test 1: None = allow all configurable fields
        extension = PermissionTenantExtension(allowed_fields=None)
        resolver = ConfigResolver(pool=memory._pool, tenant_extension=extension)

        await resolver.update_bank_config(
            bank_id, {"retain_chunk_size": 4000, "retain_extraction_mode": "verbose"}, request_context
        )
        config = await resolver.get_bank_config(bank_id, request_context)
        assert config["retain_chunk_size"] == 4000
        assert config["retain_extraction_mode"] == "verbose"

        # Reset for next test
        await resolver.reset_bank_config(bank_id)

        # Test 2: Specific set = only those fields allowed
        extension = PermissionTenantExtension(allowed_fields={"retain_chunk_size"})
        resolver = ConfigResolver(pool=memory._pool, tenant_extension=extension)

        # Should allow retain_chunk_size
        await resolver.update_bank_config(bank_id, {"retain_chunk_size": 5000}, request_context)
        config = await resolver.get_bank_config(bank_id, request_context)
        assert config["retain_chunk_size"] == 5000

        # Should reject retain_extraction_mode (not in allowed list)
        with pytest.raises(ValueError, match="Not allowed to modify fields"):
            await resolver.update_bank_config(bank_id, {"retain_extraction_mode": "verbose"}, request_context)

        # Should reject mix of allowed and disallowed
        with pytest.raises(ValueError, match="Not allowed to modify fields"):
            await resolver.update_bank_config(
                bank_id, {"retain_chunk_size": 6000, "retain_extraction_mode": "verbose"}, request_context
            )

        # Reset for next test
        await resolver.reset_bank_config(bank_id)

        # Test 3: Empty set = no modifications allowed (read-only)
        extension = PermissionTenantExtension(allowed_fields=set())
        resolver = ConfigResolver(pool=memory._pool, tenant_extension=extension)

        with pytest.raises(ValueError, match="Not allowed to modify fields"):
            await resolver.update_bank_config(bank_id, {"retain_chunk_size": 7000}, request_context)

        # Test 4: get_bank_config should filter response based on permissions
        extension = PermissionTenantExtension(allowed_fields={"retain_chunk_size", "enable_observations"})
        resolver = ConfigResolver(pool=memory._pool, tenant_extension=extension)

        config = await resolver.get_bank_config(bank_id, request_context)

        # Should only return allowed fields
        assert "retain_chunk_size" in config
        assert "enable_observations" in config
        # Other configurable fields should be filtered out
        assert "retain_extraction_mode" not in config
        assert "retain_custom_instructions" not in config

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

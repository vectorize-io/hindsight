"""
Tests for hierarchical configuration system.

Tests config resolution hierarchy (global → tenant → bank),
key normalization, API endpoints, validation, and caching.
"""

import pytest
from hindsight_api import MemoryEngine
from hindsight_api.config import HindsightConfig, normalize_config_key, normalize_config_dict
from hindsight_api.config_resolver import ConfigResolver, _CONFIG_CACHE
from hindsight_api.extensions.tenant import TenantExtension
from hindsight_api.models import RequestContext


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
    """Test that fields are correctly categorized as hierarchical or static."""
    hierarchical = HindsightConfig.get_hierarchical_fields()
    static = HindsightConfig.get_static_fields()

    # Verify no overlap
    assert len(hierarchical & static) == 0

    # Verify hierarchical fields include LLM settings
    assert "llm_provider" in hierarchical
    assert "llm_model" in hierarchical
    assert "retain_llm_provider" in hierarchical
    assert "reflect_llm_model" in hierarchical
    assert "retain_extraction_mode" in hierarchical
    assert "graph_retriever" in hierarchical
    assert "enable_observations" in hierarchical

    # Verify static fields include server settings
    assert "database_url" in static
    assert "port" in static
    assert "host" in static
    assert "embeddings_provider" in static
    assert "reranker_provider" in static
    assert "worker_enabled" in static


@pytest.mark.asyncio
async def test_config_hierarchy_resolution(memory, request_context):
    """Test that config resolution follows global → tenant → bank hierarchy."""
    bank_id = "test-hierarchy-bank"

    try:
        # Ensure bank exists in database
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Set up mock tenant extension with tenant-level config
        tenant_config = {"llm_model": "tenant-model", "retain_extraction_mode": "tenant-mode"}
        mock_tenant = MockTenantExtension(tenant_config)

        # Create config resolver with mock tenant extension
        resolver = ConfigResolver(pool=memory._pool, tenant_extension=mock_tenant)

        # Test 1: Global config only (no overrides)
        context = RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)
        config = await resolver.get_bank_config(bank_id, context)

        # Should have global defaults
        assert "llm_provider" in config  # From global config

        # Test 2: Add tenant-level overrides
        config = await resolver.get_bank_config(bank_id, context)

        # Should apply tenant overrides
        assert config["llm_model"] == "tenant-model"
        assert config["retain_extraction_mode"] == "tenant-mode"

        # Test 3: Add bank-level overrides (should take precedence)
        await resolver.update_bank_config(
            bank_id, {"llm_model": "bank-model", "retain_chunk_size": 2000}  # Override tenant setting  # Bank-only setting
        )

        # Clear cache to force reload
        resolver.invalidate_cache(bank_id)
        config = await resolver.get_bank_config(bank_id, context)

        # Bank overrides should take precedence over tenant
        assert config["llm_model"] == "bank-model"  # Bank override wins
        assert config["retain_extraction_mode"] == "tenant-mode"  # Tenant override (no bank override)
        assert config["retain_chunk_size"] == 2000  # Bank-only setting

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

        # Test 1: Hierarchical fields should work
        await resolver.update_bank_config(bank_id, {"llm_model": "gpt-4", "retain_extraction_mode": "verbose"})

        # Test 2: Static fields should raise ValueError
        with pytest.raises(ValueError, match="Cannot override static"):
            await resolver.update_bank_config(bank_id, {"port": 9000})

        with pytest.raises(ValueError, match="Cannot override static"):
            await resolver.update_bank_config(bank_id, {"database_url": "postgresql://fake"})

        with pytest.raises(ValueError, match="Cannot override static"):
            await resolver.update_bank_config(bank_id, {"embeddings_provider": "openai"})

        # Test 3: Mix of hierarchical and static should fail
        with pytest.raises(ValueError, match="Cannot override static"):
            await resolver.update_bank_config(bank_id, {"llm_model": "gpt-4", "port": 9000})

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_config_lru_cache_behavior(memory, request_context):
    """Test LRU cache hits, misses, and eviction."""
    # Clear cache before test
    _CONFIG_CACHE.clear()

    # Create test banks
    bank1 = "cache-test-1"
    bank2 = "cache-test-2"

    try:
        # Ensure banks exist in database
        await memory.get_bank_profile(bank1, request_context=request_context)
        await memory.get_bank_profile(bank2, request_context=request_context)

        resolver = ConfigResolver(pool=memory._pool)

        # Test 1: Cache miss on first access
        assert bank1 not in _CONFIG_CACHE
        config1 = await resolver.get_bank_config(bank1, None)
        assert bank1 in _CONFIG_CACHE
        assert config1 == _CONFIG_CACHE[bank1]

        # Test 2: Cache hit on second access
        config1_cached = await resolver.get_bank_config(bank1, None)
        assert config1_cached == config1
        assert config1_cached is _CONFIG_CACHE[bank1]  # Same object (cache hit)

        # Test 3: Different bank is separate cache entry
        config2 = await resolver.get_bank_config(bank2, None)
        assert bank2 in _CONFIG_CACHE
        assert bank1 in _CONFIG_CACHE
        # Configs will be equal if no overrides, but should be separate cache entries
        assert config2 is not config1  # Different object instances

        # Test 4: Cache invalidation on update
        await resolver.update_bank_config(bank1, {"llm_model": "updated-model"})
        assert bank1 not in _CONFIG_CACHE  # Should be evicted

        # Next access will be cache miss
        config1_updated = await resolver.get_bank_config(bank1, None)
        assert bank1 in _CONFIG_CACHE
        assert config1_updated["llm_model"] == "updated-model"

        # Test 5: Reset clears overrides and invalidates cache
        await resolver.reset_bank_config(bank1)
        assert bank1 not in _CONFIG_CACHE

    finally:
        await memory.delete_bank(bank1, request_context=request_context)
        await memory.delete_bank(bank2, request_context=request_context)


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
            bank_id, {"llm_model": "custom-model", "retain_extraction_mode": "verbose", "retain_chunk_size": 3000}
        )

        # Verify overrides applied
        config = await resolver.get_bank_config(bank_id, None)
        assert config["llm_model"] == "custom-model"
        assert config["retain_extraction_mode"] == "verbose"
        assert config["retain_chunk_size"] == 3000

        # Reset to defaults
        await resolver.reset_bank_config(bank_id)

        # Verify overrides removed (back to global defaults)
        config_reset = await resolver.get_bank_config(bank_id, None)
        assert config_reset["llm_model"] != "custom-model"  # Should be global default
        assert config_reset["retain_extraction_mode"] != "verbose"  # Should be global default

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
        await resolver.update_bank_config(bank_id, {"llm_model": "field-format-model"})

        config = await resolver.get_bank_config(bank_id, None)
        assert config["llm_model"] == "field-format-model"

        # Test 2: Env var format (should be normalized)
        await resolver.update_bank_config(bank_id, {"HINDSIGHT_API_LLM_MODEL": "env-format-model"})

        resolver.invalidate_cache(bank_id)
        config = await resolver.get_bank_config(bank_id, None)
        assert config["llm_model"] == "env-format-model"

        # Test 3: Mixed format in same request
        await resolver.update_bank_config(
            bank_id,
            {
                "llm_model": "mixed-1",  # Python format
                "HINDSIGHT_API_RETAIN_EXTRACTION_MODE": "verbose",  # Env format
            },
        )

        resolver.invalidate_cache(bank_id)
        config = await resolver.get_bank_config(bank_id, None)
        assert config["llm_model"] == "mixed-1"
        assert config["retain_extraction_mode"] == "verbose"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_config_only_hierarchical_fields_stored(memory, request_context):
    """Test that only hierarchical fields are stored in bank config."""
    bank_id = "test-filter-bank"

    try:
        # Ensure bank exists in database
        await memory.get_bank_profile(bank_id, request_context=request_context)

        resolver = ConfigResolver(pool=memory._pool)

        # Add valid hierarchical field
        await resolver.update_bank_config(bank_id, {"llm_model": "test-model"})

        # Load bank config and verify only hierarchical fields present
        bank_overrides = await resolver._load_bank_config(bank_id)

        for key in bank_overrides.keys():
            assert key in HindsightConfig.get_hierarchical_fields(), f"Non-hierarchical field {key} in bank config"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)

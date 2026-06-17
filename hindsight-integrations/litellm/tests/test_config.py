"""Unit tests for hindsight_litellm configuration and defaults."""

import os
from dataclasses import fields
from unittest.mock import MagicMock, patch


from hindsight_litellm import configure, wrap_openai, wrap_anthropic
from hindsight_litellm.config import (
    DEFAULT_HINDSIGHT_API_URL,
    DEFAULT_BANK_ID,
    HINDSIGHT_API_KEY_ENV,
    reset_config,
    is_configured,
)
from hindsight_litellm.wrappers import (
    HindsightOpenAI,
    HindsightAnthropic,
    HindsightCallSettings,
    _merge_settings,
)


class TestDefaults:
    """Test default configuration values."""

    def test_default_api_url(self):
        """Test default API URL is production."""
        assert DEFAULT_HINDSIGHT_API_URL == "https://api.hindsight.vectorize.io"

    def test_default_bank_id(self):
        """Test default bank ID is 'default'."""
        assert DEFAULT_BANK_ID == "default"

    def test_env_var_name(self):
        """Test environment variable name for API key."""
        assert HINDSIGHT_API_KEY_ENV == "HINDSIGHT_API_KEY"


class TestConfigure:
    """Test configure() function."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_configure_with_no_arguments(self):
        """Test configure() with no arguments uses production URL and leaves bank_id unset."""
        config = configure()

        assert config.hindsight_api_url == DEFAULT_HINDSIGHT_API_URL
        assert config.bank_id is None

    def test_configure_reads_api_key_from_env(self):
        """Test configure() reads API key from environment variable."""
        with patch.dict(os.environ, {HINDSIGHT_API_KEY_ENV: "test-api-key-123"}):
            config = configure()

        assert config.api_key == "test-api-key-123"

    def test_configure_explicit_api_key_overrides_env(self):
        """Test explicit api_key parameter overrides environment variable."""
        with patch.dict(os.environ, {HINDSIGHT_API_KEY_ENV: "env-key"}):
            config = configure(api_key="explicit-key")

        assert config.api_key == "explicit-key"

    def test_configure_explicit_values_override_defaults(self):
        """Test explicit values override defaults."""
        config = configure(
            hindsight_api_url="http://custom-url:8888",
            bank_id="custom-bank",
            api_key="custom-key",
        )

        assert config.hindsight_api_url == "http://custom-url:8888"
        assert config.bank_id == "custom-bank"
        assert config.api_key == "custom-key"

    def test_is_configured_false_without_explicit_bank_id(self):
        """Test is_configured() returns False when configure() is called without bank_id."""
        configure()
        assert is_configured() is False

    def test_is_configured_false_when_not_configured(self):
        """Test is_configured() returns False when not configured."""
        reset_config()
        assert is_configured() is False


class TestWrapOpenAI:
    """Test wrap_openai() function."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_wrap_openai_with_only_client(self):
        """Test wrap_openai() works with only the client argument."""
        mock_client = MagicMock()

        with patch.dict(os.environ, {HINDSIGHT_API_KEY_ENV: "test-key"}):
            wrapped = wrap_openai(mock_client)

        assert isinstance(wrapped, HindsightOpenAI)
        assert wrapped._default_settings.bank_id == DEFAULT_BANK_ID
        assert wrapped._api_url == DEFAULT_HINDSIGHT_API_URL
        assert wrapped._api_key == "test-key"

    def test_wrap_openai_uses_defaults(self):
        """Test wrap_openai() uses default values."""
        mock_client = MagicMock()

        wrapped = wrap_openai(mock_client)

        assert wrapped._default_settings.bank_id == DEFAULT_BANK_ID
        assert wrapped._api_url == DEFAULT_HINDSIGHT_API_URL

    def test_wrap_openai_reads_api_key_from_env(self):
        """Test wrap_openai() reads API key from environment."""
        mock_client = MagicMock()

        with patch.dict(os.environ, {HINDSIGHT_API_KEY_ENV: "env-api-key"}):
            wrapped = wrap_openai(mock_client)

        assert wrapped._api_key == "env-api-key"

    def test_wrap_openai_explicit_overrides_defaults(self):
        """Test wrap_openai() explicit values override defaults."""
        mock_client = MagicMock()

        wrapped = wrap_openai(
            mock_client,
            bank_id="my-bank",
            hindsight_api_url="http://localhost:9999",
            api_key="my-key",
        )

        assert wrapped._default_settings.bank_id == "my-bank"
        assert wrapped._api_url == "http://localhost:9999"
        assert wrapped._api_key == "my-key"

    def test_wrap_openai_all_settings_kwargs(self):
        """Test wrap_openai() accepts all HindsightCallSettings fields as kwargs."""
        mock_client = MagicMock()

        # Create kwargs with all settings
        settings_kwargs = {
            "bank_id": "test-bank",
            "document_id": "test-doc",
            "session_id": "test-session",
            "store_conversations": False,
            "inject_memories": False,
            "budget": "high",
            "fact_types": ["world", "experience"],
            "max_memories": 10,
            "max_memory_tokens": 2048,
            "include_entities": False,
            "trace": True,
            "tags": ["user:alice", "session:123"],
            "recall_tags": ["user:alice"],
            "recall_tags_match": "all",
            "use_reflect": True,
            "reflect_context": "test context",
            "reflect_response_schema": {"type": "object"},
            "reflect_include_facts": True,
            "query": "test query",
            "verbose": True,
        }

        wrapped = wrap_openai(mock_client, **settings_kwargs)

        # Verify all settings were applied
        for field_name, expected_value in settings_kwargs.items():
            actual_value = getattr(wrapped._default_settings, field_name)
            assert actual_value == expected_value, f"Field {field_name}: expected {expected_value}, got {actual_value}"


class TestWrapAnthropic:
    """Test wrap_anthropic() function."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_wrap_anthropic_with_only_client(self):
        """Test wrap_anthropic() works with only the client argument."""
        mock_client = MagicMock()

        with patch.dict(os.environ, {HINDSIGHT_API_KEY_ENV: "test-key"}):
            wrapped = wrap_anthropic(mock_client)

        assert isinstance(wrapped, HindsightAnthropic)
        assert wrapped._default_settings.bank_id == DEFAULT_BANK_ID
        assert wrapped._api_url == DEFAULT_HINDSIGHT_API_URL
        assert wrapped._api_key == "test-key"

    def test_wrap_anthropic_uses_defaults(self):
        """Test wrap_anthropic() uses default values."""
        mock_client = MagicMock()

        wrapped = wrap_anthropic(mock_client)

        assert wrapped._default_settings.bank_id == DEFAULT_BANK_ID
        assert wrapped._api_url == DEFAULT_HINDSIGHT_API_URL

    def test_wrap_anthropic_reads_api_key_from_env(self):
        """Test wrap_anthropic() reads API key from environment."""
        mock_client = MagicMock()

        with patch.dict(os.environ, {HINDSIGHT_API_KEY_ENV: "env-api-key"}):
            wrapped = wrap_anthropic(mock_client)

        assert wrapped._api_key == "env-api-key"

    def test_wrap_anthropic_explicit_overrides_defaults(self):
        """Test wrap_anthropic() explicit values override defaults."""
        mock_client = MagicMock()

        wrapped = wrap_anthropic(
            mock_client,
            bank_id="my-bank",
            hindsight_api_url="http://localhost:9999",
            api_key="my-key",
        )

        assert wrapped._default_settings.bank_id == "my-bank"
        assert wrapped._api_url == "http://localhost:9999"
        assert wrapped._api_key == "my-key"

    def test_wrap_anthropic_all_settings_kwargs(self):
        """Test wrap_anthropic() accepts all HindsightCallSettings fields as kwargs."""
        mock_client = MagicMock()

        # Create kwargs with all settings
        settings_kwargs = {
            "bank_id": "test-bank",
            "document_id": "test-doc",
            "session_id": "test-session",
            "store_conversations": False,
            "inject_memories": False,
            "budget": "high",
            "fact_types": ["world", "experience"],
            "max_memories": 10,
            "max_memory_tokens": 2048,
            "include_entities": False,
            "trace": True,
            "tags": ["user:alice", "session:123"],
            "recall_tags": ["user:alice"],
            "recall_tags_match": "all",
            "use_reflect": True,
            "reflect_context": "test context",
            "reflect_response_schema": {"type": "object"},
            "reflect_include_facts": True,
            "query": "test query",
            "verbose": True,
        }

        wrapped = wrap_anthropic(mock_client, **settings_kwargs)

        # Verify all settings were applied
        for field_name, expected_value in settings_kwargs.items():
            actual_value = getattr(wrapped._default_settings, field_name)
            assert actual_value == expected_value, f"Field {field_name}: expected {expected_value}, got {actual_value}"


class TestMergeSettings:
    """Test _merge_settings() function for per-call overrides."""

    def test_merge_settings_no_overrides(self):
        """Test _merge_settings() returns defaults when no overrides provided."""
        defaults = HindsightCallSettings(bank_id="default-bank", budget="mid")
        kwargs = {"model": "gpt-4", "messages": []}  # No hindsight_* kwargs

        merged = _merge_settings(defaults, kwargs)

        assert merged.bank_id == "default-bank"
        assert merged.budget == "mid"

    def test_merge_settings_with_overrides(self):
        """Test _merge_settings() applies hindsight_* overrides."""
        defaults = HindsightCallSettings(bank_id="default-bank", budget="mid")
        kwargs = {
            "model": "gpt-4",
            "hindsight_bank_id": "override-bank",
            "hindsight_budget": "high",
        }

        merged = _merge_settings(defaults, kwargs)

        assert merged.bank_id == "override-bank"
        assert merged.budget == "high"

    def test_merge_settings_partial_override(self):
        """Test _merge_settings() only overrides specified fields."""
        defaults = HindsightCallSettings(
            bank_id="default-bank",
            budget="mid",
            verbose=True,
            max_memories=5,
        )
        kwargs = {
            "hindsight_bank_id": "override-bank",
            # budget, verbose, max_memories not overridden
        }

        merged = _merge_settings(defaults, kwargs)

        assert merged.bank_id == "override-bank"  # Overridden
        assert merged.budget == "mid"  # Default preserved
        assert merged.verbose is True  # Default preserved
        assert merged.max_memories == 5  # Default preserved

    def test_merge_settings_ignores_invalid_fields(self):
        """Test _merge_settings() ignores hindsight_* kwargs for non-existent fields."""
        defaults = HindsightCallSettings(bank_id="default-bank")
        kwargs = {
            "hindsight_bank_id": "valid-override",
            "hindsight_nonexistent_field": "should-be-ignored",
        }

        merged = _merge_settings(defaults, kwargs)

        assert merged.bank_id == "valid-override"
        assert not hasattr(merged, "nonexistent_field")

    def test_merge_settings_all_fields(self):
        """Test _merge_settings() works with all HindsightCallSettings fields."""
        # Create defaults with non-default values
        defaults = HindsightCallSettings(
            bank_id="default-bank",
            document_id="default-doc",
            session_id="default-session",
            store_conversations=True,
            inject_memories=True,
            budget="mid",
            fact_types=None,
            max_memories=None,
            max_memory_tokens=4096,
            include_entities=True,
            trace=False,
            tags=None,
            recall_tags=None,
            recall_tags_match="any",
            use_reflect=False,
            reflect_context=None,
            reflect_response_schema=None,
            reflect_include_facts=False,
            query=None,
            verbose=False,
        )

        # Override every field
        override_values = {
            "hindsight_bank_id": "override-bank",
            "hindsight_document_id": "override-doc",
            "hindsight_session_id": "override-session",
            "hindsight_store_conversations": False,
            "hindsight_inject_memories": False,
            "hindsight_budget": "high",
            "hindsight_fact_types": ["world", "experience"],
            "hindsight_max_memories": 10,
            "hindsight_max_memory_tokens": 2048,
            "hindsight_include_entities": False,
            "hindsight_trace": True,
            "hindsight_tags": ["user:bob", "org:acme"],
            "hindsight_recall_tags": ["user:bob"],
            "hindsight_recall_tags_match": "all",
            "hindsight_use_reflect": True,
            "hindsight_reflect_context": "test context",
            "hindsight_reflect_response_schema": {"type": "object"},
            "hindsight_reflect_include_facts": True,
            "hindsight_query": "test query",
            "hindsight_verbose": True,
        }

        merged = _merge_settings(defaults, override_values)

        # Verify all fields were overridden
        assert merged.bank_id == "override-bank"
        assert merged.document_id == "override-doc"
        assert merged.session_id == "override-session"
        assert merged.store_conversations is False
        assert merged.inject_memories is False
        assert merged.budget == "high"
        assert merged.fact_types == ["world", "experience"]
        assert merged.max_memories == 10
        assert merged.max_memory_tokens == 2048
        assert merged.include_entities is False
        assert merged.trace is True
        assert merged.tags == ["user:bob", "org:acme"]
        assert merged.recall_tags == ["user:bob"]
        assert merged.recall_tags_match == "all"
        assert merged.use_reflect is True
        assert merged.reflect_context == "test context"
        assert merged.reflect_response_schema == {"type": "object"}
        assert merged.reflect_include_facts is True
        assert merged.query == "test query"
        assert merged.verbose is True


class TestHindsightCallSettingsConsistency:
    """Test that HindsightCallSettings works consistently across wrappers."""

    def test_same_settings_for_openai_and_anthropic(self):
        """Test that OpenAI and Anthropic wrappers use the same HindsightCallSettings."""
        mock_openai_client = MagicMock()
        mock_anthropic_client = MagicMock()

        settings_kwargs = {
            "bank_id": "shared-bank",
            "budget": "high",
            "use_reflect": True,
            "verbose": True,
        }

        openai_wrapped = wrap_openai(mock_openai_client, **settings_kwargs)
        anthropic_wrapped = wrap_anthropic(mock_anthropic_client, **settings_kwargs)

        # Both should have identical settings
        assert openai_wrapped._default_settings.bank_id == anthropic_wrapped._default_settings.bank_id
        assert openai_wrapped._default_settings.budget == anthropic_wrapped._default_settings.budget
        assert openai_wrapped._default_settings.use_reflect == anthropic_wrapped._default_settings.use_reflect
        assert openai_wrapped._default_settings.verbose == anthropic_wrapped._default_settings.verbose

    def test_new_field_works_for_both_wrappers(self):
        """Test that all HindsightCallSettings fields work for both wrappers."""
        mock_openai_client = MagicMock()
        mock_anthropic_client = MagicMock()

        # Get all fields from the dataclass
        all_field_names = [f.name for f in fields(HindsightCallSettings)]

        # Verify both wrappers can access all fields
        openai_wrapped = wrap_openai(mock_openai_client)
        anthropic_wrapped = wrap_anthropic(mock_anthropic_client)

        for field_name in all_field_names:
            # Both should have the field accessible
            assert hasattr(openai_wrapped._default_settings, field_name), f"OpenAI wrapper missing field: {field_name}"
            assert hasattr(anthropic_wrapped._default_settings, field_name), (
                f"Anthropic wrapper missing field: {field_name}"
            )

"""
Tests for per-operation LLM configuration.

Verifies that retain and reflect operations use their respective LLM configs.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up environment for each test, restoring original values after."""
    from hindsight_api.config import clear_config_cache

    # Save original environment values
    env_vars_to_set = {
        "HINDSIGHT_API_SKIP_LLM_VERIFICATION": "true",
        "HINDSIGHT_API_LAZY_RERANKER": "true",
        "HINDSIGHT_API_LLM_PROVIDER": "mock",
        "HINDSIGHT_API_LLM_MODEL": "default-model",
        "HINDSIGHT_API_RETAIN_LLM_PROVIDER": "mock",
        "HINDSIGHT_API_RETAIN_LLM_MODEL": "retain-model",
        "HINDSIGHT_API_REFLECT_LLM_PROVIDER": "mock",
        "HINDSIGHT_API_REFLECT_LLM_MODEL": "reflect-model",
    }

    # Save original values
    original_values = {}
    for key in env_vars_to_set:
        original_values[key] = os.environ.get(key)

    # Set test values
    for key, value in env_vars_to_set.items():
        os.environ[key] = value

    clear_config_cache()

    yield

    # Restore original environment
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

    clear_config_cache()


class TestPerOperationLLMConfig:
    """Test that per-operation LLM configs are correctly applied."""

    def test_config_loads_per_operation_settings(self):
        """Test that config correctly loads per-operation LLM settings."""
        from hindsight_api.config import get_config

        config = get_config()

        # Default config
        assert config.llm_provider == "mock"
        assert config.llm_model == "default-model"

        # Retain config
        assert config.retain_llm_provider == "mock"
        assert config.retain_llm_model == "retain-model"

        # Reflect config
        assert config.reflect_llm_provider == "mock"
        assert config.reflect_llm_model == "reflect-model"

    def test_memory_engine_creates_separate_llm_configs(self):
        """Test that MemoryEngine creates separate LLM configs for each operation."""
        from hindsight_api import MemoryEngine

        engine = MemoryEngine(
            skip_llm_verification=True,
            lazy_reranker=True,
        )

        # Verify default config
        assert engine._llm_config.provider == "mock"
        assert engine._llm_config.model == "default-model"

        # Verify retain config
        assert engine._retain_llm_config.provider == "mock"
        assert engine._retain_llm_config.model == "retain-model"

        # Verify reflect config
        assert engine._reflect_llm_config.provider == "mock"
        assert engine._reflect_llm_config.model == "reflect-model"

    def test_memory_engine_with_explicit_params(self):
        """Test that explicit params override env config."""
        from hindsight_api import MemoryEngine

        engine = MemoryEngine(
            memory_llm_provider="mock",
            memory_llm_model="explicit-default",
            retain_llm_provider="mock",
            retain_llm_model="explicit-retain",
            reflect_llm_provider="mock",
            reflect_llm_model="explicit-reflect",
            skip_llm_verification=True,
            lazy_reranker=True,
        )

        assert engine._llm_config.model == "explicit-default"
        assert engine._retain_llm_config.model == "explicit-retain"
        assert engine._reflect_llm_config.model == "explicit-reflect"

    def test_memory_engine_fallback_when_no_per_operation_config(self):
        """Test that per-operation configs fall back to default when not set."""
        from hindsight_api.config import clear_config_cache as clear_cache

        # Temporarily clear per-operation env vars
        retain_provider = os.environ.pop("HINDSIGHT_API_RETAIN_LLM_PROVIDER", None)
        retain_model = os.environ.pop("HINDSIGHT_API_RETAIN_LLM_MODEL", None)
        reflect_provider = os.environ.pop("HINDSIGHT_API_REFLECT_LLM_PROVIDER", None)
        reflect_model = os.environ.pop("HINDSIGHT_API_REFLECT_LLM_MODEL", None)

        try:
            clear_cache()
            from hindsight_api import MemoryEngine

            engine = MemoryEngine(
                skip_llm_verification=True,
                lazy_reranker=True,
            )

            # All should fall back to default
            assert engine._llm_config.model == "default-model"
            assert engine._retain_llm_config.model == "default-model"
            assert engine._reflect_llm_config.model == "default-model"
        finally:
            # Restore env vars
            if retain_provider:
                os.environ["HINDSIGHT_API_RETAIN_LLM_PROVIDER"] = retain_provider
            if retain_model:
                os.environ["HINDSIGHT_API_RETAIN_LLM_MODEL"] = retain_model
            if reflect_provider:
                os.environ["HINDSIGHT_API_REFLECT_LLM_PROVIDER"] = reflect_provider
            if reflect_model:
                os.environ["HINDSIGHT_API_REFLECT_LLM_MODEL"] = reflect_model
            clear_cache()


class TestMockLLMProvider:
    """Test the mock LLM provider functionality."""

    def test_mock_provider_records_calls(self):
        """Test that mock provider records calls."""
        from hindsight_api.engine.llm_wrapper import LLMProvider

        provider = LLMProvider(
            provider="mock",
            api_key="",
            base_url="",
            model="test-model",
        )

        import asyncio

        async def make_call():
            return await provider.call(
                messages=[{"role": "user", "content": "test"}],
                scope="test_scope",
            )

        result = asyncio.get_event_loop().run_until_complete(make_call())

        # Verify call was recorded
        calls = provider.get_mock_calls()
        assert len(calls) == 1
        assert calls[0]["model"] == "test-model"
        assert calls[0]["scope"] == "test_scope"
        assert calls[0]["messages"] == [{"role": "user", "content": "test"}]

    def test_mock_provider_returns_custom_response(self):
        """Test that mock provider can return custom responses."""
        from hindsight_api.engine.llm_wrapper import LLMProvider

        provider = LLMProvider(
            provider="mock",
            api_key="",
            base_url="",
            model="test-model",
        )

        provider.set_mock_response({"custom": "response"})

        import asyncio

        async def make_call():
            return await provider.call(
                messages=[{"role": "user", "content": "test"}],
            )

        result = asyncio.get_event_loop().run_until_complete(make_call())
        assert result == {"custom": "response"}

    def test_mock_provider_returns_usage_when_requested(self):
        """Test that mock provider returns token usage."""
        from hindsight_api.engine.llm_wrapper import LLMProvider

        provider = LLMProvider(
            provider="mock",
            api_key="",
            base_url="",
            model="test-model",
        )

        import asyncio

        async def make_call():
            return await provider.call(
                messages=[{"role": "user", "content": "test"}],
                return_usage=True,
            )

        result, usage = asyncio.get_event_loop().run_until_complete(make_call())
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5
        assert usage.total_tokens == 15


class TestRetainUsesRetainLLMConfig:
    """Test that retain operations use the retain LLM config."""

    def test_retain_llm_config_is_passed_to_orchestrator(self):
        """Verify retain operation is configured to use _retain_llm_config."""
        from hindsight_api import MemoryEngine

        engine = MemoryEngine(
            memory_llm_provider="mock",
            memory_llm_model="default-model",
            retain_llm_provider="mock",
            retain_llm_model="retain-specific-model",
            reflect_llm_provider="mock",
            reflect_llm_model="reflect-specific-model",
            skip_llm_verification=True,
            lazy_reranker=True,
        )

        # Verify the retain LLM config is set correctly
        assert engine._retain_llm_config.model == "retain-specific-model"
        assert engine._retain_llm_config.provider == "mock"

        # Verify it's different from the reflect config
        assert engine._retain_llm_config.model != engine._reflect_llm_config.model


class TestReflectUsesReflectLLMConfig:
    """Test that reflect operations use the reflect LLM config."""

    def test_reflect_llm_config_is_set_correctly(self):
        """Verify reflect/think operation is configured to use _reflect_llm_config."""
        from hindsight_api import MemoryEngine

        engine = MemoryEngine(
            memory_llm_provider="mock",
            memory_llm_model="default-model",
            retain_llm_provider="mock",
            retain_llm_model="retain-specific-model",
            reflect_llm_provider="mock",
            reflect_llm_model="reflect-specific-model",
            skip_llm_verification=True,
            lazy_reranker=True,
        )

        # Verify the reflect LLM config is set correctly
        assert engine._reflect_llm_config.model == "reflect-specific-model"
        assert engine._reflect_llm_config.provider == "mock"

        # Verify it's different from the retain config
        assert engine._reflect_llm_config.model != engine._retain_llm_config.model


class TestRetryAndBackoffConfiguration:
    """Test retry and backoff configuration options."""

    def test_global_retry_backoff_config_defaults(self):
        """Test that global retry/backoff settings have correct defaults."""
        from hindsight_api.config import get_config

        config = get_config()

        # Verify global defaults
        assert config.llm_max_retries == 10
        assert config.llm_initial_backoff == 1.0
        assert config.llm_max_backoff == 60.0

    def test_per_operation_retry_backoff_config_from_env(self):
        """Test that per-operation retry/backoff settings are loaded from environment."""
        from hindsight_api.config import clear_config_cache

        # Set per-operation overrides
        os.environ["HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES"] = "3"
        os.environ["HINDSIGHT_API_RETAIN_LLM_INITIAL_BACKOFF"] = "2.0"
        os.environ["HINDSIGHT_API_RETAIN_LLM_MAX_BACKOFF"] = "120.0"
        os.environ["HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES"] = "5"
        os.environ["HINDSIGHT_API_REFLECT_LLM_INITIAL_BACKOFF"] = "1.5"
        os.environ["HINDSIGHT_API_REFLECT_LLM_MAX_BACKOFF"] = "90.0"

        try:
            clear_config_cache()
            from hindsight_api.config import get_config

            config = get_config()

            # Verify retain overrides
            assert config.retain_llm_max_retries == 3
            assert config.retain_llm_initial_backoff == 2.0
            assert config.retain_llm_max_backoff == 120.0

            # Verify reflect overrides
            assert config.reflect_llm_max_retries == 5
            assert config.reflect_llm_initial_backoff == 1.5
            assert config.reflect_llm_max_backoff == 90.0

            # Verify global defaults remain unchanged
            assert config.llm_max_retries == 10
            assert config.llm_initial_backoff == 1.0
            assert config.llm_max_backoff == 60.0
        finally:
            # Clean up
            os.environ.pop("HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES", None)
            os.environ.pop("HINDSIGHT_API_RETAIN_LLM_INITIAL_BACKOFF", None)
            os.environ.pop("HINDSIGHT_API_RETAIN_LLM_MAX_BACKOFF", None)
            os.environ.pop("HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES", None)
            os.environ.pop("HINDSIGHT_API_REFLECT_LLM_INITIAL_BACKOFF", None)
            os.environ.pop("HINDSIGHT_API_REFLECT_LLM_MAX_BACKOFF", None)
            clear_config_cache()

    def test_per_operation_retry_backoff_fallback_to_global(self):
        """Test that per-operation settings fall back to global when not set."""
        from hindsight_api.config import clear_config_cache, get_config

        # Set only global values
        os.environ["HINDSIGHT_API_LLM_MAX_RETRIES"] = "7"
        os.environ["HINDSIGHT_API_LLM_INITIAL_BACKOFF"] = "3.0"
        os.environ["HINDSIGHT_API_LLM_MAX_BACKOFF"] = "180.0"

        try:
            clear_config_cache()
            config = get_config()

            # Per-operation should be None (will fall back to global at runtime)
            assert config.retain_llm_max_retries is None
            assert config.retain_llm_initial_backoff is None
            assert config.retain_llm_max_backoff is None

            # Global values should be set
            assert config.llm_max_retries == 7
            assert config.llm_initial_backoff == 3.0
            assert config.llm_max_backoff == 180.0
        finally:
            os.environ.pop("HINDSIGHT_API_LLM_MAX_RETRIES", None)
            os.environ.pop("HINDSIGHT_API_LLM_INITIAL_BACKOFF", None)
            os.environ.pop("HINDSIGHT_API_LLM_MAX_BACKOFF", None)
            clear_config_cache()

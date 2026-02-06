"""
Tests for configuration validation.

Verifies that config validation catches invalid parameter combinations.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up environment for each test, restoring original values after."""
    from hindsight_api.config import clear_config_cache

    # Save original environment values
    env_vars_to_save = [
        "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS",
        "HINDSIGHT_API_RETAIN_CHUNK_SIZE",
        "HINDSIGHT_API_LLM_PROVIDER",
        "HINDSIGHT_API_LLM_MODEL",
    ]

    # Save original values
    original_values = {}
    for key in env_vars_to_save:
        original_values[key] = os.environ.get(key)

    clear_config_cache()

    yield

    # Restore original environment
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

    clear_config_cache()


def test_retain_max_completion_tokens_must_be_greater_than_chunk_size():
    """Test that RETAIN_MAX_COMPLETION_TOKENS > RETAIN_CHUNK_SIZE validation works."""
    from hindsight_api.config import HindsightConfig

    # Set invalid config: max_completion_tokens <= chunk_size
    os.environ["HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS"] = "1000"
    os.environ["HINDSIGHT_API_RETAIN_CHUNK_SIZE"] = "2000"
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    # Should raise ValueError with helpful message
    with pytest.raises(ValueError) as exc_info:
        HindsightConfig.from_env()

    error_message = str(exc_info.value)

    # Verify error message contains helpful information
    assert "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS" in error_message
    assert "1000" in error_message
    assert "HINDSIGHT_API_RETAIN_CHUNK_SIZE" in error_message
    assert "2000" in error_message
    assert "must be greater than" in error_message
    assert "You have two options to fix this:" in error_message
    assert "Increase HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS" in error_message
    assert "Use a model that supports" in error_message


def test_retain_max_completion_tokens_equal_to_chunk_size_fails():
    """Test that RETAIN_MAX_COMPLETION_TOKENS == RETAIN_CHUNK_SIZE also fails."""
    from hindsight_api.config import HindsightConfig

    # Set invalid config: max_completion_tokens == chunk_size
    os.environ["HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS"] = "3000"
    os.environ["HINDSIGHT_API_RETAIN_CHUNK_SIZE"] = "3000"
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        HindsightConfig.from_env()

    error_message = str(exc_info.value)
    assert "must be greater than" in error_message


def test_valid_retain_config_succeeds():
    """Test that valid config with max_completion_tokens > chunk_size works."""
    from hindsight_api.config import HindsightConfig

    # Set valid config: max_completion_tokens > chunk_size
    os.environ["HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS"] = "64000"
    os.environ["HINDSIGHT_API_RETAIN_CHUNK_SIZE"] = "3000"
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    # Should not raise
    config = HindsightConfig.from_env()
    assert config.retain_max_completion_tokens == 64000
    assert config.retain_chunk_size == 3000


# Note: The BadRequestError wrapping is implemented in fact_extraction.py
# but requires a complex integration test setup. The functionality is
# straightforward: when a BadRequestError containing keywords like
# "max_tokens", "max_completion_tokens", or "maximum context" is caught,
# it's wrapped in a ValueError with helpful guidance.
#
# The config validation tests above ensure users get early feedback
# about invalid configurations before runtime errors occur.

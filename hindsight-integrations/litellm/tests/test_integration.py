"""Integration tests for hindsight-litellm."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from hindsight_litellm import (
    configure,
    enable,
    disable,
    is_enabled,
    cleanup,
    get_config,
    is_configured,
    reset_config,
    HindsightConfig,
    MemoryInjectionMode,
)
from hindsight_litellm.callbacks import HindsightCallback, get_callback, cleanup_callback


class TestConfiguration:
    """Tests for configuration management."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()
        disable()

    def teardown_method(self):
        """Clean up after each test."""
        cleanup()

    def test_configure_creates_config(self):
        """Test that configure creates a config object."""
        config = configure(
            bank_id="test-agent",
            hindsight_api_url="http://localhost:8888",
        )

        assert config is not None
        assert config.bank_id == "test-agent"
        assert config.hindsight_api_url == "http://localhost:8888"
        assert config.enabled is True

    def test_configure_with_all_options(self):
        """Test configure with all options."""
        config = configure(
            hindsight_api_url="http://custom:9999",
            bank_id="custom-agent",
            api_key="secret-key",
            store_conversations=False,
            inject_memories=False,
            injection_mode=MemoryInjectionMode.PREPEND_USER,
            max_memories=5,
            max_memory_tokens=1000,
            recall_budget="high",
            fact_types=["world", "opinion"],
            document_id="doc-123",
            enabled=True,
            excluded_models=["gpt-3.5*"],
            verbose=True,
        )

        assert config.hindsight_api_url == "http://custom:9999"
        assert config.bank_id == "custom-agent"
        assert config.api_key == "secret-key"
        assert config.store_conversations is False
        assert config.inject_memories is False
        assert config.injection_mode == MemoryInjectionMode.PREPEND_USER
        assert config.max_memories == 5
        assert config.max_memory_tokens == 1000
        assert config.recall_budget == "high"
        assert config.fact_types == ["world", "opinion"]
        assert config.document_id == "doc-123"
        assert config.excluded_models == ["gpt-3.5*"]
        assert config.verbose is True

    def test_is_configured_without_bank_id(self):
        """Test is_configured returns False without bank_id."""
        configure(hindsight_api_url="http://localhost:8888")
        assert is_configured() is False

    def test_is_configured_with_bank_id(self):
        """Test is_configured returns True with bank_id."""
        configure(bank_id="test-agent")
        assert is_configured() is True

    def test_reset_config(self):
        """Test reset_config clears the configuration."""
        configure(bank_id="test-agent")
        assert is_configured() is True

        reset_config()
        assert get_config() is None
        assert is_configured() is False


class TestEnableDisable:
    """Tests for enable/disable functionality."""

    def setup_method(self):
        """Reset state before each test."""
        cleanup()

    def teardown_method(self):
        """Clean up after each test."""
        cleanup()

    def test_enable_without_config_raises(self):
        """Test enable raises error without configuration."""
        with pytest.raises(RuntimeError, match="not configured"):
            enable()

    def test_enable_registers_callback(self):
        """Test enable registers callback with LiteLLM."""
        import litellm

        configure(bank_id="test-agent")
        enable()

        callback = get_callback()
        assert callback in litellm.callbacks
        assert is_enabled() is True

    def test_disable_removes_callback(self):
        """Test disable removes callback from LiteLLM."""
        import litellm

        configure(bank_id="test-agent")
        enable()
        assert is_enabled() is True

        disable()
        callback = get_callback()
        assert callback not in litellm.callbacks
        assert is_enabled() is False

    def test_enable_idempotent(self):
        """Test enable is idempotent (can be called multiple times)."""
        import litellm

        configure(bank_id="test-agent")

        # Enable multiple times
        enable()
        enable()
        enable()

        # Should only have one callback
        callback = get_callback()
        assert litellm.callbacks.count(callback) == 1


class TestCallback:
    """Tests for the HindsightCallback class."""

    def setup_method(self):
        """Reset state before each test."""
        cleanup()

    def teardown_method(self):
        """Clean up after each test."""
        cleanup()

    def test_extract_user_query_simple(self):
        """Test extracting user query from simple messages."""
        callback = HindsightCallback()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        query = callback._extract_user_query(messages)
        assert query == "What is the capital of France?"

    def test_extract_user_query_from_last_user_message(self):
        """Test extracting query from last user message."""
        callback = HindsightCallback()
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]

        query = callback._extract_user_query(messages)
        assert query == "Second question"

    def test_extract_user_query_structured_content(self):
        """Test extracting query from structured content (vision)."""
        callback = HindsightCallback()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            },
        ]

        query = callback._extract_user_query(messages)
        assert query == "What's in this image?"

    def test_extract_user_query_multiple_text_parts(self):
        """Test extracting query with multiple text parts."""
        callback = HindsightCallback()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part."},
                    {"type": "text", "text": "Second part."},
                ],
            },
        ]

        query = callback._extract_user_query(messages)
        assert query == "First part. Second part."

    def test_format_memories(self):
        """Test formatting memories into context string."""
        callback = HindsightCallback()
        config = HindsightConfig(bank_id="test", max_memories=10, verbose=False)

        memories = [
            {"text": "User likes Python", "fact_type": "world", "weight": 0.95},
            {"text": "User works at Google", "fact_type": "world", "weight": 0.8},
        ]

        formatted = callback._format_memories(memories, config)

        assert "Relevant Memories" in formatted
        assert "User likes Python" in formatted
        assert "User works at Google" in formatted
        assert "[WORLD]" in formatted

    def test_format_memories_with_verbose(self):
        """Test formatting memories with verbose mode shows weights."""
        callback = HindsightCallback()
        config = HindsightConfig(bank_id="test", max_memories=10, verbose=True)

        memories = [
            {"text": "User likes Python", "fact_type": "world", "weight": 0.95},
        ]

        formatted = callback._format_memories(memories, config)

        assert "relevance: 0.95" in formatted

    def test_inject_memories_as_system_message(self):
        """Test injecting memories as system message."""
        callback = HindsightCallback()
        config = HindsightConfig(
            bank_id="test",
            injection_mode=MemoryInjectionMode.SYSTEM_MESSAGE,
        )

        messages = [
            {"role": "user", "content": "Hello"},
        ]
        memory_context = "# Relevant Memories\n1. User is John"

        result = callback._inject_memories_into_messages(messages, memory_context, config)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "Relevant Memories" in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_inject_memories_prepend_to_existing_system(self):
        """Test injecting memories appends to existing system message."""
        callback = HindsightCallback()
        config = HindsightConfig(
            bank_id="test",
            injection_mode=MemoryInjectionMode.SYSTEM_MESSAGE,
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        memory_context = "# Relevant Memories\n1. User is John"

        result = callback._inject_memories_into_messages(messages, memory_context, config)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "You are helpful." in result[0]["content"]
        assert "Relevant Memories" in result[0]["content"]

    def test_inject_memories_prepend_user_mode(self):
        """Test injecting memories in prepend_user mode."""
        callback = HindsightCallback()
        config = HindsightConfig(
            bank_id="test",
            injection_mode=MemoryInjectionMode.PREPEND_USER,
        )

        messages = [
            {"role": "user", "content": "What's my name?"},
        ]
        memory_context = "# Relevant Memories\n1. User is John"

        result = callback._inject_memories_into_messages(messages, memory_context, config)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Relevant Memories" in result[0]["content"]
        assert "What's my name?" in result[0]["content"]

    def test_should_skip_model_exact_match(self):
        """Test model exclusion with exact match."""
        callback = HindsightCallback()
        config = HindsightConfig(
            bank_id="test",
            excluded_models=["gpt-3.5-turbo"],
        )

        assert callback._should_skip_model("gpt-3.5-turbo", config) is True
        assert callback._should_skip_model("gpt-4", config) is False

    def test_should_skip_model_wildcard(self):
        """Test model exclusion with wildcard pattern."""
        callback = HindsightCallback()
        config = HindsightConfig(
            bank_id="test",
            excluded_models=["gpt-3.5*", "claude-instant-*"],
        )

        assert callback._should_skip_model("gpt-3.5-turbo", config) is True
        assert callback._should_skip_model("gpt-3.5-turbo-16k", config) is True
        assert callback._should_skip_model("claude-instant-1.2", config) is True
        assert callback._should_skip_model("gpt-4", config) is False
        assert callback._should_skip_model("claude-3-opus", config) is False


class TestDeduplication:
    """Tests for conversation deduplication."""

    def setup_method(self):
        """Reset state before each test."""
        cleanup()

    def teardown_method(self):
        """Clean up after each test."""
        cleanup()

    def test_compute_conversation_hash(self):
        """Test computing conversation hash."""
        callback = HindsightCallback()

        hash1 = callback._compute_conversation_hash("Hello", "Hi there!")
        hash2 = callback._compute_conversation_hash("Hello", "Hi there!")
        hash3 = callback._compute_conversation_hash("Hello", "Different response")

        # Same content should produce same hash
        assert hash1 == hash2
        # Different content should produce different hash
        assert hash1 != hash3

    def test_compute_conversation_hash_case_insensitive(self):
        """Test that hash is case insensitive."""
        callback = HindsightCallback()

        hash1 = callback._compute_conversation_hash("HELLO", "HI THERE!")
        hash2 = callback._compute_conversation_hash("hello", "hi there!")

        assert hash1 == hash2

    def test_is_duplicate_first_time(self):
        """Test first occurrence is not a duplicate."""
        callback = HindsightCallback()

        result = callback._is_duplicate("abc123")

        assert result is False

    def test_is_duplicate_second_time(self):
        """Test second occurrence is a duplicate."""
        callback = HindsightCallback()

        callback._is_duplicate("abc123")  # First time
        result = callback._is_duplicate("abc123")  # Second time

        assert result is True

    def test_is_duplicate_different_hashes(self):
        """Test different hashes are not duplicates."""
        callback = HindsightCallback()

        callback._is_duplicate("abc123")
        result = callback._is_duplicate("xyz789")

        assert result is False


class TestContextManager:
    """Tests for the hindsight_memory context manager."""

    def setup_method(self):
        """Reset state before each test."""
        cleanup()

    def teardown_method(self):
        """Clean up after each test."""
        cleanup()

    def test_context_manager_enables_and_disables(self):
        """Test context manager enables and disables correctly."""
        from hindsight_litellm import hindsight_memory

        assert is_enabled() is False

        with hindsight_memory(bank_id="test-agent"):
            assert is_enabled() is True
            assert get_config().bank_id == "test-agent"

        assert is_enabled() is False

    def test_context_manager_restores_previous_config(self):
        """Test context manager restores previous configuration."""
        from hindsight_litellm import hindsight_memory

        # Set up initial config
        configure(bank_id="original-agent")
        enable()
        assert get_config().bank_id == "original-agent"

        # Use context manager with different config
        with hindsight_memory(bank_id="temporary-agent"):
            assert get_config().bank_id == "temporary-agent"

        # Should restore original config
        assert get_config().bank_id == "original-agent"
        assert is_enabled() is True

    def test_context_manager_with_fact_types(self):
        """Test context manager with fact_types parameter."""
        from hindsight_litellm import hindsight_memory

        with hindsight_memory(bank_id="test-agent", fact_types=["world", "opinion"]):
            config = get_config()
            assert config.fact_types == ["world", "opinion"]


class TestFactTypes:
    """Tests for fact_types configuration."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Clean up after each test."""
        cleanup()

    def test_configure_with_fact_types(self):
        """Test configuring with fact_types."""
        config = configure(
            bank_id="test-agent",
            fact_types=["world", "agent", "opinion"],
        )

        assert config.fact_types == ["world", "agent", "opinion"]

    def test_configure_without_fact_types(self):
        """Test configuring without fact_types defaults to None."""
        config = configure(bank_id="test-agent")

        assert config.fact_types is None

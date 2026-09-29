"""Integration tests for hindsight-litellm."""

import pytest

from hindsight_litellm import (
    configure,
    set_defaults,
    get_defaults,
    enable,
    disable,
    is_enabled,
    cleanup,
    get_config,
    is_configured,
    reset_config,
    hindsight_memory,
    MemoryInjectionMode,
)
from hindsight_litellm.callbacks import HindsightCallback


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
            hindsight_api_url="http://localhost:8888",
        )
        # Set defaults separately (new API)
        defaults = set_defaults(bank_id="test-agent")

        assert config is not None
        assert config.hindsight_api_url == "http://localhost:8888"
        assert defaults.bank_id == "test-agent"

    def test_configure_with_all_options(self):
        """Test configure with all options."""
        config = configure(
            hindsight_api_url="http://custom:9999",
            api_key="secret-key",
            store_conversations=False,
            inject_memories=False,
            injection_mode=MemoryInjectionMode.PREPEND_USER,
            excluded_models=["gpt-3.5*"],
            verbose=True,
            sync_storage=True,
        )

        # Set defaults separately (new API)
        defaults = set_defaults(
            bank_id="custom-agent",
            max_memories=5,
            max_memory_tokens=1000,
            budget="high",
            fact_types=["world", "opinion"],
            document_id="doc-123",
        )

        assert config.hindsight_api_url == "http://custom:9999"
        assert config.api_key == "secret-key"
        assert config.store_conversations is False
        assert config.inject_memories is False
        assert config.injection_mode == MemoryInjectionMode.PREPEND_USER
        assert config.excluded_models == ["gpt-3.5*"]
        assert config.verbose is True
        assert config.sync_storage is True

        assert defaults.bank_id == "custom-agent"
        assert defaults.max_memories == 5
        assert defaults.max_memory_tokens == 1000
        assert defaults.budget == "high"
        assert defaults.fact_types == ["world", "opinion"]
        assert defaults.document_id == "doc-123"

    def test_is_configured_with_defaults(self):
        """configure() alone (no bank_id) leaves is_configured() False."""
        configure()
        assert is_configured() is False

    def test_is_configured_with_bank_id_in_defaults(self):
        """Test is_configured returns True with bank_id in defaults."""
        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")
        assert is_configured() is True

    def test_is_configured_with_explicit_bank_id(self):
        """Test is_configured returns True with explicit bank_id."""
        configure(bank_id="test-agent")
        assert is_configured() is True

    def test_reset_config(self):
        """Test reset_config clears the configuration."""
        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")
        assert is_configured() is True

        reset_config()
        assert get_config() is None
        assert get_defaults() is None
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

    def test_enable_without_bank_id_raises(self):
        """enable() raises RuntimeError when no bank_id has been set."""
        configure(hindsight_api_url="http://localhost:8888")
        with pytest.raises(RuntimeError, match="bank_id"):
            enable()

    def test_enable_sets_enabled_flag(self):
        """Test enable sets the enabled flag."""
        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")
        enable()

        assert is_enabled() is True

    def test_disable_clears_enabled_flag(self):
        """Test disable clears the enabled flag."""
        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")
        enable()
        assert is_enabled() is True

        disable()
        assert is_enabled() is False

    def test_enable_idempotent(self):
        """Test enable is idempotent (can be called multiple times)."""
        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")

        # Enable multiple times
        enable()
        enable()
        enable()

        # Should still be enabled
        assert is_enabled() is True


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
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/img.png"},
                    },
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

        # Create config and defaults with new API
        configure(hindsight_api_url="http://localhost:8888", verbose=False)
        set_defaults(bank_id="test", max_memories=10)

        config = get_config()
        defaults = get_defaults()

        memories = [
            {"text": "User likes Python", "fact_type": "world", "weight": 0.95},
            {"text": "User works at Google", "fact_type": "world", "weight": 0.8},
        ]

        # Signature is: _format_memories(results, settings, config)
        formatted = callback._format_memories(memories, defaults, config)

        assert "Relevant Memories" in formatted
        assert "User likes Python" in formatted
        assert "User works at Google" in formatted
        assert "[WORLD]" in formatted

    def test_format_memories_with_verbose(self):
        """Test formatting memories with verbose mode shows weights."""
        callback = HindsightCallback()

        # Create config and defaults with new API
        configure(hindsight_api_url="http://localhost:8888", verbose=True)
        set_defaults(bank_id="test", max_memories=10)

        config = get_config()
        defaults = get_defaults()

        memories = [
            {"text": "User likes Python", "fact_type": "world", "weight": 0.95},
        ]

        # Signature is: _format_memories(results, settings, config)
        formatted = callback._format_memories(memories, defaults, config)

        assert "relevance: 0.95" in formatted

    def test_inject_memories_as_system_message(self):
        """Test injecting memories as system message."""
        callback = HindsightCallback()

        configure(
            hindsight_api_url="http://localhost:8888",
            injection_mode=MemoryInjectionMode.SYSTEM_MESSAGE,
        )
        set_defaults(bank_id="test")

        config = get_config()

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

        configure(
            hindsight_api_url="http://localhost:8888",
            injection_mode=MemoryInjectionMode.SYSTEM_MESSAGE,
        )
        set_defaults(bank_id="test")

        config = get_config()

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

        configure(
            hindsight_api_url="http://localhost:8888",
            injection_mode=MemoryInjectionMode.PREPEND_USER,
        )
        set_defaults(bank_id="test")

        config = get_config()

        messages = [
            {"role": "user", "content": "What's my name?"},
        ]
        memory_context = "# Relevant Memories\n1. User is John"

        result = callback._inject_memories_into_messages(messages, memory_context, config)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Relevant Memories" in result[0]["content"]
        assert "What's my name?" in result[0]["content"]

    def test_inject_memories_uses_last_user_message_when_no_hindsight_query(self):
        """Regression test: inject_memories=True should not require hindsight_query.

        The documented Quick Start example does not pass hindsight_query; the
        injection path must fall back to the last user message automatically.
        See: feat(litellm) #167 regression.
        """
        from unittest.mock import MagicMock, patch

        callback = HindsightCallback()

        configure(
            hindsight_api_url="http://localhost:8888",
            inject_memories=True,
        )
        set_defaults(bank_id="test-agent")

        messages = [{"role": "user", "content": "What did we discuss about AI?"}]
        kwargs = {}  # No hindsight_query provided — this is the regression scenario

        mock_memory = MagicMock()
        mock_memory.text = "AI is cool"
        mock_memory.type = "world"
        mock_memory.weight = 0.9

        with patch.object(callback, "_recall_memories_sync", return_value=[mock_memory]) as mock_recall:
            callback.log_pre_api_call(
                model="gpt-4o-mini",
                messages=messages,
                kwargs=kwargs,
            )
            # Should have called recall with the last user message as query
            mock_recall.assert_called_once()
            query_used = mock_recall.call_args[0][0]
            assert query_used == "What did we discuss about AI?"

        # Memories should have been injected into messages
        assert any("AI is cool" in str(m.get("content", "")) for m in messages)

    def test_inject_memories_hindsight_query_takes_precedence(self):
        """When hindsight_query is provided it should be used over the last user message."""
        from unittest.mock import MagicMock, patch

        callback = HindsightCallback()

        configure(
            hindsight_api_url="http://localhost:8888",
            inject_memories=True,
        )
        set_defaults(bank_id="test-agent")

        messages = [{"role": "user", "content": "Hello"}]
        kwargs = {"hindsight_query": "What do I know about Alice?"}

        mock_memory = MagicMock()
        mock_memory.text = "Alice likes cats"
        mock_memory.type = "world"
        mock_memory.weight = 0.9

        with patch.object(callback, "_recall_memories_sync", return_value=[mock_memory]) as mock_recall:
            callback.log_pre_api_call(
                model="gpt-4o-mini",
                messages=messages,
                kwargs=kwargs,
            )
            query_used = mock_recall.call_args[0][0]
            assert query_used == "What do I know about Alice?"

    def test_should_skip_model_exact_match(self):
        """Test model exclusion with exact match."""
        callback = HindsightCallback()

        configure(
            hindsight_api_url="http://localhost:8888",
            excluded_models=["gpt-3.5-turbo"],
        )
        set_defaults(bank_id="test")

        config = get_config()

        assert callback._should_skip_model("gpt-3.5-turbo", config) is True
        assert callback._should_skip_model("gpt-4", config) is False

    def test_should_skip_model_wildcard(self):
        """Test model exclusion with wildcard pattern."""
        callback = HindsightCallback()

        configure(
            hindsight_api_url="http://localhost:8888",
            excluded_models=["gpt-3.5*", "claude-instant-*"],
        )
        set_defaults(bank_id="test")

        config = get_config()

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
            defaults = get_defaults()
            assert defaults.bank_id == "test-agent"

        assert is_enabled() is False

    def test_context_manager_restores_previous_config(self):
        """Test context manager restores previous configuration."""
        from hindsight_litellm import hindsight_memory

        # Set up initial config
        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="original-agent")
        enable()
        assert get_defaults().bank_id == "original-agent"

        # Use context manager with different config
        with hindsight_memory(bank_id="temporary-agent"):
            assert get_defaults().bank_id == "temporary-agent"

        # Should restore original config
        assert get_defaults().bank_id == "original-agent"
        assert is_enabled() is True

    def test_context_manager_with_fact_types(self):
        """Test context manager with fact_types parameter."""
        from hindsight_litellm import hindsight_memory

        with hindsight_memory(bank_id="test-agent", fact_types=["world", "opinion"]):
            defaults = get_defaults()
            assert defaults.fact_types == ["world", "opinion"]


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
        configure(hindsight_api_url="http://localhost:8888")
        defaults = set_defaults(
            bank_id="test-agent",
            fact_types=["world", "agent", "opinion"],
        )

        assert defaults.fact_types == ["world", "agent", "opinion"]

    def test_configure_without_fact_types(self):
        """Test configuring without fact_types defaults to None."""
        configure(hindsight_api_url="http://localhost:8888")
        defaults = set_defaults(bank_id="test-agent")

        assert defaults.fact_types is None


class TestSetDefaults:
    """Tests for set_defaults functionality."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Clean up after each test."""
        cleanup()

    def test_set_defaults_creates_defaults(self):
        """Test set_defaults creates a defaults object."""
        defaults = set_defaults(bank_id="test-agent")

        assert defaults is not None
        assert defaults.bank_id == "test-agent"

    def test_set_defaults_with_all_options(self):
        """Test set_defaults with all options."""
        defaults = set_defaults(
            bank_id="test-agent",
            document_id="doc-123",
            budget="high",
            fact_types=["world", "opinion"],
            max_memories=10,
            max_memory_tokens=2048,
            use_reflect=True,
            reflect_include_facts=True,
            reflect_context="I am a helpful assistant.",
            include_entities=False,
            trace=True,
        )

        assert defaults.bank_id == "test-agent"
        assert defaults.document_id == "doc-123"
        assert defaults.budget == "high"
        assert defaults.fact_types == ["world", "opinion"]
        assert defaults.max_memories == 10
        assert defaults.max_memory_tokens == 2048
        assert defaults.use_reflect is True
        assert defaults.reflect_include_facts is True
        assert defaults.reflect_context == "I am a helpful assistant."
        assert defaults.include_entities is False
        assert defaults.trace is True

    def test_set_defaults_updates_existing(self):
        """Test set_defaults updates existing defaults."""
        set_defaults(bank_id="first-agent", budget="low")
        defaults = set_defaults(budget="high")  # Only update budget

        assert defaults.bank_id == "first-agent"  # Preserved
        assert defaults.budget == "high"  # Updated

    def test_get_defaults_returns_none_initially(self):
        """Test get_defaults returns None when not set."""
        assert get_defaults() is None


class TestStreamingResponseHandling:
    """Tests for streaming response handling in monkeypatch wrappers and callbacks.

    Regression tests for https://github.com/vectorize-io/hindsight/issues/1221
    CustomStreamWrapper objects don't have .choices and crash _format_conversation_for_storage.
    """

    def setup_method(self):
        """Reset state before each test."""
        cleanup()

    def teardown_method(self):
        """Clean up after each test."""
        cleanup()

    def _make_fake_chunks(self):
        """Create fake streaming chunks mimicking LiteLLM's ModelResponseStream."""
        from unittest.mock import MagicMock

        chunks = []
        for text in ["Hello", " ", "world", "!"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = text
            chunks.append(chunk)
        return chunks

    def test_format_conversation_for_storage_returns_empty_for_stream(self):
        """Test that _format_conversation_for_storage returns empty for stream objects.

        Reproduces the exact bug from issue #1221.
        """
        from hindsight_litellm import _format_conversation_for_storage

        class FakeStreamWrapper:
            """Mimics litellm.utils.CustomStreamWrapper which lacks .choices."""

            pass

        messages = [{"role": "user", "content": "Hello"}]
        stream_response = FakeStreamWrapper()

        result = _format_conversation_for_storage(messages, stream_response)
        assert result == ""

    def test_store_conversation_skips_stream_response(self):
        """Test that _store_conversation handles streaming responses gracefully."""
        from hindsight_litellm import _store_conversation

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")

        class FakeStreamWrapper:
            pass

        messages = [{"role": "user", "content": "Hello"}]
        stream_response = FakeStreamWrapper()

        # Should not raise
        _store_conversation(messages, stream_response, "gpt-4o-mini")

    def test_sync_storage_forwards_sync_true_to_retain(self):
        """Regression: sync_storage=True must call retain(..., sync=True).

        Without sync=True, the inner retain() dispatches to a daemon thread
        and returns immediately, so the POST never lands on the server before
        a short-lived process exits — cross-process drop-in silently fails.
        """
        from unittest.mock import patch, MagicMock
        import hindsight_litellm

        configure(hindsight_api_url="http://localhost:8888", sync_storage=True, store_conversations=True)
        set_defaults(bank_id="test-bank")

        messages = [{"role": "user", "content": "Hi"}]
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Hello back."
        response.choices[0].message.tool_calls = None

        with patch.object(hindsight_litellm, "retain") as mock_retain:
            hindsight_litellm._store_conversation(messages, response, "gpt-4o-mini")
            mock_retain.assert_called_once()
            assert mock_retain.call_args.kwargs.get("sync") is True, (
                "sync_storage=True must forward sync=True to retain() so the POST completes before the function returns"
            )

    def test_sync_storage_streamed_forwards_sync_true_to_retain(self):
        """Regression: streamed sync_storage path must also pass sync=True."""
        from unittest.mock import patch
        import hindsight_litellm

        configure(hindsight_api_url="http://localhost:8888", sync_storage=True, store_conversations=True)
        set_defaults(bank_id="test-bank")

        with patch.object(hindsight_litellm, "retain") as mock_retain:
            hindsight_litellm._store_conversation_from_text("USER: hi\n\nASSISTANT: hello", "gpt-4o-mini")
            mock_retain.assert_called_once()
            assert mock_retain.call_args.kwargs.get("sync") is True

    def test_wrapped_completion_returns_stream_wrapper(self):
        """Test that completion() wraps streaming responses for deferred storage."""
        from unittest.mock import patch, MagicMock
        from hindsight_litellm import _LiteLLMStreamWrapper

        configure(hindsight_api_url="http://localhost:8888", store_conversations=True)
        set_defaults(bank_id="test-agent")
        enable()

        # Create a fake stream (no .choices attribute)
        class FakeStreamWrapper:
            pass

        fake_stream = FakeStreamWrapper()

        with patch("litellm.completion", return_value=fake_stream):
            import hindsight_litellm

            response = hindsight_litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
            # Should return a stream wrapper, not the raw stream
            assert isinstance(response, _LiteLLMStreamWrapper)

    def test_stream_wrapper_yields_all_chunks(self):
        """Test that _LiteLLMStreamWrapper passes through all chunks."""
        from hindsight_litellm import _LiteLLMStreamWrapper

        configure(hindsight_api_url="http://localhost:8888", store_conversations=False)
        set_defaults(bank_id="test-agent")

        chunks = self._make_fake_chunks()
        messages = [{"role": "user", "content": "Hello"}]

        wrapper = _LiteLLMStreamWrapper(iter(chunks), messages, "gpt-4o-mini")

        collected = list(wrapper)
        assert len(collected) == 4

    def test_stream_wrapper_stores_conversation_on_exhaustion(self):
        """Test that _LiteLLMStreamWrapper stores conversation when stream is consumed."""
        from unittest.mock import patch
        from hindsight_litellm import _LiteLLMStreamWrapper

        configure(hindsight_api_url="http://localhost:8888", store_conversations=True)
        set_defaults(bank_id="test-agent")

        chunks = self._make_fake_chunks()
        messages = [{"role": "user", "content": "Hello"}]

        wrapper = _LiteLLMStreamWrapper(iter(chunks), messages, "gpt-4o-mini")

        with patch("hindsight_litellm._store_conversation_from_text") as mock_store:
            # Consume all chunks
            list(wrapper)

            mock_store.assert_called_once()
            stored_text = mock_store.call_args[0][0]
            assert "USER: Hello" in stored_text
            assert "ASSISTANT: Hello world!" in stored_text

    def test_stream_wrapper_stores_on_context_manager_exit(self):
        """Test that _LiteLLMStreamWrapper stores conversation on context manager exit."""
        from unittest.mock import patch
        from hindsight_litellm import _LiteLLMStreamWrapper

        configure(hindsight_api_url="http://localhost:8888", store_conversations=True)
        set_defaults(bank_id="test-agent")

        chunks = self._make_fake_chunks()
        messages = [{"role": "user", "content": "Hello"}]

        wrapper = _LiteLLMStreamWrapper(iter(chunks), messages, "gpt-4o-mini")

        with patch("hindsight_litellm._store_conversation_from_text") as mock_store:
            with wrapper:
                for _ in wrapper:
                    pass
            # Should store exactly once (not double-store)
            mock_store.assert_called_once()

    def test_callback_log_success_with_stream_response(self):
        """Test that HindsightCallback.log_success_event handles stream responses."""
        from hindsight_litellm.callbacks import HindsightCallback

        configure(hindsight_api_url="http://localhost:8888", store_conversations=True)
        set_defaults(bank_id="test-agent")

        callback = HindsightCallback()

        class FakeStreamWrapper:
            pass

        kwargs = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        # Should not raise AttributeError
        callback.log_success_event(
            kwargs=kwargs,
            response_obj=FakeStreamWrapper(),
            start_time=0.0,
            end_time=1.0,
        )


class TestStreamingSupport:
    """Tests for streaming support in wrappers."""

    def test_wrap_openai_with_stream_no_error(self):
        """Test that wrap_openai handles streaming without errors."""
        from unittest.mock import Mock, MagicMock
        from hindsight_litellm.wrappers import wrap_openai

        # Create mock OpenAI client
        mock_client = Mock()
        mock_stream = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream

        # Wrap the client with store_conversations=False
        wrapped = wrap_openai(
            mock_client,
            hindsight_api_url="http://localhost:8888",
            bank_id="test-agent",
            store_conversations=False,  # Disable storage for this test
        )

        # Call with stream=True
        result = wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        # Should return the stream without errors
        assert result == mock_stream
        # Verify the underlying client was called with stream=True
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True

    def test_wrap_anthropic_with_stream_no_error(self):
        """Test that wrap_anthropic handles streaming without errors."""
        from unittest.mock import Mock, MagicMock
        from hindsight_litellm.wrappers import wrap_anthropic

        # Create mock Anthropic client
        mock_client = Mock()
        mock_stream = MagicMock()
        mock_client.messages.create.return_value = mock_stream

        # Wrap the client with store_conversations=False
        wrapped = wrap_anthropic(
            mock_client,
            hindsight_api_url="http://localhost:8888",
            bank_id="test-agent",
            store_conversations=False,  # Disable storage for this test
        )

        # Call with stream=True
        result = wrapped.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        # Should return the stream without errors
        assert result == mock_stream
        # Verify the underlying client was called with stream=True
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["stream"] is True

    def test_wrap_openai_stream_stores_conversation(self):
        """Test that streaming stores conversation after all chunks are consumed."""
        from unittest.mock import Mock, MagicMock, patch
        from hindsight_litellm.wrappers import wrap_openai

        # Create mock OpenAI client
        mock_client = Mock()

        # Create mock stream chunks
        class MockChunk:
            def __init__(self, content):
                self.choices = [MagicMock()]
                self.choices[0].delta.content = content

        chunks = [
            MockChunk("Hello"),
            MockChunk(" "),
            MockChunk("world"),
            MockChunk("!"),
        ]
        mock_client.chat.completions.create.return_value = iter(chunks)

        # Wrap the client
        wrapped = wrap_openai(
            mock_client,
            hindsight_api_url="http://localhost:8888",
            bank_id="test-agent",
            store_conversations=True,  # Enable storage
        )

        # Mock the hindsight client
        mock_hindsight_client = MagicMock()
        with patch.object(wrapped, "_get_hindsight_client", return_value=mock_hindsight_client):
            # Call with stream=True
            result = wrapped.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )

            # Consume all chunks
            collected = []
            for chunk in result:
                collected.append(chunk)

            # Verify all chunks were yielded
            assert len(collected) == 4

            # Verify retain was called with the complete conversation
            mock_hindsight_client.retain.assert_called_once()
            call_kwargs = mock_hindsight_client.retain.call_args[1]
            assert "USER: Hello" in call_kwargs["content"]
            assert "ASSISTANT: Hello world!" in call_kwargs["content"]

    def test_wrap_anthropic_stream_stores_conversation(self):
        """Test that streaming stores conversation after all chunks are consumed."""
        from unittest.mock import Mock, MagicMock, patch
        from hindsight_litellm.wrappers import wrap_anthropic

        # Create mock Anthropic client
        mock_client = Mock()

        # Create mock stream chunks
        class MockChunk:
            def __init__(self, content):
                self.type = "content_block_delta"
                self.delta = MagicMock()
                self.delta.text = content

        chunks = [
            MockChunk("Hello"),
            MockChunk(" "),
            MockChunk("world"),
            MockChunk("!"),
        ]
        mock_client.messages.create.return_value = iter(chunks)

        # Wrap the client
        wrapped = wrap_anthropic(
            mock_client,
            hindsight_api_url="http://localhost:8888",
            bank_id="test-agent",
            store_conversations=True,  # Enable storage
        )

        # Mock the hindsight client
        mock_hindsight_client = MagicMock()
        with patch.object(wrapped, "_get_hindsight_client", return_value=mock_hindsight_client):
            # Call with stream=True
            result = wrapped.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )

            # Consume all chunks
            collected = []
            for chunk in result:
                collected.append(chunk)

            # Verify all chunks were yielded
            assert len(collected) == 4

            # Verify retain was called with the complete conversation
            mock_hindsight_client.retain.assert_called_once()
            call_kwargs = mock_hindsight_client.retain.call_args[1]
            assert "USER: Hello" in call_kwargs["content"]
            assert "ASSISTANT: Hello world!" in call_kwargs["content"]


class TestRecallNewParams:
    """Tests for new parameters added to recall()."""

    def setup_method(self):
        reset_config()

    def teardown_method(self):
        cleanup()

    def test_recall_passes_include_entities(self):
        """recall() should forward include_entities to the client."""
        from unittest.mock import MagicMock, patch
        from hindsight_litellm import recall

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")

        mock_client = MagicMock()
        mock_client.recall.return_value = []

        with patch("hindsight_litellm.wrappers._get_client", return_value=mock_client):
            recall("test query", include_entities=False)
            call_kwargs = mock_client.recall.call_args[1]
            assert call_kwargs["include_entities"] is False

    def test_recall_passes_trace(self):
        """recall() should forward trace to the client."""
        from unittest.mock import MagicMock, patch
        from hindsight_litellm import recall

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")

        mock_client = MagicMock()
        mock_client.recall.return_value = []

        with patch("hindsight_litellm.wrappers._get_client", return_value=mock_client):
            recall("test query", trace=True)
            call_kwargs = mock_client.recall.call_args[1]
            assert call_kwargs["trace"] is True

    def test_recall_passes_recall_tags(self):
        """recall() should forward recall_tags and recall_tags_match to the client."""
        from unittest.mock import MagicMock, patch
        from hindsight_litellm import recall

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")

        mock_client = MagicMock()
        mock_client.recall.return_value = []

        with patch("hindsight_litellm.wrappers._get_client", return_value=mock_client):
            recall("test query", recall_tags=["user:alice"], recall_tags_match="any_strict")
            call_kwargs = mock_client.recall.call_args[1]
            assert call_kwargs["tags"] == ["user:alice"]
            assert call_kwargs["tags_match"] == "any_strict"

    def test_recall_no_tags_key_when_empty(self):
        """recall() should not pass tags key when recall_tags is None."""
        from unittest.mock import MagicMock, patch
        from hindsight_litellm import recall

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")

        mock_client = MagicMock()
        mock_client.recall.return_value = []

        with patch("hindsight_litellm.wrappers._get_client", return_value=mock_client):
            recall("test query")
            call_kwargs = mock_client.recall.call_args[1]
            assert "tags" not in call_kwargs

    def test_recall_inherits_include_entities_from_defaults(self):
        """recall() should inherit include_entities from set_defaults() if not overridden."""
        from unittest.mock import MagicMock, patch
        from hindsight_litellm import recall

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent", include_entities=False)

        mock_client = MagicMock()
        mock_client.recall.return_value = []

        with patch("hindsight_litellm.wrappers._get_client", return_value=mock_client):
            recall("test query")
            call_kwargs = mock_client.recall.call_args[1]
            assert call_kwargs["include_entities"] is False


class TestReflectNewParams:
    """Tests for new parameters added to reflect()."""

    def setup_method(self):
        reset_config()

    def teardown_method(self):
        cleanup()

    def test_reflect_passes_recall_tags(self):
        """reflect() should forward recall_tags and recall_tags_match to the client."""
        from unittest.mock import MagicMock, patch
        from hindsight_litellm import reflect

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")

        mock_result = MagicMock()
        mock_result.text = "some reflection"
        mock_result.based_on = None
        mock_client = MagicMock()
        mock_client.reflect.return_value = mock_result

        with patch("hindsight_litellm.wrappers._get_client", return_value=mock_client):
            reflect("test query", recall_tags=["user:bob"], recall_tags_match="all")
            call_kwargs = mock_client.reflect.call_args[1]
            assert call_kwargs["tags"] == ["user:bob"]
            assert call_kwargs["tags_match"] == "all"

    def test_reflect_no_tags_key_when_empty(self):
        """reflect() should not pass tags key when recall_tags is None."""
        from unittest.mock import MagicMock, patch
        from hindsight_litellm import reflect

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test-agent")

        mock_result = MagicMock()
        mock_result.text = "some reflection"
        mock_result.based_on = None
        mock_client = MagicMock()
        mock_client.reflect.return_value = mock_result

        with patch("hindsight_litellm.wrappers._get_client", return_value=mock_client):
            reflect("test query")
            call_kwargs = mock_client.reflect.call_args[1]
            assert "tags" not in call_kwargs


class TestHindsightMemoryNewParams:
    """Tests for new parameters added to hindsight_memory() context manager."""

    def setup_method(self):
        cleanup()

    def teardown_method(self):
        cleanup()

    def test_hindsight_memory_session_id(self):
        """hindsight_memory() should pass session_id through to defaults."""
        from hindsight_litellm import hindsight_memory

        with hindsight_memory(bank_id="test-agent", session_id="conv-123"):
            defaults = get_defaults()
            assert defaults.session_id == "conv-123"

    def test_hindsight_memory_use_reflect(self):
        """hindsight_memory() should pass use_reflect through to defaults."""
        from hindsight_litellm import hindsight_memory

        with hindsight_memory(bank_id="test-agent", use_reflect=True):
            defaults = get_defaults()
            assert defaults.use_reflect is True

    def test_hindsight_memory_tags(self):
        """hindsight_memory() should pass tags and recall_tags through to defaults."""
        from hindsight_litellm import hindsight_memory

        with hindsight_memory(
            bank_id="test-agent",
            tags=["session:abc"],
            recall_tags=["session:abc"],
            recall_tags_match="any_strict",
        ):
            defaults = get_defaults()
            assert defaults.tags == ["session:abc"]
            assert defaults.recall_tags == ["session:abc"]
            assert defaults.recall_tags_match == "any_strict"

    def test_hindsight_memory_reflect_context(self):
        """hindsight_memory() should pass reflect_context through to defaults."""
        from hindsight_litellm import hindsight_memory

        with hindsight_memory(bank_id="test-agent", reflect_context="I am an assistant."):
            defaults = get_defaults()
            assert defaults.reflect_context == "I am an assistant."

    def test_hindsight_memory_default_url_is_cloud(self):
        """hindsight_memory() default URL should be the cloud endpoint, not localhost."""
        from hindsight_litellm import hindsight_memory
        from hindsight_litellm.config import DEFAULT_HINDSIGHT_API_URL

        with hindsight_memory(bank_id="test-agent"):
            config = get_config()
            assert config.hindsight_api_url == DEFAULT_HINDSIGHT_API_URL


class TestValidation:
    """Tests for input validation in configure() and set_defaults()."""

    def setup_method(self):
        reset_config()

    def teardown_method(self):
        cleanup()

    def test_configure_invalid_budget_raises(self):
        """configure() raises ValueError for invalid budget."""
        with pytest.raises(ValueError, match="budget"):
            configure(budget="extreme")

    def test_set_defaults_invalid_budget_raises(self):
        """set_defaults() raises ValueError for invalid budget."""
        configure(hindsight_api_url="http://localhost:8888")
        with pytest.raises(ValueError, match="budget"):
            set_defaults(bank_id="test", budget="extreme")

    def test_configure_invalid_recall_tags_match_raises(self):
        """configure() raises ValueError for invalid recall_tags_match."""
        with pytest.raises(ValueError, match="recall_tags_match"):
            configure(recall_tags_match="fuzzy")

    def test_set_defaults_invalid_recall_tags_match_raises(self):
        """set_defaults() raises ValueError for invalid recall_tags_match."""
        configure(hindsight_api_url="http://localhost:8888")
        with pytest.raises(ValueError, match="recall_tags_match"):
            set_defaults(bank_id="test", recall_tags_match="fuzzy")

    def test_configure_valid_budgets_accepted(self):
        """configure() accepts all valid budget values."""
        for budget in ("low", "mid", "high"):
            configure(hindsight_api_url="http://localhost:8888", budget=budget)
            assert get_config().default_settings.budget == budget

    def test_configure_valid_tags_match_accepted(self):
        """configure() accepts all valid recall_tags_match values."""
        for match in ("any", "all", "any_strict", "all_strict"):
            configure(hindsight_api_url="http://localhost:8888", recall_tags_match=match)
            assert get_config().default_settings.recall_tags_match == match

    def test_configure_document_id_emits_deprecation_warning(self):
        """configure() with document_id emits DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="document_id"):
            configure(hindsight_api_url="http://localhost:8888", document_id="doc-123")

    def test_set_defaults_document_id_emits_deprecation_warning(self):
        """set_defaults() with document_id emits DeprecationWarning."""
        configure(hindsight_api_url="http://localhost:8888")
        with pytest.warns(DeprecationWarning, match="document_id"):
            set_defaults(bank_id="test", document_id="doc-123")


class TestInjectionMode:
    """Tests for injection_mode in _inject_memories()."""

    def setup_method(self):
        reset_config()

    def teardown_method(self):
        cleanup()

    def test_system_message_injection_appends_to_existing_system(self):
        """SYSTEM_MESSAGE mode appends memories to an existing system message."""
        from unittest.mock import MagicMock, patch
        from hindsight_litellm import _inject_memories

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test", injection_mode=MemoryInjectionMode.SYSTEM_MESSAGE)

        mock_result = MagicMock()
        mock_result.text = "User likes Rust"
        mock_result.type = "world"

        mock_client = MagicMock()
        mock_client.recall.return_value = [mock_result]

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What do I like?"},
        ]

        with patch("hindsight_litellm._get_client", return_value=mock_client):
            result = _inject_memories(messages)

        assert result[0]["role"] == "system"
        assert "You are helpful." in result[0]["content"]
        assert "Rust" in result[0]["content"]

    def test_system_message_injection_creates_system_when_absent(self):
        """SYSTEM_MESSAGE mode creates a new system message when none exists."""
        from unittest.mock import MagicMock, patch
        from hindsight_litellm import _inject_memories

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test", injection_mode=MemoryInjectionMode.SYSTEM_MESSAGE)

        mock_result = MagicMock()
        mock_result.text = "User likes Python"
        mock_result.type = "world"

        mock_client = MagicMock()
        mock_client.recall.return_value = [mock_result]

        messages = [{"role": "user", "content": "Hello"}]

        with patch("hindsight_litellm._get_client", return_value=mock_client):
            result = _inject_memories(messages)

        assert result[0]["role"] == "system"
        assert "Python" in result[0]["content"]

    def test_prepend_user_injection_prepends_to_last_user_message(self):
        """PREPEND_USER mode prepends memories to the last user message."""
        from unittest.mock import MagicMock, patch
        from hindsight_litellm import _inject_memories

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test", injection_mode=MemoryInjectionMode.PREPEND_USER)

        mock_result = MagicMock()
        mock_result.text = "User likes Go"
        mock_result.type = "world"

        mock_client = MagicMock()
        mock_client.recall.return_value = [mock_result]

        messages = [{"role": "user", "content": "What do I like?"}]

        with patch("hindsight_litellm._get_client", return_value=mock_client):
            result = _inject_memories(messages)

        # No system message should be created
        assert all(m["role"] != "system" for m in result)
        user_msg = next(m for m in result if m["role"] == "user")
        content = user_msg["content"]
        assert "Go" in content
        assert "What do I like?" in content
        # Memory context should come before user text
        assert content.index("Go") < content.index("What do I like?")

    def test_prepend_user_does_not_touch_system_message(self):
        """PREPEND_USER mode leaves existing system message untouched."""
        from unittest.mock import MagicMock, patch
        from hindsight_litellm import _inject_memories

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test", injection_mode=MemoryInjectionMode.PREPEND_USER)

        mock_result = MagicMock()
        mock_result.text = "User likes TypeScript"
        mock_result.type = "world"

        mock_client = MagicMock()
        mock_client.recall.return_value = [mock_result]

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        with patch("hindsight_litellm._get_client", return_value=mock_client):
            result = _inject_memories(messages)

        system_msg = next(m for m in result if m["role"] == "system")
        assert system_msg["content"] == "You are helpful."
        assert "TypeScript" not in system_msg["content"]


class TestInjectionPathPassesApiKey:
    """Regression pin: ``_inject_memories`` must pass ``config.api_key`` to ``_get_client``.

    Without this, the inject path constructs an un-keyed Hindsight client and any
    Cloud read (recall/reflect) fails with HTTP 401 even when the storage path of
    the same package authenticates fine — the bug the 2026-06-02 audit caught.
    The retain path threads api_key correctly via wrappers.py; the inject path
    in __init__.py did not.
    """

    def setup_method(self):
        reset_config()

    def teardown_method(self):
        cleanup()

    def test_inject_path_passes_configured_api_key_to_get_client(self):
        """Configure with api_key + url → _inject_memories must call _get_client(url, api_key)."""
        from unittest.mock import MagicMock, patch
        from hindsight_litellm import _inject_memories

        configure(
            hindsight_api_url="https://api.hindsight.vectorize.io",
            api_key="hsk_test_key_for_cloud_auth_regression_pin",
        )
        set_defaults(bank_id="test")

        mock_result = MagicMock(text="User likes Haskell", type="world")
        mock_client = MagicMock()
        mock_client.recall.return_value = [mock_result]

        messages = [{"role": "user", "content": "What do I like?"}]
        with patch("hindsight_litellm._get_client", return_value=mock_client) as gc:
            _inject_memories(messages)

        gc.assert_called()
        call_args = gc.call_args
        # Tolerate positional or keyword forms — both end up at the same callsite.
        url = call_args.kwargs.get("api_url") or (call_args.args[0] if call_args.args else None)
        key = call_args.kwargs.get("api_key") or (call_args.args[1] if len(call_args.args) > 1 else None)
        assert url == "https://api.hindsight.vectorize.io", (
            f"inject path did not forward configured URL to _get_client: got url={url!r}"
        )
        assert key == "hsk_test_key_for_cloud_auth_regression_pin", (
            f"inject path did not forward configured api_key to _get_client: got key={key!r}. "
            "This regression silently breaks reads against the hosted backend (401)."
        )


class TestQueryField:
    """Tests for the query field in HindsightCallSettings."""

    def setup_method(self):
        reset_config()

    def teardown_method(self):
        cleanup()

    def test_defaults_query_used_when_no_custom_query(self):
        """defaults.query is used as recall query when no hindsight_query kwarg given."""
        from unittest.mock import MagicMock, patch
        from hindsight_litellm import _inject_memories
        from hindsight_litellm.config import HindsightCallSettings

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test")

        # Manually set query on defaults
        import hindsight_litellm.config as cfg

        cfg._global_config.default_settings.query = "favorite language"

        mock_result = MagicMock()
        mock_result.text = "User likes Rust"
        mock_result.type = "world"

        mock_client = MagicMock()
        mock_client.recall.return_value = [mock_result]

        messages = [{"role": "user", "content": "Tell me something"}]

        with patch("hindsight_litellm._get_client", return_value=mock_client):
            _inject_memories(messages)

        call_kwargs = mock_client.recall.call_args[1]
        assert call_kwargs["query"] == "favorite language"

    def test_custom_query_overrides_defaults_query(self):
        """custom_query parameter overrides defaults.query."""
        from unittest.mock import MagicMock, patch
        from hindsight_litellm import _inject_memories

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="test")

        import hindsight_litellm.config as cfg

        cfg._global_config.default_settings.query = "default query"

        mock_client = MagicMock()
        mock_client.recall.return_value = []

        messages = [{"role": "user", "content": "Hello"}]

        with patch("hindsight_litellm._get_client", return_value=mock_client):
            _inject_memories(messages, custom_query="override query")

        call_kwargs = mock_client.recall.call_args[1]
        assert call_kwargs["query"] == "override query"


class TestHindsightErrorConsistency:
    """Tests that HindsightError (not ValueError) is raised for config errors."""

    def setup_method(self):
        reset_config()

    def teardown_method(self):
        cleanup()

    def test_inject_memories_raises_hindsight_error_when_no_bank_id(self):
        """_inject_memories raises HindsightError (not ValueError) when bank_id missing."""
        from hindsight_litellm import HindsightError, _inject_memories
        from hindsight_litellm.callbacks import HindsightError as CallbackHindsightError

        configure(hindsight_api_url="http://localhost:8888")
        # No bank_id set

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(HindsightError):
            _inject_memories(messages)

    def test_callback_log_pre_api_call_raises_hindsight_error_when_no_bank_id(self):
        """HindsightCallback.log_pre_api_call raises HindsightError (not ValueError)."""
        from hindsight_litellm.callbacks import HindsightCallback, HindsightError

        configure(hindsight_api_url="http://localhost:8888")
        # No bank_id set

        callback = HindsightCallback()
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(HindsightError):
            callback.log_pre_api_call("gpt-4o-mini", messages, {})


class TestContextManagerFullRestore:
    """Tests that hindsight_memory() restores ALL settings on exit."""

    def setup_method(self):
        cleanup()

    def teardown_method(self):
        cleanup()

    def test_restores_sync_storage(self):
        """hindsight_memory() restores sync_storage after exit."""
        configure(hindsight_api_url="http://localhost:8888", sync_storage=True)
        set_defaults(bank_id="original")

        with hindsight_memory(bank_id="temp", hindsight_api_url="http://localhost:8888"):
            assert get_config().sync_storage is False  # default inside context

        assert get_config().sync_storage is True  # restored

    def test_restores_tags(self):
        """hindsight_memory() restores tags after exit."""
        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="original", tags=["env:prod"])

        with hindsight_memory(bank_id="temp", hindsight_api_url="http://localhost:8888", tags=["env:test"]):
            assert get_defaults().tags == ["env:test"]

        assert get_defaults().tags == ["env:prod"]

    def test_restores_recall_tags(self):
        """hindsight_memory() restores recall_tags after exit."""
        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="original", recall_tags=["user:alice"], recall_tags_match="any_strict")

        with hindsight_memory(bank_id="temp", hindsight_api_url="http://localhost:8888"):
            assert get_defaults().recall_tags is None

        assert get_defaults().recall_tags == ["user:alice"]
        assert get_defaults().recall_tags_match == "any_strict"

    def test_restores_reflect_context(self):
        """hindsight_memory() restores reflect_context after exit."""
        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="original", reflect_context="Be concise.")

        with hindsight_memory(bank_id="temp", hindsight_api_url="http://localhost:8888"):
            assert get_defaults().reflect_context is None

        assert get_defaults().reflect_context == "Be concise."

    def test_restores_to_none_when_no_prior_config(self):
        """hindsight_memory() resets config to None if none was set before."""
        assert get_config() is None

        with hindsight_memory(bank_id="temp"):
            assert get_config() is not None

        assert get_config() is None

    def test_re_enables_if_was_enabled_before(self):
        """hindsight_memory() re-enables if integration was enabled before entering."""
        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="original")
        enable()
        assert is_enabled() is True

        with hindsight_memory(bank_id="temp", hindsight_api_url="http://localhost:8888"):
            assert is_enabled() is True

        assert is_enabled() is True  # still enabled after context


class TestSetBankMission:
    """set_bank_mission() configures a bank's mission via hindsight-client."""

    def test_requires_bank_id_when_not_configured(self):
        """Raises HindsightError if no bank_id can be resolved."""
        from hindsight_litellm import HindsightError, reset_config, set_bank_mission

        reset_config()
        with pytest.raises(HindsightError, match="bank_id"):
            set_bank_mission("Some mission")

    def test_forwards_to_hindsight_client_create_bank(self):
        """Calls Hindsight.create_bank with the resolved bank_id, name, and mission."""
        from unittest.mock import MagicMock, patch

        from hindsight_litellm import configure, set_bank_mission, set_defaults

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="agent-7")

        mock_client = MagicMock()
        with patch("hindsight_client.Hindsight", return_value=mock_client):
            set_bank_mission(
                "Remember user preferences across conversations.",
                name="Pref Agent",
            )

        mock_client.create_bank.assert_called_once_with(
            bank_id="agent-7",
            name="Pref Agent",
            mission="Remember user preferences across conversations.",
        )

    def test_explicit_bank_id_overrides_default(self):
        """Explicit bank_id arg takes precedence over the configured default."""
        from unittest.mock import MagicMock, patch

        from hindsight_litellm import configure, set_bank_mission, set_defaults

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="default-bank")

        mock_client = MagicMock()
        with patch("hindsight_client.Hindsight", return_value=mock_client):
            set_bank_mission("override mission", bank_id="other-bank")

        called_kwargs = mock_client.create_bank.call_args.kwargs
        assert called_kwargs["bank_id"] == "other-bank"

    def test_wraps_client_failure_in_hindsight_error(self):
        """Underlying client errors are re-raised as HindsightError."""
        from unittest.mock import MagicMock, patch

        from hindsight_litellm import HindsightError, configure, set_bank_mission, set_defaults

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="agent-7")

        mock_client = MagicMock()
        mock_client.create_bank.side_effect = RuntimeError("server unavailable")
        with patch("hindsight_client.Hindsight", return_value=mock_client):
            with pytest.raises(HindsightError, match="server unavailable"):
                set_bank_mission("a mission")


class TestEnableDualInjectionGuard:
    """enable() warns if a HindsightCallback is already registered."""

    def setup_method(self):
        cleanup()

    def teardown_method(self):
        cleanup()

    def test_warns_when_callback_already_registered(self):
        import litellm

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="agent-7")

        cb = HindsightCallback()
        original = list(getattr(litellm, "callbacks", []) or [])
        litellm.callbacks = original + [cb]

        try:
            with pytest.warns(RuntimeWarning, match="HindsightCallback"):
                enable()
        finally:
            disable()
            litellm.callbacks = original

    def test_no_warning_without_callback(self):
        import warnings as _warnings

        configure(hindsight_api_url="http://localhost:8888")
        set_defaults(bank_id="agent-7")

        with _warnings.catch_warnings():
            _warnings.simplefilter("error", RuntimeWarning)
            enable()  # should not raise
        disable()


class TestExcludedModelsInEnablePath:
    """excluded_models patterns are honored by the enable() monkeypatch path."""

    def setup_method(self):
        cleanup()

    def teardown_method(self):
        cleanup()

    def test_excluded_model_bypasses_injection(self):
        """A model matching excluded_models should bypass the wrapped path."""
        from unittest.mock import patch

        from hindsight_litellm import _wrapped_completion

        configure(
            hindsight_api_url="http://localhost:8888",
            excluded_models=["gpt-3.5*"],
        )
        set_defaults(bank_id="agent-7")
        enable()

        try:
            with patch("hindsight_litellm._original_completion") as orig:
                with patch("hindsight_litellm._inject_memories") as inj:
                    orig.return_value = "ok"
                    _wrapped_completion(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "hi"}],
                    )
                    inj.assert_not_called()
                    orig.assert_called_once()
        finally:
            disable()

    def test_non_excluded_model_still_injects(self):
        """A model not matching excluded_models should run the normal flow."""
        from unittest.mock import patch

        from hindsight_litellm import _wrapped_completion

        configure(
            hindsight_api_url="http://localhost:8888",
            excluded_models=["gpt-3.5*"],
            store_conversations=False,  # avoid the storage path in this unit test
        )
        set_defaults(bank_id="agent-7")
        enable()

        try:
            with patch("hindsight_litellm._original_completion") as orig:
                with patch("hindsight_litellm._inject_memories") as inj:
                    inj.return_value = [{"role": "user", "content": "hi"}]
                    orig.return_value = "ok"
                    _wrapped_completion(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "hi"}],
                    )
                    inj.assert_called_once()
        finally:
            disable()


class TestDedupLRU:
    """Dedup cache uses true LRU eviction and is thread-safe."""

    def test_oldest_hash_evicted_first(self):
        callback = HindsightCallback()
        callback._max_hash_cache = 3

        # Fill the cache
        callback._is_duplicate("a")
        callback._is_duplicate("b")
        callback._is_duplicate("c")
        # All three present
        assert "a" in callback._recent_hashes
        assert "b" in callback._recent_hashes
        assert "c" in callback._recent_hashes

        # Add a fourth — oldest ("a") should be evicted
        callback._is_duplicate("d")
        assert "a" not in callback._recent_hashes
        assert "d" in callback._recent_hashes

    def test_touch_on_hit_protects_from_eviction(self):
        callback = HindsightCallback()
        callback._max_hash_cache = 3

        callback._is_duplicate("a")
        callback._is_duplicate("b")
        callback._is_duplicate("c")

        # Touch "a" — now "b" should be the oldest
        callback._is_duplicate("a")

        # Add new entry — "b" should be evicted, not "a"
        callback._is_duplicate("d")
        assert "a" in callback._recent_hashes
        assert "b" not in callback._recent_hashes


class TestWrapperClose:
    """close() / context-manager support on the native client wrappers."""

    def test_close_closes_cached_hindsight_client_and_underlying(self):
        from unittest.mock import MagicMock, Mock

        from hindsight_litellm.wrappers import wrap_openai

        mock_client = Mock()
        wrapped = wrap_openai(
            mock_client,
            hindsight_api_url="http://localhost:8888",
            bank_id="test-agent",
        )
        # Force the lazy Hindsight client into existence.
        fake_hs = MagicMock()
        wrapped._hindsight_client = fake_hs

        wrapped.close()

        fake_hs.close.assert_called_once()
        mock_client.close.assert_called_once()
        assert wrapped._hindsight_client is None

    def test_close_is_idempotent(self):
        from unittest.mock import Mock

        from hindsight_litellm.wrappers import wrap_openai

        mock_client = Mock()
        wrapped = wrap_openai(mock_client, hindsight_api_url="http://localhost:8888", bank_id="b")
        wrapped.close()
        wrapped.close()  # second call must not raise
        assert wrapped._hindsight_client is None

    def test_close_without_underlying_close_is_safe(self):
        from unittest.mock import MagicMock

        from hindsight_litellm.wrappers import wrap_openai

        # Underlying client with no close() attribute.
        class _NoClose:
            def __init__(self):
                self.chat = MagicMock()

        wrapped = wrap_openai(_NoClose(), hindsight_api_url="http://localhost:8888", bank_id="b")
        wrapped.close()  # must not raise even though underlying has no close()

    def test_context_manager_closes_on_exit(self):
        from unittest.mock import MagicMock, Mock

        from hindsight_litellm.wrappers import wrap_openai

        mock_client = Mock()
        with wrap_openai(mock_client, hindsight_api_url="http://localhost:8888", bank_id="b") as wrapped:
            wrapped._hindsight_client = MagicMock()
            hs = wrapped._hindsight_client
        hs.close.assert_called_once()
        mock_client.close.assert_called_once()

    def test_wrap_anthropic_close(self):
        from unittest.mock import MagicMock, Mock

        from hindsight_litellm.wrappers import wrap_anthropic

        mock_client = Mock()
        wrapped = wrap_anthropic(mock_client, hindsight_api_url="http://localhost:8888", bank_id="b")
        fake_hs = MagicMock()
        wrapped._hindsight_client = fake_hs
        wrapped.close()
        fake_hs.close.assert_called_once()
        mock_client.close.assert_called_once()
        assert wrapped._hindsight_client is None


class TestWrapBankSetupOwnsLoop:
    """wrap_*() bank/mission setup must own the thread loop before creating a client."""

    def test_wrap_openai_mission_calls_ensure_loop(self):
        from unittest.mock import Mock, patch

        from hindsight_litellm.wrappers import wrap_openai

        with (
            patch("hindsight_litellm.wrappers.ensure_loop") as mock_ensure,
            patch("hindsight_client.Hindsight") as mock_hs,
        ):
            wrap_openai(
                Mock(),
                hindsight_api_url="http://localhost:8888",
                bank_id="b",
                mission="learn the user's stack",
                bank_name="My Bank",
            )
        mock_ensure.assert_called()
        mock_hs.return_value.create_bank.assert_called_once()

    def test_wrap_anthropic_mission_calls_ensure_loop(self):
        from unittest.mock import Mock, patch

        from hindsight_litellm.wrappers import wrap_anthropic

        with (
            patch("hindsight_litellm.wrappers.ensure_loop") as mock_ensure,
            patch("hindsight_client.Hindsight") as mock_hs,
        ):
            wrap_anthropic(
                Mock(),
                hindsight_api_url="http://localhost:8888",
                bank_id="b",
                mission="learn the user's stack",
                bank_name="My Bank",
            )
        mock_ensure.assert_called()
        mock_hs.return_value.create_bank.assert_called_once()

    def test_wrap_openai_without_mission_skips_bank_setup(self):
        from unittest.mock import Mock, patch

        from hindsight_litellm.wrappers import wrap_openai

        with patch("hindsight_client.Hindsight") as mock_hs:
            wrap_openai(Mock(), hindsight_api_url="http://localhost:8888", bank_id="b")
        mock_hs.assert_not_called()


class TestDeterministicInjectionFlow:
    """Full hindsight_litellm.completion() inject flow, fully mocked.

    The in-CI / no-keys analog of the live enable()/completion tests in
    test_e2e.py: it spies on litellm.completion and mocks the Hindsight client's
    recall, so the inject path (recall -> format -> inject into LLM messages) is
    exercised deterministically. Deterministic bucket (no requires_real_llm).
    """

    def setup_method(self):
        cleanup()

    def teardown_method(self):
        cleanup()

    def test_completion_injects_recalled_memory_into_llm_messages(self):
        import litellm
        from unittest.mock import MagicMock, patch

        import hindsight_litellm

        configure(hindsight_api_url="http://localhost:8888", store_conversations=False)
        set_defaults(bank_id="t")

        # Mocked Hindsight client: recall() returns one canned memory.
        mem = MagicMock()
        mem.text = "The user's favorite language is Haskell"
        mem.type = "world"
        mock_client = MagicMock()
        mock_client.recall.return_value = [mem]

        captured = {}

        def spy_completion(*args, **kwargs):
            captured["messages"] = kwargs.get("messages")
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "ok"
            resp.choices = [choice]
            return resp

        with (
            patch("hindsight_litellm._get_client", return_value=mock_client),
            patch.object(litellm, "completion", spy_completion),
        ):
            response = hindsight_litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "What language do I prefer?"}],
            )

        assert response.choices[0].message.content == "ok"
        mock_client.recall.assert_called_once()
        assert captured["messages"], "litellm.completion received no messages"
        joined = " ".join(str(m.get("content", "")) for m in captured["messages"]).lower()
        assert "haskell" in joined, f"recalled memory not injected into LLM messages: {joined}"

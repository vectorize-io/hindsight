"""Test that API namespaces ensure daemon is started before each call."""

from unittest.mock import Mock, patch

import pytest

from hindsight import HindsightEmbedded


@pytest.fixture
def embedded_client():
    """Create an embedded client for testing."""
    return HindsightEmbedded(
        profile="test",
        llm_provider="openai",
        llm_api_key="test-key",
    )


def test_banks_create_ensures_daemon_started(embedded_client):
    """Test that banks.create() calls _ensure_started()."""
    # Mock _ensure_started to track calls
    with patch.object(embedded_client, "_ensure_started") as mock_ensure:
        # Mock the underlying client to avoid actual API call
        mock_client = Mock()
        embedded_client._client = mock_client

        # Call namespace method
        try:
            embedded_client.banks.create(bank_id="test", name="Test Bank")
        except Exception:
            pass  # We don't care if the actual call fails

        # Verify _ensure_started was called
        mock_ensure.assert_called_once()


def test_mental_models_list_ensures_daemon_started(embedded_client):
    """Test that mental_models.list() calls _ensure_started()."""
    with patch.object(embedded_client, "_ensure_started") as mock_ensure:
        mock_client = Mock()
        embedded_client._client = mock_client

        try:
            embedded_client.mental_models.list(bank_id="test")
        except Exception:
            pass

        mock_ensure.assert_called_once()


def test_directives_list_ensures_daemon_started(embedded_client):
    """Test that directives.list() calls _ensure_started()."""
    with patch.object(embedded_client, "_ensure_started") as mock_ensure:
        mock_client = Mock()
        embedded_client._client = mock_client

        try:
            embedded_client.directives.list(bank_id="test")
        except Exception:
            pass

        mock_ensure.assert_called_once()


def test_memories_list_ensures_daemon_started(embedded_client):
    """Test that memories.list() calls _ensure_started()."""
    with patch.object(embedded_client, "_ensure_started") as mock_ensure:
        mock_client = Mock()
        embedded_client._client = mock_client

        try:
            embedded_client.memories.list(bank_id="test")
        except Exception:
            pass

        mock_ensure.assert_called_once()


def test_multiple_calls_ensure_daemon_each_time(embedded_client):
    """Test that each namespace call ensures daemon is started."""
    with patch.object(embedded_client, "_ensure_started") as mock_ensure:
        mock_client = Mock()
        embedded_client._client = mock_client

        # Make multiple calls
        try:
            embedded_client.banks.create(bank_id="test", name="Test")
        except Exception:
            pass

        try:
            embedded_client.mental_models.list(bank_id="test")
        except Exception:
            pass

        try:
            embedded_client.directives.list(bank_id="test")
        except Exception:
            pass

        # Should be called 3 times (once per namespace method call)
        assert mock_ensure.call_count == 3


def test_daemon_restart_handling(embedded_client):
    """Test that namespace methods can recover from daemon crash."""
    call_count = 0

    def mock_ensure_started():
        """Mock that simulates daemon restart."""
        nonlocal call_count
        call_count += 1
        # Create a new mock client each time (simulating daemon restart)
        embedded_client._client = Mock()
        embedded_client._started = True

    with patch.object(embedded_client, "_ensure_started", side_effect=mock_ensure_started):
        # First call - daemon starts
        embedded_client.banks.create(bank_id="test", name="Test")
        assert call_count == 1

        # Simulate daemon crash by clearing client
        embedded_client._client = None
        embedded_client._started = False

        # Second call - daemon restarts
        embedded_client.banks.create(bank_id="test", name="Test")
        assert call_count == 2


def test_ensure_started_calls_manager(embedded_client):
    """Test that _ensure_started actually starts the daemon via manager."""
    # Mock the manager
    mock_manager = Mock()
    mock_manager.ensure_running.return_value = True
    mock_manager.get_url.return_value = "http://localhost:54321"

    embedded_client._manager = mock_manager

    # Mock Hindsight client constructor
    with patch("hindsight.embedded.Hindsight") as mock_hindsight_class:
        mock_client = Mock()
        mock_hindsight_class.return_value = mock_client

        # Call _ensure_started
        embedded_client._ensure_started()

        # Verify manager was called
        mock_manager.ensure_running.assert_called_once_with(
            embedded_client.config, embedded_client.profile
        )
        mock_manager.get_url.assert_called_once_with(embedded_client.profile)

        # Verify Hindsight client was created
        mock_hindsight_class.assert_called_once_with(base_url="http://localhost:54321")


def test_namespace_singleton_behavior(embedded_client):
    """Test that namespace properties return the same instance."""
    banks1 = embedded_client.banks
    banks2 = embedded_client.banks

    # Should be the same instance
    assert banks1 is banks2

    # Same for other namespaces
    assert embedded_client.mental_models is embedded_client.mental_models
    assert embedded_client.directives is embedded_client.directives
    assert embedded_client.memories is embedded_client.memories

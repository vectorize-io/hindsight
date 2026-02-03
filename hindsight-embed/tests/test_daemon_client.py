"""Tests for daemon_client module."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from hindsight_embed import daemon_client


@pytest.fixture
def config():
    """Default config for tests."""
    return {
        "llm_api_key": "test-key",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "bank_id": "test-bank",
    }


@pytest.fixture
def mock_cli_binary(tmp_path):
    """Create a mock CLI binary."""
    cli_path = tmp_path / "hindsight"
    cli_path.write_text("#!/bin/bash\nexit 0")
    cli_path.chmod(0o755)
    return cli_path


class TestRunCli:
    """Tests for run_cli function."""

    def test_run_cli_with_external_api_url(self, config, mock_cli_binary, monkeypatch):
        """Test that external HINDSIGHT_EMBED_API_URL skips daemon startup."""
        # Set up environment with external API URL
        external_api_url = "http://external-api:8000"
        monkeypatch.setenv("HINDSIGHT_EMBED_API_URL", external_api_url)

        # Mock functions
        mock_ensure_cli = Mock(return_value=True)
        mock_find_cli = Mock(return_value=mock_cli_binary)
        mock_ensure_daemon = Mock(return_value=True)
        mock_subprocess_run = Mock(return_value=Mock(returncode=0))

        with (
            patch.object(daemon_client, "ensure_cli_installed", mock_ensure_cli),
            patch.object(daemon_client, "find_cli_binary", mock_find_cli),
            patch.object(daemon_client, "ensure_daemon_running", mock_ensure_daemon),
            patch("subprocess.run", mock_subprocess_run),
        ):
            # Run CLI
            exit_code = daemon_client.run_cli(["memory", "recall", "test", "query"], config)

            # Verify daemon was NOT started (since external API URL is set)
            assert mock_ensure_daemon.call_count == 0

            # Verify CLI was called
            assert mock_subprocess_run.call_count == 1
            call_args = mock_subprocess_run.call_args

            # Verify environment contains the external API URL
            assert call_args.kwargs["env"]["HINDSIGHT_API_URL"] == external_api_url

            # Verify exit code
            assert exit_code == 0

    def test_run_cli_without_external_api_url(self, config, mock_cli_binary, monkeypatch):
        """Test that without external API URL, daemon is started."""
        # Ensure HINDSIGHT_EMBED_API_URL is not set
        monkeypatch.delenv("HINDSIGHT_EMBED_API_URL", raising=False)

        # Mock functions
        mock_ensure_cli = Mock(return_value=True)
        mock_find_cli = Mock(return_value=mock_cli_binary)
        mock_ensure_daemon = Mock(return_value=True)
        mock_subprocess_run = Mock(return_value=Mock(returncode=0))

        with (
            patch.object(daemon_client, "ensure_cli_installed", mock_ensure_cli),
            patch.object(daemon_client, "find_cli_binary", mock_find_cli),
            patch.object(daemon_client, "ensure_daemon_running", mock_ensure_daemon),
            patch("subprocess.run", mock_subprocess_run),
        ):
            # Run CLI
            exit_code = daemon_client.run_cli(["memory", "recall", "test", "query"], config)

            # Verify daemon WAS started (since no external API URL)
            assert mock_ensure_daemon.call_count == 1
            assert mock_ensure_daemon.call_args[0][0] == config

            # Verify CLI was called
            assert mock_subprocess_run.call_count == 1
            call_args = mock_subprocess_run.call_args

            # Verify environment contains the local daemon URL
            assert call_args.kwargs["env"]["HINDSIGHT_API_URL"] == daemon_client.get_daemon_url()

            # Verify exit code
            assert exit_code == 0

    def test_run_cli_daemon_startup_failure(self, config, mock_cli_binary, monkeypatch):
        """Test that daemon startup failure is handled properly."""
        # Ensure HINDSIGHT_EMBED_API_URL is not set
        monkeypatch.delenv("HINDSIGHT_EMBED_API_URL", raising=False)

        # Mock functions - daemon startup fails
        mock_ensure_cli = Mock(return_value=True)
        mock_find_cli = Mock(return_value=mock_cli_binary)
        mock_ensure_daemon = Mock(return_value=False)  # Daemon fails to start

        with (
            patch.object(daemon_client, "ensure_cli_installed", mock_ensure_cli),
            patch.object(daemon_client, "find_cli_binary", mock_find_cli),
            patch.object(daemon_client, "ensure_daemon_running", mock_ensure_daemon),
        ):
            # Run CLI
            exit_code = daemon_client.run_cli(["memory", "recall", "test", "query"], config)

            # Verify daemon startup was attempted
            assert mock_ensure_daemon.call_count == 1

            # Verify exit code indicates failure
            assert exit_code == 1

    def test_run_cli_without_cli_binary(self, config, monkeypatch):
        """Test that missing CLI binary is handled properly."""
        # Ensure HINDSIGHT_EMBED_API_URL is not set
        monkeypatch.delenv("HINDSIGHT_EMBED_API_URL", raising=False)

        # Mock functions - CLI not installed
        mock_ensure_cli = Mock(return_value=True)
        mock_find_cli = Mock(return_value=None)  # CLI not found

        with (
            patch.object(daemon_client, "ensure_cli_installed", mock_ensure_cli),
            patch.object(daemon_client, "find_cli_binary", mock_find_cli),
        ):
            # Run CLI
            exit_code = daemon_client.run_cli(["memory", "recall", "test", "query"], config)

            # Verify exit code indicates failure
            assert exit_code == 1

    def test_run_cli_with_api_token(self, config, mock_cli_binary, monkeypatch):
        """Test that HINDSIGHT_EMBED_API_TOKEN is passed through to the CLI."""
        # Set up environment with external API URL and token
        external_api_url = "http://external-api:8000"
        api_token = "test-bearer-token-12345"
        monkeypatch.setenv("HINDSIGHT_EMBED_API_URL", external_api_url)
        monkeypatch.setenv("HINDSIGHT_EMBED_API_TOKEN", api_token)

        # Mock functions
        mock_ensure_cli = Mock(return_value=True)
        mock_find_cli = Mock(return_value=mock_cli_binary)
        mock_ensure_daemon = Mock(return_value=True)
        mock_subprocess_run = Mock(return_value=Mock(returncode=0))

        with (
            patch.object(daemon_client, "ensure_cli_installed", mock_ensure_cli),
            patch.object(daemon_client, "find_cli_binary", mock_find_cli),
            patch.object(daemon_client, "ensure_daemon_running", mock_ensure_daemon),
            patch("subprocess.run", mock_subprocess_run),
        ):
            # Run CLI
            exit_code = daemon_client.run_cli(["memory", "recall", "test", "query"], config)

            # Verify daemon was NOT started (since external API URL is set)
            assert mock_ensure_daemon.call_count == 0

            # Verify CLI was called
            assert mock_subprocess_run.call_count == 1
            call_args = mock_subprocess_run.call_args

            # Verify environment contains both the API URL and the API key
            assert call_args.kwargs["env"]["HINDSIGHT_API_URL"] == external_api_url
            assert call_args.kwargs["env"]["HINDSIGHT_API_KEY"] == api_token

            # Verify exit code
            assert exit_code == 0


class TestStartDaemon:
    """Tests for _start_daemon function."""

    def test_start_daemon_respects_database_url_env(self, config, monkeypatch):
        """Test that HINDSIGHT_EMBED_API_DATABASE_URL is respected if already set."""
        custom_db_url = "postgresql://custom:password@localhost:5432/custom_db"
        monkeypatch.setenv("HINDSIGHT_EMBED_API_DATABASE_URL", custom_db_url)

        # Mock subprocess.Popen to capture the env
        captured_env = {}

        def mock_popen(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            # Return a mock process
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            return mock_proc

        # Reduce timeout to 0.1s to avoid waiting
        monkeypatch.setattr(daemon_client, "DAEMON_STARTUP_TIMEOUT", 0.1)

        # Mock daemon health check to fail immediately (so we don't wait for startup)
        mock_is_running = Mock(return_value=False)

        with (
            patch("subprocess.Popen", side_effect=mock_popen),
            patch.object(daemon_client, "_is_daemon_running", mock_is_running),
            patch.object(daemon_client, "_find_hindsight_api_command", return_value=["fake-cmd"]),
        ):
            # Start daemon (will fail health check, but we just want to verify env)
            daemon_client._start_daemon(config)

            # Verify the custom database URL was NOT overwritten
            assert captured_env.get("HINDSIGHT_API_DATABASE_URL") == custom_db_url

    def test_start_daemon_sets_default_database_url(self, config, monkeypatch):
        """Test that default database URL is set if not already in env."""
        # Ensure HINDSIGHT_EMBED_API_DATABASE_URL is not set
        monkeypatch.delenv("HINDSIGHT_EMBED_API_DATABASE_URL", raising=False)

        # Mock subprocess.Popen to capture the env
        captured_env = {}

        def mock_popen(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            # Return a mock process
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            return mock_proc

        # Reduce timeout to 0.1s to avoid waiting
        monkeypatch.setattr(daemon_client, "DAEMON_STARTUP_TIMEOUT", 0.1)

        # Mock daemon health check to fail immediately (so we don't wait for startup)
        mock_is_running = Mock(return_value=False)

        with (
            patch("subprocess.Popen", side_effect=mock_popen),
            patch.object(daemon_client, "_is_daemon_running", mock_is_running),
            patch.object(daemon_client, "_find_hindsight_api_command", return_value=["fake-cmd"]),
        ):
            # Start daemon (will fail health check, but we just want to verify env)
            daemon_client._start_daemon(config)

            # Verify the default database URL was set
            assert captured_env.get("HINDSIGHT_API_DATABASE_URL") == "pg0://hindsight-embed"


class TestIsDaemonRunning:
    """Tests for _is_daemon_running function."""

    def test_daemon_running_returns_true_on_200(self):
        """Test that daemon is considered running when health check returns 200."""
        mock_response = Mock()
        mock_response.status_code = 200

        mock_client = MagicMock()
        mock_client.__enter__.return_value.get.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            assert daemon_client._is_daemon_running() is True

    def test_daemon_not_running_returns_false_on_error(self):
        """Test that daemon is considered not running when health check fails."""
        mock_client = MagicMock()
        mock_client.__enter__.return_value.get.side_effect = Exception("Connection refused")

        with patch("httpx.Client", return_value=mock_client):
            assert daemon_client._is_daemon_running() is False

    def test_daemon_not_running_returns_false_on_non_200(self):
        """Test that daemon is considered not running when health check returns non-200."""
        mock_response = Mock()
        mock_response.status_code = 500

        mock_client = MagicMock()
        mock_client.__enter__.return_value.get.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            assert daemon_client._is_daemon_running() is False

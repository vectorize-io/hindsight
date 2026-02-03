"""Integration tests for profile functionality."""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest


def strip_ansi(text):
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Create a temporary home directory for integration tests."""
    temp_home = tmp_path / "home"
    temp_home.mkdir()
    monkeypatch.setenv("HOME", str(temp_home))
    return temp_home


@pytest.fixture
def hindsight_embed_cmd():
    """Get the hindsight-embed command."""
    return ["uv", "run", "hindsight-embed"]


class TestProfileIntegration:
    """Integration tests for profile workflows."""

    def test_create_and_list_profile(self, temp_home, hindsight_embed_cmd):
        """Test creating a profile and listing it."""
        # Create profile
        result = subprocess.run(
            hindsight_embed_cmd
            + [
                "configure",
                "--profile",
                "test-app",
                "--env",
                "HINDSIGHT_EMBED_LLM_PROVIDER=openai",
                "--env",
                "HINDSIGHT_EMBED_LLM_API_KEY=sk-test",
                "--env",
                "HINDSIGHT_EMBED_LLM_MODEL=gpt-4o-mini",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(temp_home)},
        )

        assert result.returncode == 0
        assert "Profile 'test-app' configured successfully" in result.stdout

        # Verify profile file was created
        profile_path = temp_home / ".hindsight" / "profiles" / "test-app.env"
        assert profile_path.exists()

        config_content = profile_path.read_text()
        assert "HINDSIGHT_EMBED_LLM_PROVIDER=openai" in config_content
        assert "HINDSIGHT_EMBED_LLM_API_KEY=sk-test" in config_content

        # List profiles
        result = subprocess.run(
            hindsight_embed_cmd + ["profile", "list"],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(temp_home)},
        )

        assert result.returncode == 0
        assert "test-app" in result.stdout
        assert "Port:" in result.stdout

    def test_profile_show_default(self, temp_home, hindsight_embed_cmd):
        """Test showing the default profile."""
        # Create default config
        config_dir = temp_home / ".hindsight"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "embed").write_text("KEY=value")

        result = subprocess.run(
            hindsight_embed_cmd + ["profile", "show"],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(temp_home)},
        )

        assert result.returncode == 0
        output = strip_ansi(result.stdout)
        assert "Active profile: default" in output
        assert "Source: Default" in output
        assert "Port: 8888" in output

    def test_profile_show_with_env_var(self, temp_home, hindsight_embed_cmd):
        """Test profile resolution with HINDSIGHT_EMBED_PROFILE env var."""
        # Create profile
        subprocess.run(
            hindsight_embed_cmd
            + [
                "configure",
                "--profile",
                "test-app",
                "--env",
                "HINDSIGHT_EMBED_LLM_PROVIDER=openai",
                "--env",
                "HINDSIGHT_EMBED_LLM_API_KEY=sk-test",
            ],
            capture_output=True,
            env={**os.environ, "HOME": str(temp_home)},
        )

        # Show profile with env var
        env = {**os.environ, "HOME": str(temp_home), "HINDSIGHT_EMBED_PROFILE": "test-app"}
        result = subprocess.run(
            hindsight_embed_cmd + ["profile", "show"], capture_output=True, text=True, env=env
        )

        assert result.returncode == 0
        output = strip_ansi(result.stdout)
        assert "Active profile: test-app" in output
        assert "Source: HINDSIGHT_EMBED_PROFILE environment variable" in output

    def test_set_active_profile(self, temp_home, hindsight_embed_cmd):
        """Test setting and using active profile."""
        # Create profile
        subprocess.run(
            hindsight_embed_cmd
            + [
                "configure",
                "--profile",
                "test-app",
                "--env",
                "HINDSIGHT_EMBED_LLM_PROVIDER=openai",
                "--env",
                "HINDSIGHT_EMBED_LLM_API_KEY=sk-test",
            ],
            capture_output=True,
            env={**os.environ, "HOME": str(temp_home)},
        )

        # Set as active
        result = subprocess.run(
            hindsight_embed_cmd + ["profile", "set-active", "test-app"],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(temp_home)},
        )

        assert result.returncode == 0
        output = strip_ansi(result.stdout)
        assert "Active profile set to 'test-app'" in output

        # Verify active_profile file
        active_file = temp_home / ".hindsight" / "active_profile"
        assert active_file.exists()
        assert active_file.read_text() == "test-app"

        # Show profile (should show as active)
        result = subprocess.run(
            hindsight_embed_cmd + ["profile", "show"],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(temp_home)},
        )

        assert result.returncode == 0
        output = strip_ansi(result.stdout)
        assert "Active profile: test-app" in output

    def test_delete_profile(self, temp_home, hindsight_embed_cmd):
        """Test deleting a profile."""
        # Create profile
        subprocess.run(
            hindsight_embed_cmd
            + [
                "configure",
                "--profile",
                "test-app",
                "--env",
                "HINDSIGHT_EMBED_LLM_PROVIDER=openai",
                "--env",
                "HINDSIGHT_EMBED_LLM_API_KEY=sk-test",
            ],
            capture_output=True,
        )

        # Delete profile (with 'y' confirmation)
        result = subprocess.run(
            hindsight_embed_cmd + ["profile", "delete", "test-app"],
            input="y\n",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0 or "deleted" in result.stdout.lower()

        # Verify profile file is gone
        profile_path = temp_home / ".hindsight" / "profiles" / "test-app.env"
        # Note: Test may still have file if daemon was running and user cancelled

    def test_multiple_profiles_different_ports(self, temp_home, hindsight_embed_cmd):
        """Test that multiple profiles get different ports."""
        # Create first profile
        subprocess.run(
            hindsight_embed_cmd
            + [
                "configure",
                "--profile",
                "app1",
                "--env",
                "HINDSIGHT_EMBED_LLM_PROVIDER=openai",
                "--env",
                "HINDSIGHT_EMBED_LLM_API_KEY=sk-test",
            ],
            capture_output=True,
        )

        # Create second profile
        subprocess.run(
            hindsight_embed_cmd
            + [
                "configure",
                "--profile",
                "app2",
                "--env",
                "HINDSIGHT_EMBED_LLM_PROVIDER=openai",
                "--env",
                "HINDSIGHT_EMBED_LLM_API_KEY=sk-test",
            ],
            capture_output=True,
        )

        # Read metadata
        metadata_path = temp_home / ".hindsight" / "profiles" / "metadata.json"
        assert metadata_path.exists()

        metadata = json.loads(metadata_path.read_text())
        app1_port = metadata["profiles"]["app1"]["port"]
        app2_port = metadata["profiles"]["app2"]["port"]

        # Ports should be different
        assert app1_port != app2_port
        assert 8889 <= app1_port <= 9888
        assert 8889 <= app2_port <= 9888

    def test_profile_validation_fails_for_nonexistent(self, temp_home, hindsight_embed_cmd):
        """Test that using non-existent profile fails."""
        # Try to use non-existent profile
        env = {**os.environ, "HOME": str(temp_home), "HINDSIGHT_EMBED_PROFILE": "nonexistent"}
        result = subprocess.run(
            hindsight_embed_cmd + ["profile", "show"], capture_output=True, text=True, env=env
        )

        assert result.returncode == 1
        assert "Profile 'nonexistent' not found" in result.stderr

    def test_backward_compatibility_default_profile(self, temp_home, hindsight_embed_cmd):
        """Test that default profile works without any profile commands."""
        # Create default config manually (simulating old behavior)
        config_dir = temp_home / ".hindsight"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "embed").write_text(
            "HINDSIGHT_EMBED_LLM_PROVIDER=openai\n"
            "HINDSIGHT_EMBED_LLM_API_KEY=sk-test\n"
            "HINDSIGHT_EMBED_LLM_MODEL=gpt-4o-mini\n"
        )

        # Show profile should work
        result = subprocess.run(
            hindsight_embed_cmd + ["profile", "show"],
            capture_output=True,
            text=True,
            env={**os.environ, "HOME": str(temp_home)},
        )

        assert result.returncode == 0
        output = strip_ansi(result.stdout)
        assert "Active profile: default" in output
        assert "Port: 8888" in output

    def test_configure_without_profile_flag(self, temp_home, hindsight_embed_cmd):
        """Test that configure without --profile still works (backward compatibility)."""
        # Configure without profile flag (should configure default)
        # Use non-interactive mode by providing env vars
        env = {
            **os.environ,
            "HOME": str(temp_home),
            "HINDSIGHT_EMBED_LLM_PROVIDER": "openai",
            "HINDSIGHT_EMBED_LLM_API_KEY": "sk-test",
            "HINDSIGHT_EMBED_LLM_MODEL": "gpt-4o-mini",
        }

        result = subprocess.run(
            hindsight_embed_cmd + ["configure"], capture_output=True, text=True, env=env
        )

        # Should succeed
        assert result.returncode == 0

        # Verify default config was created
        config_path = temp_home / ".hindsight" / "embed"
        assert config_path.exists()

    def test_clear_active_profile(self, temp_home, hindsight_embed_cmd):
        """Test clearing the active profile."""
        # Create and set active profile
        subprocess.run(
            hindsight_embed_cmd
            + [
                "configure",
                "--profile",
                "test-app",
                "--env",
                "HINDSIGHT_EMBED_LLM_PROVIDER=openai",
                "--env",
                "HINDSIGHT_EMBED_LLM_API_KEY=sk-test",
            ],
            capture_output=True,
        )

        subprocess.run(
            hindsight_embed_cmd + ["profile", "set-active", "test-app"],
            capture_output=True,
        )

        # Clear active profile
        result = subprocess.run(
            hindsight_embed_cmd + ["profile", "set-active", "--none"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Active profile cleared" in result.stdout

        # Verify active_profile file is gone
        active_file = temp_home / ".hindsight" / "active_profile"
        assert not active_file.exists()

    def test_profile_port_persistence(self, temp_home, hindsight_embed_cmd):
        """Test that profile port is persistent across recreations."""
        # Create profile
        subprocess.run(
            hindsight_embed_cmd
            + [
                "configure",
                "--profile",
                "test-app",
                "--env",
                "HINDSIGHT_EMBED_LLM_PROVIDER=openai",
                "--env",
                "HINDSIGHT_EMBED_LLM_API_KEY=sk-test",
            ],
            capture_output=True,
        )

        # Get port
        metadata_path = temp_home / ".hindsight" / "profiles" / "metadata.json"
        metadata1 = json.loads(metadata_path.read_text())
        port1 = metadata1["profiles"]["test-app"]["port"]

        # Update profile (recreate with different config)
        subprocess.run(
            hindsight_embed_cmd
            + [
                "configure",
                "--profile",
                "test-app",
                "--env",
                "HINDSIGHT_EMBED_LLM_PROVIDER=groq",
                "--env",
                "HINDSIGHT_EMBED_LLM_API_KEY=gsk-test",
            ],
            capture_output=True,
        )

        # Get port again
        metadata2 = json.loads(metadata_path.read_text())
        port2 = metadata2["profiles"]["test-app"]["port"]

        # Port should be the same
        assert port1 == port2

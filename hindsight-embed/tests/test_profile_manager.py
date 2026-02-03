"""Tests for profile_manager module."""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from hindsight_embed.profile_manager import (
    CONFIG_DIR,
    PROFILES_DIR,
    ProfileInfo,
    ProfileManager,
    ProfilePaths,
    resolve_active_profile,
    validate_profile_exists,
)


@pytest.fixture
def temp_hindsight_dir(tmp_path, monkeypatch):
    """Create a temporary hindsight directory for tests."""
    temp_config = tmp_path / ".hindsight"
    temp_config.mkdir()
    monkeypatch.setattr("hindsight_embed.profile_manager.CONFIG_DIR", temp_config)
    monkeypatch.setattr("hindsight_embed.profile_manager.PROFILES_DIR", temp_config / "profiles")
    monkeypatch.setattr(
        "hindsight_embed.profile_manager.METADATA_FILE", temp_config / "profiles" / "metadata.json"
    )
    monkeypatch.setattr("hindsight_embed.profile_manager.ACTIVE_PROFILE_FILE", temp_config / "active_profile")
    return temp_config


@pytest.fixture
def profile_manager(temp_hindsight_dir):
    """Create a ProfileManager with temp directory."""
    return ProfileManager()


class TestProfileManager:
    """Tests for ProfileManager class."""

    def test_create_profile_success(self, profile_manager, temp_hindsight_dir):
        """Test creating a new profile."""
        config = {
            "HINDSIGHT_EMBED_LLM_PROVIDER": "openai",
            "HINDSIGHT_EMBED_LLM_API_KEY": "sk-test",
            "HINDSIGHT_EMBED_LLM_MODEL": "gpt-4o-mini",
        }

        profile_manager.create_profile("test-profile", config)

        # Verify config file was created
        config_path = temp_hindsight_dir / "profiles" / "test-profile.env"
        assert config_path.exists()

        # Verify config contents
        config_content = config_path.read_text()
        assert "HINDSIGHT_EMBED_LLM_PROVIDER=openai" in config_content
        assert "HINDSIGHT_EMBED_LLM_API_KEY=sk-test" in config_content
        assert "HINDSIGHT_EMBED_LLM_MODEL=gpt-4o-mini" in config_content

        # Verify metadata was created
        metadata_path = temp_hindsight_dir / "profiles" / "metadata.json"
        assert metadata_path.exists()

        metadata = json.loads(metadata_path.read_text())
        assert "test-profile" in metadata["profiles"]
        assert "port" in metadata["profiles"]["test-profile"]
        assert "created_at" in metadata["profiles"]["test-profile"]

    def test_create_profile_invalid_name(self, profile_manager):
        """Test creating profile with invalid name fails."""
        config = {"KEY": "value"}

        # Empty name
        with pytest.raises(ValueError, match="Profile name cannot be empty"):
            profile_manager.create_profile("", config)

        # Invalid characters
        with pytest.raises(ValueError, match="Invalid profile name"):
            profile_manager.create_profile("test profile", config)

        with pytest.raises(ValueError, match="Invalid profile name"):
            profile_manager.create_profile("test@profile", config)

    def test_profile_exists(self, profile_manager, temp_hindsight_dir):
        """Test checking if a profile exists."""
        # Default profile doesn't exist initially
        assert not profile_manager.profile_exists("")

        # Create default config
        (temp_hindsight_dir / "embed").write_text("KEY=value")
        assert profile_manager.profile_exists("")

        # Named profile doesn't exist
        assert not profile_manager.profile_exists("test")

        # Create named profile
        profile_manager.create_profile("test", {"KEY": "value"})
        assert profile_manager.profile_exists("test")

    def test_delete_profile(self, profile_manager, temp_hindsight_dir):
        """Test deleting a profile."""
        # Create profile
        profile_manager.create_profile("test-profile", {"KEY": "value"})
        assert profile_manager.profile_exists("test-profile")

        # Delete profile
        profile_manager.delete_profile("test-profile")
        assert not profile_manager.profile_exists("test-profile")

        # Verify all files are removed
        profiles_dir = temp_hindsight_dir / "profiles"
        assert not (profiles_dir / "test-profile.env").exists()
        assert not (profiles_dir / "test-profile.lock").exists()
        assert not (profiles_dir / "test-profile.log").exists()

        # Verify metadata no longer contains profile
        metadata_path = profiles_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            assert "test-profile" not in metadata.get("profiles", {})

    def test_delete_nonexistent_profile(self, profile_manager):
        """Test deleting a non-existent profile fails."""
        with pytest.raises(ValueError, match="does not exist"):
            profile_manager.delete_profile("nonexistent")

    def test_delete_default_profile_fails(self, profile_manager):
        """Test deleting default profile fails."""
        with pytest.raises(ValueError, match="Cannot delete default profile"):
            profile_manager.delete_profile("")

    def test_set_active_profile(self, profile_manager, temp_hindsight_dir):
        """Test setting active profile."""
        # Create a profile
        profile_manager.create_profile("test-profile", {"KEY": "value"})

        # Set as active
        profile_manager.set_active_profile("test-profile")

        # Verify active profile file was created
        active_file = temp_hindsight_dir / "active_profile"
        assert active_file.exists()
        assert active_file.read_text() == "test-profile"

        # Get active profile
        assert profile_manager.get_active_profile() == "test-profile"

    def test_clear_active_profile(self, profile_manager, temp_hindsight_dir):
        """Test clearing active profile."""
        # Create and set active profile
        profile_manager.create_profile("test-profile", {"KEY": "value"})
        profile_manager.set_active_profile("test-profile")
        assert profile_manager.get_active_profile() == "test-profile"

        # Clear active profile
        profile_manager.set_active_profile(None)
        assert profile_manager.get_active_profile() == ""

        # Verify file was removed
        active_file = temp_hindsight_dir / "active_profile"
        assert not active_file.exists()

    def test_set_active_nonexistent_profile_fails(self, profile_manager):
        """Test setting non-existent profile as active fails."""
        with pytest.raises(ValueError, match="does not exist"):
            profile_manager.set_active_profile("nonexistent")

    def test_list_profiles(self, profile_manager, temp_hindsight_dir):
        """Test listing profiles."""
        # Initially no profiles
        profiles = profile_manager.list_profiles()
        assert len(profiles) == 0

        # Create default config
        (temp_hindsight_dir / "embed").write_text("KEY=value")

        # List profiles - should show default
        profiles = profile_manager.list_profiles()
        assert len(profiles) == 1
        assert profiles[0].name == ""
        assert profiles[0].port == 8888

        # Create named profiles
        profile_manager.create_profile("profile1", {"KEY": "value"})
        profile_manager.create_profile("profile2", {"KEY": "value"})

        # List all profiles
        profiles = profile_manager.list_profiles()
        assert len(profiles) == 3

        # Verify sorting (default first, then alphabetical)
        assert profiles[0].name == ""
        assert profiles[1].name == "profile1"
        assert profiles[2].name == "profile2"

    def test_list_profiles_with_active(self, profile_manager, temp_hindsight_dir):
        """Test listing profiles shows active status."""
        profile_manager.create_profile("profile1", {"KEY": "value"})
        profile_manager.create_profile("profile2", {"KEY": "value"})

        # Set profile1 as active
        profile_manager.set_active_profile("profile1")

        # List profiles
        profiles = profile_manager.list_profiles()

        # Find profile1
        profile1 = next(p for p in profiles if p.name == "profile1")
        profile2 = next(p for p in profiles if p.name == "profile2")

        assert profile1.is_active is True
        assert profile2.is_active is False

    def test_resolve_profile_paths_default(self, profile_manager, temp_hindsight_dir):
        """Test resolving paths for default profile."""
        paths = profile_manager.resolve_profile_paths("")

        assert paths.config == temp_hindsight_dir / "embed"
        assert paths.lock == temp_hindsight_dir / "daemon.lock"
        assert paths.log == temp_hindsight_dir / "daemon.log"
        assert paths.port == 8888

    def test_resolve_profile_paths_named(self, profile_manager, temp_hindsight_dir):
        """Test resolving paths for named profile."""
        # Create profile first
        profile_manager.create_profile("test-profile", {"KEY": "value"})

        paths = profile_manager.resolve_profile_paths("test-profile")

        assert paths.config == temp_hindsight_dir / "profiles" / "test-profile.env"
        assert paths.lock == temp_hindsight_dir / "profiles" / "test-profile.lock"
        assert paths.log == temp_hindsight_dir / "profiles" / "test-profile.log"
        assert 8889 <= paths.port <= 9888  # Port in valid range

    def test_port_allocation_deterministic(self, profile_manager):
        """Test that port allocation is deterministic for same profile name."""
        profile_manager.create_profile("test1", {"KEY": "value"})
        paths1 = profile_manager.resolve_profile_paths("test1")

        # Delete and recreate
        profile_manager.delete_profile("test1")
        profile_manager.create_profile("test1", {"KEY": "value"})
        paths2 = profile_manager.resolve_profile_paths("test1")

        # Port should be the same
        assert paths1.port == paths2.port

    def test_port_allocation_unique(self, profile_manager):
        """Test that different profiles get different ports."""
        profile_manager.create_profile("profile1", {"KEY": "value"})
        profile_manager.create_profile("profile2", {"KEY": "value"})

        paths1 = profile_manager.resolve_profile_paths("profile1")
        paths2 = profile_manager.resolve_profile_paths("profile2")

        # Ports should be different
        assert paths1.port != paths2.port

    def test_daemon_running_status(self, profile_manager):
        """Test that daemon running status is checked correctly."""
        with patch("httpx.Client") as mock_client:
            # Mock successful health check
            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response

            profile_manager.create_profile("test", {"KEY": "value"})
            profiles = profile_manager.list_profiles()

            test_profile = next(p for p in profiles if p.name == "test")
            assert test_profile.daemon_running is True

    def test_update_existing_profile(self, profile_manager, temp_hindsight_dir):
        """Test updating an existing profile."""
        # Create profile
        profile_manager.create_profile("test", {"KEY": "old_value"})

        # Update profile
        profile_manager.create_profile("test", {"KEY": "new_value"})

        # Verify config was updated
        config_path = temp_hindsight_dir / "profiles" / "test.env"
        config_content = config_path.read_text()
        assert "KEY=new_value" in config_content
        assert "KEY=old_value" not in config_content

    def test_metadata_persistence(self, profile_manager, temp_hindsight_dir):
        """Test that metadata persists across ProfileManager instances."""
        # Create profile with first manager
        profile_manager.create_profile("test", {"KEY": "value"})

        # Create new manager and verify it sees the profile
        new_manager = ProfileManager()
        assert new_manager.profile_exists("test")

        profiles = new_manager.list_profiles()
        test_profile = next(p for p in profiles if p.name == "test")
        assert test_profile.port > 0

    def test_delete_active_profile_clears_active_file(self, profile_manager, temp_hindsight_dir):
        """Test that deleting active profile clears the active_profile file."""
        profile_manager.create_profile("test", {"KEY": "value"})
        profile_manager.set_active_profile("test")

        # Verify active
        assert profile_manager.get_active_profile() == "test"

        # Delete profile
        profile_manager.delete_profile("test")

        # Active profile should be cleared
        assert profile_manager.get_active_profile() == ""


class TestResolveActiveProfile:
    """Tests for resolve_active_profile function."""

    def test_priority_env_var(self, monkeypatch, profile_manager):
        """Test that HINDSIGHT_EMBED_PROFILE env var has highest priority."""
        # Set up all possible sources
        monkeypatch.setenv("HINDSIGHT_EMBED_PROFILE", "from-env")

        profile_manager.create_profile("from-file", {"KEY": "value"})
        profile_manager.set_active_profile("from-file")

        # Import cli to set override
        from hindsight_embed import cli

        cli.set_cli_profile_override("from-flag")

        # Env var should win
        assert resolve_active_profile() == "from-env"

    def test_priority_cli_flag(self, monkeypatch, profile_manager):
        """Test that CLI flag has second highest priority."""
        # No env var
        monkeypatch.delenv("HINDSIGHT_EMBED_PROFILE", raising=False)

        profile_manager.create_profile("from-file", {"KEY": "value"})
        profile_manager.set_active_profile("from-file")

        # Import cli to set override
        from hindsight_embed import cli

        cli.set_cli_profile_override("from-flag")

        # CLI flag should win
        assert resolve_active_profile() == "from-flag"

    def test_priority_active_file(self, monkeypatch, profile_manager):
        """Test that active file has third priority."""
        # No env var or CLI flag
        monkeypatch.delenv("HINDSIGHT_EMBED_PROFILE", raising=False)

        from hindsight_embed import cli

        cli.set_cli_profile_override(None)

        profile_manager.create_profile("from-file", {"KEY": "value"})
        profile_manager.set_active_profile("from-file")

        # Active file should be used
        assert resolve_active_profile() == "from-file"

    def test_priority_default(self, monkeypatch, profile_manager):
        """Test that default is used when no sources are set."""
        # No env var, CLI flag, or active file
        monkeypatch.delenv("HINDSIGHT_EMBED_PROFILE", raising=False)

        from hindsight_embed import cli

        cli.set_cli_profile_override(None)

        # Default should be used
        assert resolve_active_profile() == ""


class TestValidateProfileExists:
    """Tests for validate_profile_exists function."""

    def test_validate_default_profile_always_passes(self):
        """Test that default profile always passes validation."""
        # Default profile (empty string) should never fail
        validate_profile_exists("")  # Should not raise

    def test_validate_existing_profile_passes(self, profile_manager):
        """Test that existing profile passes validation."""
        profile_manager.create_profile("test", {"KEY": "value"})
        validate_profile_exists("test")  # Should not raise

    def test_validate_nonexistent_profile_fails(self, profile_manager, capsys):
        """Test that non-existent profile fails validation."""
        with pytest.raises(SystemExit) as exc_info:
            validate_profile_exists("nonexistent")

        assert exc_info.value.code == 1

        # Check error message
        captured = capsys.readouterr()
        assert "Profile 'nonexistent' not found" in captured.err
        assert "hindsight-embed configure --profile nonexistent" in captured.err
